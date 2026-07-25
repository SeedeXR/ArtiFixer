# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-aware job scheduler and stage runner.

Design:

* One asyncio worker task per GPU. A stage is a full CUDA process, so admitting
  more concurrent jobs than devices would only cause OOM and thrash; the queue is
  the admission control.
* Each worker pins its child processes with ``CUDA_VISIBLE_DEVICES``, so two jobs
  never contend for the same device.
* Stages run as subprocesses with ``exec`` semantics (argv list, no shell) in their
  own session, which makes both timeout and cancellation a process-group signal
  rather than a best-effort kill of one pid.
* Stage output is streamed to a per-stage log file while a bounded tail is kept in
  memory for error reporting and for parsing the handful of ``key=value`` lines the
  repo's CLIs print.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splat_api.app import pipeline
from splat_api.app.config import Settings
from splat_api.app.errors import Conflict, NotFound, ServiceUnavailable
from splat_api.app.jobstore import (
    ArtifactRecord,
    AsyncJobStore,
    JobRecord,
    StageRecord,
    public_request,
    utc_now,
)
from splat_api.app.paths import is_within, safe_join

logger = logging.getLogger("splat_api.scheduler")

LOG_TAIL_LINES = 400
MAX_HASH_BYTES = 8 * 1024**3
TERMINATE_GRACE_SECONDS = 20.0
STREAM_READ_CHUNK = 64 * 1024
# Cap on a single unbroken run of output kept in the tail, so a child that emits
# megabytes without a line break cannot grow our memory.
MAX_PENDING_LINE_BYTES = 16 * 1024
DRAIN_FLUSH_SECONDS = 30.0
# Progress bars use CR; treat both terminators as line breaks.
_LINE_BREAK = re.compile(rb"[\r\n]")
# Cap what we echo back to a caller: stage logs can contain long tracebacks, and
# the error field is meant to be actionable, not a transcript.
ERROR_EXCERPT_LINES = 12


class StageFailure(RuntimeError):
    """A stage exited non-zero, timed out, or produced no usable output."""

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class RunningProcess:
    process: asyncio.subprocess.Process
    stage_name: str


class Scheduler:
    """Owns the queue, the workers, and the lifecycle of every job directory."""

    def __init__(self, settings: Settings, store: AsyncJobStore) -> None:
        self._settings = settings
        self._store = store
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.queue_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._running: dict[str, RunningProcess] = {}
        self._cancelled: set[str] = set()
        self._queued_order: deque[str] = deque()
        self._cache_lock = asyncio.Lock()
        self._shutting_down = False

    # ---- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        worker_count = self._settings.max_concurrent_jobs
        devices = self._settings.cuda_devices
        for index in range(worker_count):
            gpu_index = devices[index] if index < len(devices) else None
            self._workers.append(asyncio.create_task(self._worker(index, gpu_index), name=f"worker-{index}"))
        await self._readmit_pending()
        logger.info(
            "scheduler started with %d worker(s), devices=%s",
            worker_count,
            list(devices) or "cpu-only",
        )

    async def stop(self) -> None:
        self._shutting_down = True
        for task in self._workers:
            task.cancel()
        for job_id, running in list(self._running.items()):
            logger.warning("terminating stage %s of job %s for shutdown", running.stage_name, job_id)
            await self._terminate(running.process)
        for task in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._workers.clear()

    async def _readmit_pending(self) -> None:
        """Re-queue jobs left behind by a restart.

        Safe to re-run: `prepare_colmap_artifixer_inputs`, `run_inference` and
        `run_artifixer3d` all skip work whose outputs already exist, so an
        interrupted job resumes at the stage it died in.
        """
        pending = await asyncio.to_thread(self._store.sync.pending_jobs_in_order)
        for job in pending:
            if job.state == "running":
                job.restarts += 1
                job.state = "queued"
                job.gpu_index = None
                for stage in job.stages:
                    if stage.state == "running":
                        stage.state = "pending"
                        stage.started_at = None
                        stage.message = "Interrupted by a service restart; will be retried"
                await self._store.update_job(job)
            try:
                self._queue.put_nowait(job.job_id)
                self._queued_order.append(job.job_id)
            except asyncio.QueueFull:
                logger.error("queue full while re-admitting job %s", job.job_id)
                break
        if pending:
            logger.info("re-admitted %d pending job(s)", len(pending))

    # ---- submission ---------------------------------------------------

    async def submit(self, job: JobRecord) -> None:
        if self._shutting_down:
            raise ServiceUnavailable("Service is shutting down")
        try:
            self._queue.put_nowait(job.job_id)
        except asyncio.QueueFull as exc:
            raise ServiceUnavailable(
                f"Job queue is full ({self._settings.queue_capacity} waiting). Retry shortly."
            ) from exc
        self._queued_order.append(job.job_id)

    def queue_position(self, job_id: str) -> int | None:
        if job_id in self._running:
            return 0
        try:
            return self._queued_order.index(job_id) + 1
        except ValueError:
            return None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def cancel(self, job_id: str) -> JobRecord:
        job = await self._store.get_job(job_id)
        if job.state in ("succeeded", "failed", "cancelled"):
            raise Conflict(f"Job {job_id} already finished with state {job.state}")
        self._cancelled.add(job_id)
        running = self._running.get(job_id)
        if running is not None:
            await self._terminate(running.process)
        else:
            # Queued, or between stages. Mark terminal through a read-modify-write
            # under the store lock: the worker may be inside _finalize_success right
            # now (its process is already reaped, so _running no longer has an
            # entry), and a blind write of this stale snapshot would drop the
            # artifacts it just published.
            job = await self._store.cancel_if_active(job_id)
        return job

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Stop a stage and everything it spawned.

        Signals the process group (the child runs with ``start_new_session=True``)
        because torchrun/dataloader workers are children of the stage process and
        would otherwise survive and keep holding the GPU.
        """
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                await process.wait()

    # ---- worker loop --------------------------------------------------

    async def _worker(self, worker_index: int, gpu_index: int | None) -> None:
        while True:
            job_id = await self._queue.get()
            with contextlib.suppress(ValueError):
                self._queued_order.remove(job_id)
            try:
                if job_id in self._cancelled:
                    self._cancelled.discard(job_id)
                    continue
                await self._run_job(job_id, gpu_index)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker %d crashed handling job %s", worker_index, job_id)
                with contextlib.suppress(Exception):
                    job = await self._store.get_job(job_id)
                    if job.state not in ("succeeded", "cancelled"):
                        job.state = "failed"
                        job.error = "Internal scheduler error; see server logs"
                        job.finished_at = utc_now()
                        await self._store.update_job(job)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str, gpu_index: int | None) -> None:
        job = await self._store.get_job(job_id)
        settings = self._settings
        paths = self._job_paths(job)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        job.state = "running"
        job.started_at = job.started_at or utc_now()
        job.gpu_index = gpu_index
        await self._store.update_job(job)
        logger.info("job %s started on gpu=%s mode=%s", job_id, gpu_index, job.mode)

        request = job.request
        has_trajectory = bool(request.get("trajectory"))
        # Always present: even when the caller named no subset, the API holds views
        # back so 3DGRUT's validation split is non-empty (see routes.py).
        has_selected = bool(request.get("selected_image_names"))
        env = pipeline.build_subprocess_env(settings, gpu_index=gpu_index, base_env=dict(os.environ))

        try:
            cached_checkpoint = await self._cached_reconstruction(job)
            # When the base reconstruction is reused, `prepare` never writes a
            # job-local checkpoint, so every later reference (export, warm start)
            # must resolve to the cached file instead.
            job.metrics["reconstruction_checkpoint"] = str(
                cached_checkpoint or paths.reconstruction_checkpoint
            )
            commands: dict[str, pipeline.StageCommand] = {
                pipeline.STAGE_PREPARE: pipeline.prepare_command(
                    settings,
                    paths,
                    mode=job.mode,
                    has_selected_names=has_selected,
                    has_trajectory=has_trajectory,
                    metric_scale=request.get("metric_scale"),
                    cached_checkpoint=cached_checkpoint,
                )
            }

            for stage_name in [stage.name for stage in job.stages]:
                if job_id in self._cancelled:
                    raise StageFailure("Cancelled")
                stage = job.stage(stage_name)
                if stage_name == pipeline.STAGE_PREPARE:
                    command = commands[pipeline.STAGE_PREPARE]
                elif stage_name == pipeline.STAGE_ARTIFIXER:
                    command = pipeline.artifixer_command(
                        settings,
                        paths,
                        has_trajectory=has_trajectory,
                        inference_steps=int(request.get("inference_steps", 4)),
                    )
                elif stage_name == pipeline.STAGE_ARTIFIXER3D:
                    frames_dir = self._resolve_frames_dir(job, paths, has_trajectory=has_trajectory)
                    command = pipeline.artifixer3d_command(
                        settings,
                        paths,
                        frames_dir=frames_dir,
                        # Upstream trains ArtiFixer3D from scratch by default
                        # (README: "ArtiFixer3D trains a fresh 3DGRUT
                        # optimization by default"); warm-starting is opt-in.
                        base_checkpoint=(
                            self._reconstruction_checkpoint(job, paths)
                            if settings.artifixer3d_warm_start
                            else None
                        ),
                    )
                elif stage_name == pipeline.STAGE_ARTIFIXER3D_PLUS:
                    command = pipeline.artifixer3d_plus_command(
                        settings,
                        paths,
                        has_trajectory=has_trajectory,
                        inference_steps=int(request.get("inference_steps", 4)),
                    )
                elif stage_name == pipeline.STAGE_EXPORT:
                    command = pipeline.export_command(
                        settings,
                        checkpoint=self._final_checkpoint(job, paths),
                        output=paths.splat_ply_path,
                        stats_path=paths.output_dir / "splat_stats.json",
                    )
                else:  # pragma: no cover - stage_sequence controls this set
                    raise StageFailure(f"Unknown stage {stage_name!r}")

                await self._run_stage(job, stage, command, paths=paths, env=env)

            await self._finalize_success(job, paths)
        except StageFailure as exc:
            await self._finalize_failure(job, exc)
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.error = "Cancelled during shutdown"
            job.finished_at = utc_now()
            await self._store.update_job(job)
            raise
        finally:
            self._cancelled.discard(job_id)
            self._running.pop(job_id, None)

    # ---- stage execution ---------------------------------------------

    def _job_dir(self, job: JobRecord) -> Path:
        return safe_join(self._settings.jobs_dir, job.job_id)

    def _job_paths(self, job: JobRecord) -> pipeline.JobPaths:
        job_dir = self._job_dir(job)
        scene = job.request["scene_root"]
        return pipeline.JobPaths(
            job_dir=job_dir,
            scene_dir=Path(scene),
            scene_id=job.scene_id,
            reconstruction_steps=int(job.request["reconstruction_steps"]),
            artifixer3d_steps=int(job.request["artifixer3d_steps"]),
        )

    async def _run_stage(
        self,
        job: JobRecord,
        stage: StageRecord,
        command: pipeline.StageCommand,
        *,
        paths: pipeline.JobPaths,
        env: dict[str, str],
    ) -> None:
        if stage.state in ("succeeded", "skipped"):
            return
        log_path = paths.logs_dir / f"{stage.name}.log"
        stage.state = "running"
        stage.started_at = utc_now()
        stage.command = command.display()
        stage.description = command.description
        await self._store.update_job(job)
        logger.info("job %s stage %s: %s", job.job_id, stage.name, stage.command)

        started = time.monotonic()
        stage_env = dict(env)
        stage_env.update(command.env)
        try:
            exit_code, tail = await self._spawn(
                job, stage.name, command.argv, log_path=log_path, env=stage_env
            )
        except asyncio.TimeoutError:
            stage.state = "failed"
            stage.finished_at = utc_now()
            stage.duration_seconds = round(time.monotonic() - started, 3)
            stage.message = f"Timed out after {self._settings.stage_timeout_seconds}s"
            await self._store.update_job(job)
            raise StageFailure(f"Stage {stage.name!r} timed out") from None

        stage.duration_seconds = round(time.monotonic() - started, 3)
        stage.finished_at = utc_now()
        stage.exit_code = exit_code
        job.metrics.setdefault("stage_seconds", {})[stage.name] = stage.duration_seconds

        if job.job_id in self._cancelled:
            stage.state = "cancelled"
            await self._store.update_job(job)
            raise StageFailure("Cancelled")

        if exit_code != 0:
            stage.state = "failed"
            excerpt = "\n".join(tail[-ERROR_EXCERPT_LINES:])
            stage.message = f"Exited with code {exit_code}"
            await self._store.update_job(job)
            raise StageFailure(
                f"Stage {stage.name!r} failed with exit code {exit_code}:\n{excerpt}", exit_code=exit_code
            )

        stage.state = "succeeded"
        self._absorb_stage_output(job, stage.name, tail)
        await self._store.update_job(job)

    async def _spawn(
        self,
        job: JobRecord,
        stage_name: str,
        argv: tuple[str, ...],
        *,
        log_path: Path,
        env: dict[str, str],
    ) -> tuple[int, list[str]]:
        """Run ``argv``, tee output to ``log_path``, return exit code and tail."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._settings.repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._running[job.job_id] = RunningProcess(process=process, stage_name=stage_name)
        tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)

        async def drain() -> None:
            """Copy child output to the log and keep a bounded tail.

            Reads fixed-size chunks rather than lines. ``StreamReader.readline``
            raises ``ValueError`` once a "line" passes the 64 KiB stream limit, and
            the repo's CLIs emit carriage-return progress bars that never contain a
            newline. That exception would kill this task, leave the pipe unread
            until the child blocked on write, and then make ``process.wait()``
            unresolvable — a permanently wedged worker. Chunked reads cannot raise
            it. Splitting on CR as well as LF also turns progress updates into
            discrete tail entries instead of one unbounded string.
            """
            assert process.stdout is not None
            pending = bytearray()
            with log_path.open("ab") as sink:
                sink.write(f"$ {' '.join(argv)}\n".encode())
                while True:
                    chunk = await process.stdout.read(STREAM_READ_CHUNK)
                    if not chunk:
                        break
                    sink.write(chunk)
                    sink.flush()
                    pending.extend(chunk)
                    *complete, remainder = _LINE_BREAK.split(bytes(pending))
                    pending = bytearray(remainder)
                    for part in complete:
                        text = part.decode("utf-8", errors="replace").rstrip()
                        if text:
                            tail.append(text)
                    if len(pending) > MAX_PENDING_LINE_BYTES:
                        # An unbroken run of output longer than we will ever need:
                        # keep the newest bytes so the tail stays useful and bounded.
                        tail.append(
                            bytes(pending[-MAX_PENDING_LINE_BYTES:]).decode("utf-8", errors="replace")
                        )
                        pending.clear()
                if pending:
                    text = bytes(pending).decode("utf-8", errors="replace").rstrip()
                    if text:
                        tail.append(text)

        drain_task = asyncio.create_task(drain())

        async def finish_drain() -> None:
            # Never let a drain problem mask the stage result, and never leave the
            # task holding an open log handle.
            if not drain_task.done():
                drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await drain_task

        try:
            await asyncio.wait_for(process.wait(), timeout=self._settings.stage_timeout_seconds)
        except asyncio.TimeoutError:
            await self._terminate(process)
            await finish_drain()
            raise
        except asyncio.CancelledError:
            await self._terminate(process)
            await finish_drain()
            raise
        finally:
            self._running.pop(job.job_id, None)

        # The child has exited, so the pipe is at EOF and drain finishes promptly.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(drain_task), timeout=DRAIN_FLUSH_SECONDS)
        await finish_drain()
        return int(process.returncode or 0), list(tail)

    @staticmethod
    def _parse_key_values(lines: list[str]) -> dict[str, str]:
        """Collect the ``key=value`` lines the repo's CLIs print on success."""
        parsed: dict[str, str] = {}
        for line in lines:
            if "=" not in line or line.startswith(("$", " ")):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.isidentifier():
                parsed[key] = value.strip()
        return parsed

    def _absorb_stage_output(self, job: JobRecord, stage_name: str, tail: list[str]) -> None:
        parsed = self._parse_key_values(tail)
        stage_metrics = job.metrics.setdefault("stage_outputs", {})
        interesting = {
            key: value
            for key, value in parsed.items()
            if key
            in {
                "metric_scale",
                "camera_scale",
                "selected_views",
                "num_gaussians",
                "sh_degree",
                "ply_bytes",
                "artifixer3d_checkpoint",
                "artifixer3d_render_dir",
                "artifixer3d_plus_inference_split",
            }
        }
        if interesting:
            stage_metrics[stage_name] = interesting
        # run_inference.py:811 prints the canonical output directory; prefer it
        # over our computed path so a change upstream cannot silently mislead us.
        for line in tail:
            if line.startswith("Writing outputs to "):
                reported = line[len("Writing outputs to ") :].strip()
                # Stage stdout is untrusted by the threat model. Only accept a path
                # inside this job's own directory.
                if reported and is_within(self._job_dir(job), Path(reported)):
                    job.metrics.setdefault("inference_dirs", {})[stage_name] = reported
                else:
                    logger.warning(
                        "ignoring out-of-tree output directory reported by stage %s of job %s",
                        stage_name,
                        job.job_id,
                    )

    # ---- path resolution ---------------------------------------------

    def _resolve_frames_dir(
        self, job: JobRecord, paths: pipeline.JobPaths, *, has_trajectory: bool
    ) -> Path:
        """Locate the ArtiFixer prediction frames for the distillation stage."""
        reported = job.metrics.get("inference_dirs", {}).get(pipeline.STAGE_ARTIFIXER)
        candidates: list[Path] = []
        if reported:
            candidates.append(pipeline.predicted_frames_dir(Path(reported), job.scene_id))
        assert self._settings.artifixer_checkpoint is not None
        computed = pipeline.predicted_frames_dir(
            pipeline.inference_output_dir(
                paths.artifixer_save_dir,
                self._settings.artifixer_checkpoint,
                has_trajectory=has_trajectory,
            ),
            job.scene_id,
        )
        candidates.append(computed)
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.png")):
                return candidate
        raise StageFailure(
            "ArtiFixer produced no prediction frames. Looked in: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    @staticmethod
    def _reconstruction_checkpoint(job: JobRecord, paths: pipeline.JobPaths) -> Path:
        """The base 3DGUT checkpoint, honouring a cache hit."""
        recorded = job.metrics.get("reconstruction_checkpoint")
        return Path(recorded) if recorded else paths.reconstruction_checkpoint

    def _final_checkpoint(self, job: JobRecord, paths: pipeline.JobPaths) -> Path:
        """The checkpoint whose Gaussians are the job's deliverable."""
        if job.mode == "reconstruct":
            return self._reconstruction_checkpoint(job, paths)
        reported = job.metrics.get("stage_outputs", {}).get(pipeline.STAGE_ARTIFIXER3D, {})
        path = reported.get("artifixer3d_checkpoint")
        # Same containment rule as the inference output directory: a path scraped
        # from stage stdout is only usable if it is inside this job.
        if path and is_within(paths.job_dir, Path(path)):
            return Path(path)
        return paths.artifixer3d_checkpoint

    # ---- reconstruction cache ----------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._settings.data_root / "cache" / "reconstruction" / key / "checkpoint.pt"

    async def _cached_reconstruction(self, job: JobRecord) -> Path | None:
        key = job.request.get("reconstruction_cache_key")
        if not key:
            return None
        path = self._cache_path(key)
        if path.is_file():
            logger.info("job %s reusing cached reconstruction %s", job.job_id, key)
            return path
        return None

    async def _store_reconstruction_cache(self, job: JobRecord, checkpoint: Path) -> None:
        """Publish the base reconstruction for reuse by later jobs.

        Hardlinked when possible (O(1), no extra bytes) and published by atomic
        rename so a concurrent reader never sees a half-written checkpoint.
        """
        key = job.request.get("reconstruction_cache_key")
        if not key or not checkpoint.is_file():
            return
        target = self._cache_path(key)
        if target.is_file():
            return
        async with self._cache_lock:
            if target.is_file():
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f"checkpoint.{os.getpid()}.partial")
            try:
                await asyncio.to_thread(os.link, checkpoint, staging)
            except OSError:
                await asyncio.to_thread(shutil.copy2, checkpoint, staging)
            await asyncio.to_thread(os.replace, staging, target)
        logger.info("cached reconstruction %s for job %s", key, job.job_id)

    # ---- completion ---------------------------------------------------

    async def _finalize_failure(self, job: JobRecord, exc: StageFailure) -> None:
        cancelled = job.job_id in self._cancelled or str(exc) == "Cancelled"
        job.state = "cancelled" if cancelled else "failed"
        # errors.py requires that client-visible messages carry no absolute server
        # paths. Stage messages are partly hand-built and partly a log excerpt, so
        # both go through the same redaction as the manifest.
        job.error = "Cancelled by request" if cancelled else self._redact(str(exc))
        job.finished_at = utc_now()
        for stage in job.stages:
            if stage.state == "pending":
                stage.state = "cancelled" if cancelled else "skipped"
        await self._store.update_job(job)
        logger.warning("job %s %s: %s", job.job_id, job.state, job.error)

    async def _finalize_success(self, job: JobRecord, paths: pipeline.JobPaths) -> None:
        artifacts = await asyncio.to_thread(self._collect_artifacts, job, paths)
        job.artifacts = artifacts
        await self._store_reconstruction_cache(job, paths.reconstruction_checkpoint)
        job.state = "succeeded"
        job.finished_at = utc_now()
        job.error = None
        # The manifest lists the other artifacts, so it is written last and then
        # registered itself. It therefore never describes its own digest.
        await asyncio.to_thread(self._write_manifest, job, paths)
        job.artifacts.append(
            await asyncio.to_thread(
                self._describe,
                paths.manifest_path,
                paths,
                name="manifest.json",
                kind="metadata",
                description="Full record of this run: request, stage timings, commands and artifacts",
            )
        )
        if not self._settings.keep_intermediate:
            await asyncio.to_thread(self._prune_intermediates, paths)
        await self._store.update_job(job)
        logger.info("job %s succeeded with %d artifact(s)", job.job_id, len(job.artifacts))

    def _collect_artifacts(self, job: JobRecord, paths: pipeline.JobPaths) -> list[ArtifactRecord]:
        """Publish deliverables under ``output/`` and describe them.

        Large binaries are hardlinked, never copied: the checkpoint can be
        gigabytes and both names live on the same filesystem.
        """
        records: list[ArtifactRecord] = []
        output_dir = paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if paths.splat_ply_path.is_file():
            records.append(
                self._describe(
                    paths.splat_ply_path,
                    paths,
                    name="splat.ply",
                    kind="splat_ply",
                    description="Gaussian splat in 3DGS-compatible binary PLY format",
                )
            )

        final_checkpoint = self._final_checkpoint(job, paths)
        if final_checkpoint.is_file():
            published = output_dir / "splat_checkpoint.pt"
            self._publish(final_checkpoint, published)
            records.append(
                self._describe(
                    published,
                    paths,
                    name="splat_checkpoint.pt",
                    kind="splat_checkpoint",
                    description="3DGRUT checkpoint for the delivered splat",
                )
            )

        frames_dir = self._frames_dir_if_present(job, paths)
        if frames_dir is not None:
            archive = paths.frames_archive_path
            count = self._zip_directory(frames_dir, archive)
            if count:
                records.append(
                    self._describe(
                        archive,
                        paths,
                        name="corrected_frames.zip",
                        kind="frames",
                        description=f"{count} ArtiFixer-corrected frames (PNG)",
                    )
                )

        for source, name, description in self._preview_candidates(job, paths):
            if source.is_file():
                published = output_dir / name
                self._publish(source, published)
                records.append(
                    self._describe(published, paths, name=name, kind="video", description=description)
                )

        if paths.splat_stats_path.is_file():
            records.append(
                self._describe(
                    paths.splat_stats_path,
                    paths,
                    name="splat_stats.json",
                    kind="metadata",
                    description="Gaussian count, SH degree and training step of the delivered splat",
                )
            )

        for stage in job.stages:
            log_path = paths.logs_dir / f"{stage.name}.log"
            if log_path.is_file():
                records.append(
                    self._describe(
                        log_path,
                        paths,
                        name=f"logs/{stage.name}.log",
                        kind="log",
                        description=f"stdout/stderr of the {stage.name} stage",
                        hash_file=False,
                    )
                )
        return records

    def _frames_dir_if_present(self, job: JobRecord, paths: pipeline.JobPaths) -> Path | None:
        stage_name = (
            pipeline.STAGE_ARTIFIXER3D_PLUS
            if job.mode == "artifixer3d_plus"
            else pipeline.STAGE_ARTIFIXER
        )
        reported = job.metrics.get("inference_dirs", {}).get(stage_name)
        if not reported:
            return None
        candidate = pipeline.predicted_frames_dir(Path(reported), job.scene_id)
        if not is_within(paths.job_dir, candidate):
            return None
        return candidate if candidate.is_dir() else None

    @staticmethod
    def _preview_candidates(job: JobRecord, paths: pipeline.JobPaths) -> list[tuple[Path, str, str]]:
        """Preview videos written by the 3DGRUT renderer (render.py:543-546)."""
        candidates = [
            (
                paths.reconstruction_render_dir / "trajectory.mp4",
                "reconstruction_preview.mp4",
                "Turntable render of the base 3DGUT reconstruction",
            )
        ]
        if job.mode != "reconstruct":
            candidates.append(
                (
                    paths.artifixer3d_render_dir / "trajectory.mp4",
                    "artifixer3d_preview.mp4",
                    "Turntable render of the ArtiFixer3D splat",
                )
            )
        return candidates

    @staticmethod
    def _publish(source: Path, target: Path) -> None:
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _describe(
        self,
        path: Path,
        paths: pipeline.JobPaths,
        *,
        name: str,
        kind: str,
        description: str,
        hash_file: bool = True,
    ) -> ArtifactRecord:
        size = path.stat().st_size
        digest = None
        if hash_file and size <= MAX_HASH_BYTES:
            digest = self._sha256(path)
        return ArtifactRecord(
            name=name,
            kind=kind,
            relative_path=str(path.relative_to(paths.job_dir)),
            size_bytes=size,
            sha256=digest,
            description=description,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _zip_directory(source: Path, archive: Path) -> int:
        """Bundle PNG frames with ZIP_STORED.

        PNG is already deflated; recompressing costs CPU and saves nothing, and
        stored entries let clients seek directly to a frame.
        """
        frames = sorted(source.glob("*.png"))
        if not frames:
            return 0
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = archive.with_name(archive.name + ".partial")
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as bundle:
            for frame in frames:
                bundle.write(frame, arcname=frame.name)
        os.replace(staging, archive)
        return len(frames)

    def _redact(self, value: str) -> str:
        """Strip deployment-specific absolute paths from a string.

        The manifest is a downloadable artifact, so it must not become a way to
        read back the data root, the repo location, or the internal cache key.
        Placeholders keep the record readable and reproducible in shape.
        """
        for root, placeholder in (
            (str(self._settings.data_root), "<data_root>"),
            (str(self._settings.repo_root), "<repo_root>"),
        ):
            if root:
                value = value.replace(root, placeholder)
        return value

    def _redacted_metrics(self, job: JobRecord) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key, value in job.metrics.items():
            if isinstance(value, dict):
                metrics[key] = {
                    inner_key: self._redact(inner) if isinstance(inner, str) else inner
                    for inner_key, inner in value.items()
                }
            elif isinstance(value, str):
                metrics[key] = self._redact(value)
            else:
                metrics[key] = value
        return metrics

    def _write_manifest(self, job: JobRecord, paths: pipeline.JobPaths) -> None:
        manifest = {
            "job_id": job.job_id,
            "scene_id": job.scene_id,
            "mode": job.mode,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            # Same projection the HTTP API exposes: the stored request also holds
            # the absolute scene root and the cache key, which stay internal.
            "request": public_request(job.request),
            "metrics": self._redacted_metrics(job),
            "artifacts": [
                {
                    "name": artifact.name,
                    "kind": artifact.kind,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                }
                for artifact in job.artifacts
            ],
            "stages": [
                {
                    "name": stage.name,
                    "state": stage.state,
                    "duration_seconds": stage.duration_seconds,
                    "command": self._redact(stage.command) if stage.command else None,
                }
                for stage in job.stages
            ],
        }
        paths.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _prune_intermediates(paths: pipeline.JobPaths) -> None:
        """Drop the working tree once artifacts are published.

        Only directories this job created are removed, and ``output/`` and
        ``logs/`` are always kept.
        """
        for directory in (
            paths.prepared_root.parent,
            paths.artifixer_save_dir,
            paths.artifixer3d_root,
            paths.artifixer3d_plus_save_dir,
        ):
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory, ignore_errors=True)


def build_stage_records(mode: str, *, export_ply: bool) -> list[StageRecord]:
    return [StageRecord(name=name) for name in pipeline.stage_sequence(mode, export_ply=export_ply)]


def artifact_path(settings: Settings, job: JobRecord, artifact_name: str) -> Path:
    """Resolve an artifact download to a real file, with containment enforced."""
    for artifact in job.artifacts:
        if artifact.name == artifact_name:
            job_dir = safe_join(settings.jobs_dir, job.job_id)
            path = safe_join(job_dir, *Path(artifact.relative_path).parts)
            if not path.is_file():
                raise NotFound(f"Artifact {artifact_name!r} is no longer available")
            return path
    raise NotFound(f"Unknown artifact: {artifact_name}")
