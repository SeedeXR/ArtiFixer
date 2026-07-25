# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable scene and job records backed by SQLite.

Why SQLite rather than in-process dicts: a GPU job runs for minutes to hours, so a
restart must not lose the queue or orphan a job in ``running`` forever. Why not an
external broker: a single-node GPU service gains nothing from one and pays for it
in operational surface.

Concurrency model: one connection, one lock, all synchronous calls dispatched to a
worker thread by the async wrappers. WAL mode keeps readers from blocking the
writer. Serializing writes behind a lock is not a bottleneck here — job mutation
happens a handful of times per stage, while the GPU work takes minutes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splat_api.app.errors import Conflict, NotFound

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id   TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    digest     TEXT,
    source     TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scenes_digest ON scenes (digest);
CREATE INDEX IF NOT EXISTS scenes_created ON scenes (created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    scene_id     TEXT NOT NULL,
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    client_token TEXT,
    api_key_id   TEXT,
    payload      TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_client_token ON jobs (client_token)
    WHERE client_token IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs (state, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_scene ON jobs (scene_id);
"""

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})

# The stored request holds absolute server paths and the cache key. Only these
# fields ever leave the process, and every consumer (the HTTP projection and the
# artifact manifest) shares this one list so they cannot drift apart.
PUBLIC_REQUEST_FIELDS = frozenset(
    {
        "mode",
        "reconstruction_steps",
        "artifixer3d_steps",
        "inference_steps",
        "selected_image_count",
        "validation_holdout_auto",
        "trajectory_frames",
        "metric_scale",
        "export_ply",
    }
)


def public_request(request: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key in PUBLIC_REQUEST_FIELDS}


def utc_now() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class StageRecord:
    name: str
    description: str = ""
    state: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    message: str | None = None
    command: str | None = None


@dataclass
class ArtifactRecord:
    name: str
    kind: str
    relative_path: str
    size_bytes: int
    sha256: str | None = None
    description: str | None = None


@dataclass
class JobRecord:
    """Everything known about one pipeline run."""

    job_id: str
    scene_id: str
    mode: str
    state: str
    created_at: str
    updated_at: str
    request: dict[str, Any]
    stages: list[StageRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    client_token: str | None = None
    api_key_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    gpu_index: int | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    restarts: int = 0

    @property
    def progress(self) -> float:
        """Fraction of stages that actually completed.

        Skipped stages do not count: a job that failed at its first stage has
        accomplished nothing, and reporting 0.8 because four later stages were
        marked skipped would be actively misleading.
        """
        if not self.stages:
            return 1.0 if self.state in TERMINAL_STATES else 0.0
        done = sum(1 for stage in self.stages if stage.state == "succeeded")
        return round(done / len(self.stages), 4)

    def stage(self, name: str) -> StageRecord:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> JobRecord:
        data = json.loads(raw)
        data["stages"] = [StageRecord(**stage) for stage in data.get("stages", [])]
        data["artifacts"] = [ArtifactRecord(**artifact) for artifact in data.get("artifacts", [])]
        return cls(**data)


@dataclass
class SceneRecord:
    scene_id: str
    created_at: str
    source: str
    summary: dict[str, Any]
    size_bytes: int
    label: str | None = None
    digest: str | None = None
    root_override: str | None = None
    image_names: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> SceneRecord:
        return cls(**json.loads(raw))


class JobStore:
    """Synchronous storage core plus async wrappers."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._lock = threading.Lock()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._connection.commit()
        try:
            database_path.chmod(0o640)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # ---- scenes -------------------------------------------------------

    def insert_scene(self, scene: SceneRecord) -> None:
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO scenes (scene_id, created_at, digest, source, payload) VALUES (?, ?, ?, ?, ?)",
                    (scene.scene_id, scene.created_at, scene.digest, scene.source, scene.to_json()),
                )
            except sqlite3.IntegrityError as exc:
                raise Conflict(f"Scene {scene.scene_id} already exists") from exc
            self._connection.commit()

    def get_scene(self, scene_id: str) -> SceneRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM scenes WHERE scene_id = ?", (scene_id,)
            ).fetchone()
        if row is None:
            raise NotFound(f"Unknown scene: {scene_id}")
        return SceneRecord.from_json(row["payload"])

    def find_scene_by_digest(self, digest: str) -> SceneRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM scenes WHERE digest = ? ORDER BY created_at LIMIT 1", (digest,)
            ).fetchone()
        return SceneRecord.from_json(row["payload"]) if row else None

    def list_scenes(self, limit: int) -> list[SceneRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM scenes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [SceneRecord.from_json(row["payload"]) for row in rows]

    def delete_scene(self, scene_id: str) -> None:
        with self._lock:
            active = self._connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE scene_id = ? AND state IN ('queued', 'running')",
                (scene_id,),
            ).fetchone()["count"]
            if active:
                raise Conflict(f"Scene {scene_id} has {active} active job(s)")
            cursor = self._connection.execute("DELETE FROM scenes WHERE scene_id = ?", (scene_id,))
            self._connection.commit()
        if cursor.rowcount == 0:
            raise NotFound(f"Unknown scene: {scene_id}")

    # ---- jobs ---------------------------------------------------------

    def insert_job(self, job: JobRecord) -> None:
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO jobs (job_id, scene_id, state, created_at, updated_at, client_token, "
                    "api_key_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job.job_id,
                        job.scene_id,
                        job.state,
                        job.created_at,
                        job.updated_at,
                        job.client_token,
                        job.api_key_id,
                        job.to_json(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise Conflict("A job with this client_token already exists") from exc
            self._connection.commit()

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"Unknown job: {job_id}")
        return JobRecord.from_json(row["payload"])

    def find_job_by_client_token(self, token: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM jobs WHERE client_token = ?", (token,)
            ).fetchone()
        return JobRecord.from_json(row["payload"]) if row else None

    def update_job(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        with self._lock:
            self._connection.execute(
                "UPDATE jobs SET state = ?, updated_at = ?, payload = ? WHERE job_id = ?",
                (job.state, job.updated_at, job.to_json(), job.job_id),
            )
            self._connection.commit()

    def list_jobs(
        self, *, state: str | None, limit: int, before: str | None, scene_id: str | None
    ) -> list[JobRecord]:
        """Keyset pagination on ``(created_at, job_id)``.

        ``created_at`` is a millisecond ISO-8601 string, which sorts
        lexicographically in time order, so keyset paging works without OFFSET
        scans. It is not unique, though — two submissions inside one millisecond is
        ordinary under load — so ``job_id`` is the tiebreaker. Without it a page
        boundary landing inside a group of same-millisecond jobs would skip the
        rest of that group entirely.
        """
        query = ["SELECT payload FROM jobs WHERE 1 = 1"]
        params: list[Any] = []
        if state is not None:
            query.append("AND state = ?")
            params.append(state)
        if scene_id is not None:
            query.append("AND scene_id = ?")
            params.append(scene_id)
        if before is not None:
            created_at, _, job_id = before.partition("|")
            if job_id:
                query.append("AND (created_at < ? OR (created_at = ? AND job_id < ?))")
                params += [created_at, created_at, job_id]
            else:
                query.append("AND created_at < ?")
                params.append(created_at)
        query.append("ORDER BY created_at DESC, job_id DESC LIMIT ?")
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(" ".join(query), params).fetchall()
        return [JobRecord.from_json(row["payload"]) for row in rows]

    def delete_job(self, job_id: str) -> None:
        """Remove a job row.

        Used only to undo an insert whose queue admission failed; otherwise a
        phantom `queued` row would never run, never reach a terminal state, hold
        its `client_token`, and block scene deletion forever.
        """
        with self._lock:
            self._connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self._connection.commit()

    def cancel_if_active(self, job_id: str) -> JobRecord:
        """Atomically move a non-terminal job to ``cancelled``.

        Read-modify-write inside the store lock so a concurrent worker write cannot
        be lost. If the worker got there first, its result stands and is returned
        unchanged.
        """
        with self._lock:
            row = self._connection.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"Unknown job: {job_id}")
            job = JobRecord.from_json(row["payload"])
            if job.state in TERMINAL_STATES:
                return job
            job.state = "cancelled"
            job.finished_at = utc_now()
            job.error = "Cancelled by request"
            for stage in job.stages:
                if stage.state in ("pending", "running"):
                    stage.state = "cancelled"
            job.updated_at = utc_now()
            self._connection.execute(
                "UPDATE jobs SET state = ?, updated_at = ?, payload = ? WHERE job_id = ?",
                (job.state, job.updated_at, job.to_json(), job.job_id),
            )
            self._connection.commit()
            return job

    def count_active_jobs(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE state IN ('queued', 'running')"
            ).fetchone()
        return int(row["count"])

    def pending_jobs_in_order(self) -> list[JobRecord]:
        """Jobs to (re)admit at startup, oldest first.

        ``running`` jobs are included: every pipeline stage skips work whose
        outputs already exist, so re-running an interrupted job resumes rather
        than repeating it.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM jobs WHERE state IN ('queued', 'running') ORDER BY created_at ASC"
            ).fetchall()
        return [JobRecord.from_json(row["payload"]) for row in rows]

    def state_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute("SELECT state, COUNT(*) AS count FROM jobs GROUP BY state").fetchall()
        return {row["state"]: int(row["count"]) for row in rows}


class AsyncJobStore:
    """Async facade so request handlers never block the event loop on disk I/O."""

    def __init__(self, store: JobStore) -> None:
        self._store = store

    @property
    def sync(self) -> JobStore:
        return self._store

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self._store, name), *args, **kwargs)

    async def insert_scene(self, scene: SceneRecord) -> None:
        await self._call("insert_scene", scene)

    async def get_scene(self, scene_id: str) -> SceneRecord:
        return await self._call("get_scene", scene_id)

    async def find_scene_by_digest(self, digest: str) -> SceneRecord | None:
        return await self._call("find_scene_by_digest", digest)

    async def list_scenes(self, limit: int) -> list[SceneRecord]:
        return await self._call("list_scenes", limit)

    async def delete_scene(self, scene_id: str) -> None:
        await self._call("delete_scene", scene_id)

    async def insert_job(self, job: JobRecord) -> None:
        await self._call("insert_job", job)

    async def get_job(self, job_id: str) -> JobRecord:
        return await self._call("get_job", job_id)

    async def find_job_by_client_token(self, token: str) -> JobRecord | None:
        return await self._call("find_job_by_client_token", token)

    async def update_job(self, job: JobRecord) -> None:
        await self._call("update_job", job)

    async def delete_job(self, job_id: str) -> None:
        await self._call("delete_job", job_id)

    async def cancel_if_active(self, job_id: str) -> JobRecord:
        return await self._call("cancel_if_active", job_id)

    async def list_jobs(
        self, *, state: str | None, limit: int, before: str | None, scene_id: str | None
    ) -> list[JobRecord]:
        return await self._call("list_jobs", state=state, limit=limit, before=before, scene_id=scene_id)

    async def count_active_jobs(self) -> int:
        return await self._call("count_active_jobs")

    async def state_counts(self) -> dict[str, int]:
        return await self._call("state_counts")


def monotonic_duration(start: float) -> float:
    return round(time.monotonic() - start, 3)
