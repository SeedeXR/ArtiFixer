# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface: scenes, jobs, artifacts, capabilities."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse

from splat_api import __version__
from splat_api.app import pipeline
from splat_api.app.archives import extract_colmap_archive
from splat_api.app.colmap_input import ColmapSummary, validate_scene
from splat_api.app.config import Settings
from splat_api.app.errors import (
    ApiError,
    BadRequest,
    Conflict,
    NotFound,
    PayloadTooLarge,
    ServiceUnavailable,
    UnprocessableInput,
)
from splat_api.app.jobstore import (
    ArtifactRecord,
    AsyncJobStore,
    JobRecord,
    SceneRecord,
    StageRecord,
    public_request,
    utc_now,
)
from splat_api.app.paths import directory_size, new_id, safe_join, validate_id
from splat_api.app.scheduler import Scheduler, artifact_path, build_stage_records
from splat_api.app.schemas import (
    ArtifactInfo,
    Capabilities,
    JobCreate,
    JobInfo,
    JobList,
    SceneInfo,
    SceneRegister,
    StageInfo,
)
from splat_api.app.security import Principal, authenticate

logger = logging.getLogger("splat_api.routes")

router = APIRouter()

UPLOAD_CHUNK = 4 * 1024 * 1024
MAX_LOG_TAIL_BYTES = 256 * 1024

# 3DGRUT always builds a validation ColmapDataset alongside the training one
# (threedgrut/datasets/__init__.py:46-56), and with a selected-indices file the
# validation split is `setdiff1d(all, selected)`
# (threedgrut/datasets/dataset_colmap.py:101-104). Selecting every image therefore
# leaves it empty and training dies in compute_spatial_extents with
# "Expected reduction dim 0 to have non-zero size".
#
# prepare_colmap_artifixer_inputs always passes that file, so the only way through
# the public CLI is to hold some views back. We hold out every 8th image, matching
# the engine's own `test_split_interval: 8` (configs/dataset/colmap.yaml:3).
VALIDATION_HOLDOUT_INTERVAL = 8


# --- dependencies ---------------------------------------------------------


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> AsyncJobStore:
    return request.app.state.store


def get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


def require_scope(scope: str):
    """Build a dependency that authenticates and checks one scope."""

    def dependency(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> Principal:
        principal = authenticate(request, settings)
        principal.require(scope)
        request.state.principal = principal
        return principal

    return dependency


RequireRead = Annotated[Principal, Depends(require_scope("read"))]
RequireWrite = Annotated[Principal, Depends(require_scope("write"))]
RequireAdmin = Annotated[Principal, Depends(require_scope("admin"))]


# --- serialization helpers -------------------------------------------------


def scene_to_info(scene: SceneRecord, *, include_names: bool = False) -> SceneInfo:
    summary = scene.summary
    return SceneInfo(
        scene_id=scene.scene_id,
        created_at=scene.created_at,
        label=scene.label,
        source=scene.source,  # type: ignore[arg-type]
        image_count=int(summary["image_count"]),
        camera_count=int(summary["camera_count"]),
        camera_models=list(summary["camera_models"]),
        point_count=int(summary["point_count"]),
        colmap_width=int(summary["colmap_width"]),
        colmap_height=int(summary["colmap_height"]),
        size_bytes=scene.size_bytes,
        image_names=list(scene.image_names) if include_names else None,
    )


def _artifact_to_info(job_id: str, artifact: ArtifactRecord) -> ArtifactInfo:
    return ArtifactInfo(
        name=artifact.name,
        kind=artifact.kind,  # type: ignore[arg-type]
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        download_path=f"/v1/jobs/{job_id}/artifacts/{artifact.name}",
        description=artifact.description,
    )


def _stage_to_info(stage: StageRecord) -> StageInfo:
    return StageInfo(
        name=stage.name,
        state=stage.state,  # type: ignore[arg-type]
        started_at=stage.started_at,
        finished_at=stage.finished_at,
        duration_seconds=stage.duration_seconds,
        exit_code=stage.exit_code,
        message=stage.message,
    )


def job_to_info(job: JobRecord, scheduler: Scheduler | None = None) -> JobInfo:
    """Project a stored job into its public representation.

    ``request`` is filtered: the stored copy holds absolute server paths (scene
    root, cache key) that must not leave the process.
    """
    return JobInfo(
        job_id=job.job_id,
        scene_id=job.scene_id,
        state=job.state,  # type: ignore[arg-type]
        mode=job.mode,  # type: ignore[arg-type]
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        queue_position=scheduler.queue_position(job.job_id) if scheduler and job.state == "queued" else None,
        gpu_index=job.gpu_index,
        stages=[_stage_to_info(stage) for stage in job.stages],
        artifacts=[_artifact_to_info(job.job_id, artifact) for artifact in job.artifacts],
        error=job.error,
        request=public_request(job.request),
    )


# --- capabilities ---------------------------------------------------------


@router.get("/v1/capabilities", response_model=Capabilities, tags=["meta"])
async def capabilities(
    _: RequireRead, settings: Annotated[Settings, Depends(get_settings)]
) -> Capabilities:
    modes: list[Any] = ["reconstruct"]
    if settings.artifixer_available:
        modes += ["artifixer3d", "artifixer3d_plus"]
    return Capabilities(
        service_version=__version__,
        modes=modes,
        artifixer_checkpoint_configured=settings.artifixer_available,
        artifixer_model_id=settings.artifixer_model_id,
        gpu_count=len(settings.cuda_devices),
        max_concurrent_jobs=settings.max_concurrent_jobs,
        queue_capacity=settings.queue_capacity,
        limits={
            "max_upload_bytes": settings.max_upload_bytes,
            "max_images": settings.max_images,
            "max_archive_members": settings.max_archive_members,
            "max_uncompressed_bytes": settings.max_uncompressed_bytes,
            "max_trajectory_frames": settings.max_trajectory_frames,
            "reconstruction_steps_max": settings.reconstruction_steps_max,
            "artifixer3d_steps_max": settings.artifixer3d_steps_max,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        },
    )


# --- scene upload ---------------------------------------------------------


async def _persist_upload(request: Request, settings: Settings, destination: Path) -> tuple[int, str]:
    """Stream the request body to ``destination``, hashing as we go.

    Two content types are accepted. ``application/zip`` streams the raw body and
    is the fast path: no multipart framing, no intermediate spool file.
    ``multipart/form-data`` is supported for ``curl -F``/browser clients and costs
    one extra copy.

    The byte cap is enforced against bytes actually received, not the declared
    ``Content-Length``, because chunked uploads declare nothing.
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    digest = hashlib.sha256()
    total = 0

    def guard(chunk: bytes) -> None:
        nonlocal total
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise PayloadTooLarge(
                f"Upload exceeds the {settings.max_upload_bytes} byte limit"
            )
        digest.update(chunk)

    if content_type == "multipart/form-data":
        # Starlette's multipart parser spools every file part to a temporary file
        # with no size cap of its own, and it runs to completion before any of our
        # code sees a byte. So the declared length must be checked first, and a
        # multipart body with no declared length is refused outright — otherwise a
        # chunked upload could fill the system temp directory.
        declared = request.headers.get("content-length")
        if declared is None:
            raise BadRequest(
                "multipart/form-data uploads must declare Content-Length. "
                "Use Content-Type: application/zip with a raw body to stream without one."
            )
        try:
            declared_length = int(declared)
        except ValueError as exc:
            raise BadRequest("Malformed Content-Length header") from exc
        if declared_length > settings.max_upload_bytes:
            raise PayloadTooLarge(
                f"Upload declares {declared_length} bytes; the limit is {settings.max_upload_bytes}"
            )
        form = await request.form(max_files=1, max_fields=8)
        try:
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise BadRequest("multipart upload must include a 'file' part containing the ZIP archive")
            with destination.open("wb") as sink:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK)
                    if not chunk:
                        break
                    guard(chunk)
                    sink.write(chunk)
        finally:
            with contextlib.suppress(Exception):
                await form.close()
    elif content_type in ("application/zip", "application/x-zip-compressed", "application/octet-stream"):
        with destination.open("wb") as sink:
            async for chunk in request.stream():
                if not chunk:
                    continue
                guard(chunk)
                sink.write(chunk)
    else:
        raise BadRequest(
            "Upload Content-Type must be one of application/zip, application/x-zip-compressed, "
            "application/octet-stream (raw body), or multipart/form-data with a 'file' part; "
            f"got {content_type!r}"
        )

    if total == 0:
        raise BadRequest("Uploaded archive is empty")
    destination.chmod(0o640)
    return total, digest.hexdigest()


def _ingest_scene(scene_dir: Path, settings: Settings) -> tuple[ColmapSummary, dict[str, object]]:
    """Blocking half of ingestion: extract, validate, measure.

    Everything raised here must be an :class:`ApiError`. A caller-supplied archive
    can make zipfile or Pillow raise their own exception types, and letting those
    escape would turn ordinary bad input into a 500 with a traceback in the log.
    """
    try:
        report = extract_colmap_archive(scene_dir / "upload.zip", scene_dir, settings)
        summary = validate_scene(scene_dir, max_images=settings.max_images)
    except ApiError:
        raise
    except (zipfile.BadZipFile, NotImplementedError) as exc:
        raise BadRequest(f"Archive could not be read: {type(exc).__name__}: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise UnprocessableInput(f"Scene could not be processed: {type(exc).__name__}: {exc}") from exc
    return summary, report.as_dict()


@router.post(
    "/v1/scenes",
    response_model=SceneInfo,
    status_code=201,
    tags=["scenes"],
    summary="Upload a COLMAP scene as a ZIP archive",
)
async def upload_scene(
    request: Request,
    response: Response,
    _: RequireWrite,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
    label: Annotated[str | None, Query(max_length=128)] = None,
    dedupe: Annotated[bool, Query(description="Return an existing scene when the archive matches byte-for-byte")] = True,
) -> SceneInfo:
    """Accept ``images/`` + ``sparse/0/{cameras,images,points3D}.bin`` inside a ZIP.

    The archive may be rooted at a scene folder (``truck/images/...``). Only the
    files the pipeline needs are extracted; anything else in the archive is
    ignored. On success the response describes the validated scene, and the
    ``scene_id`` is what you pass to ``POST /v1/jobs``.
    """
    if label is not None and not label.strip():
        raise BadRequest("label must not be blank")

    scene_id = new_id("scene")
    staging_dir = safe_join(settings.uploads_dir, scene_id)
    staging_dir.mkdir(parents=True, exist_ok=False)
    archive_path = staging_dir / "upload.zip"
    promoted = False

    try:
        size, digest = await _persist_upload(request, settings, archive_path)

        if dedupe:
            existing = await store.find_scene_by_digest(digest)
            if existing is not None:
                logger.info("upload deduplicated to scene %s", existing.scene_id)
                response.status_code = 200
                response.headers["X-Scene-Deduplicated"] = "true"
                return scene_to_info(existing, include_names=True)

        summary, details = await asyncio.to_thread(_ingest_scene, staging_dir, settings)

        scene_dir = safe_join(settings.scenes_dir, scene_id)
        archive_path.unlink(missing_ok=True)
        await asyncio.to_thread(os.replace, staging_dir, scene_dir)
        staging_dir = scene_dir  # cleanup below now targets the final location

        record = SceneRecord(
            scene_id=scene_id,
            created_at=utc_now(),
            source="upload",
            summary=summary.as_dict(),
            size_bytes=await asyncio.to_thread(directory_size, scene_dir),
            label=label,
            digest=digest,
            image_names=list(summary.image_names),
        )
        await store.insert_scene(record)
        promoted = True
        logger.info(
            "scene %s ingested: %d images, %d points, %d bytes uploaded (%s)",
            scene_id,
            summary.image_count,
            summary.point_count,
            size,
            details,
        )
        return scene_to_info(record, include_names=True)
    finally:
        # Anything that is not a successfully registered scene is scratch: a
        # rejected archive, a failed extraction, or a dedupe hit whose bytes we
        # already have. Leaving it behind would let a caller fill the disk by
        # re-POSTing the same archive.
        if not promoted:
            await asyncio.to_thread(shutil.rmtree, staging_dir, True)


@router.post(
    "/v1/scenes/register",
    response_model=SceneInfo,
    status_code=201,
    tags=["scenes"],
    summary="Register a COLMAP scene already present on the server (admin)",
)
async def register_scene(
    payload: SceneRegister,
    _: RequireAdmin,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
) -> SceneInfo:
    """Zero-copy ingestion for scenes already on server storage.

    Admin-only and confined to the single import root ``<data_root>/import``: the
    caller names a filesystem path, so this is an authorization decision rather
    than a convenience. Nothing is copied, which makes it the fastest way to
    onboard terabyte-scale captures.
    """
    import_root = settings.data_root / "import"
    if not import_root.is_dir():
        raise UnprocessableInput(
            "No import root configured. Create <data_root>/import and place scenes under it."
        )
    relative = payload.path.strip().lstrip("/")
    if not relative:
        raise BadRequest("path must name a directory beneath the import root")
    scene_dir = safe_join(import_root, *Path(relative).parts)
    if not scene_dir.is_dir():
        raise NotFound(f"No scene directory at import path {relative!r}")

    summary = await asyncio.to_thread(validate_scene, scene_dir, max_images=settings.max_images)
    scene_id = new_id("scene")
    record = SceneRecord(
        scene_id=scene_id,
        created_at=utc_now(),
        source="registered",
        summary=summary.as_dict(),
        size_bytes=await asyncio.to_thread(directory_size, scene_dir),
        label=payload.label,
        digest=None,
        root_override=str(scene_dir),
        image_names=list(summary.image_names),
    )
    await store.insert_scene(record)
    logger.info("scene %s registered from import path %s", scene_id, relative)
    return scene_to_info(record, include_names=True)


@router.get("/v1/scenes", response_model=list[SceneInfo], tags=["scenes"])
async def list_scenes(
    _: RequireRead,
    store: Annotated[AsyncJobStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SceneInfo]:
    scenes = await store.list_scenes(limit)
    return [scene_to_info(scene) for scene in scenes]


@router.get("/v1/scenes/{scene_id}", response_model=SceneInfo, tags=["scenes"])
async def get_scene(
    scene_id: str, _: RequireRead, store: Annotated[AsyncJobStore, Depends(get_store)]
) -> SceneInfo:
    validate_id(scene_id, kind="scene_id")
    scene = await store.get_scene(scene_id)
    return scene_to_info(scene, include_names=True)


@router.delete("/v1/scenes/{scene_id}", status_code=204, tags=["scenes"])
async def delete_scene(
    scene_id: str,
    _: RequireAdmin,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
) -> Response:
    validate_id(scene_id, kind="scene_id")
    scene = await store.get_scene(scene_id)
    await store.delete_scene(scene_id)
    if scene.source == "upload":
        # Registered scenes are not ours to delete: the bytes belong to whoever
        # placed them under the import root.
        scene_dir = safe_join(settings.scenes_dir, scene_id)
        await asyncio.to_thread(shutil.rmtree, scene_dir, True)
    return Response(status_code=204)


# --- jobs -----------------------------------------------------------------


def _scene_root(settings: Settings, scene: SceneRecord) -> Path:
    if scene.root_override:
        path = Path(scene.root_override)
        if not path.is_dir():
            raise UnprocessableInput(f"Registered scene directory is gone: {scene.scene_id}")
        return path
    return safe_join(settings.scenes_dir, scene.scene_id)


def _validate_job_against_scene(payload: JobCreate, scene: SceneRecord, settings: Settings) -> None:
    """Reject impossible jobs before anything is queued."""
    if payload.mode != "reconstruct" and not settings.artifixer_available:
        raise ServiceUnavailable(
            f"mode={payload.mode} needs an ArtiFixer checkpoint. Set "
            "SPLAT_API_ARTIFIXER_CHECKPOINT (and SPLAT_API_ARTIFIXER_MODEL_ID to the matching "
            "base model) to enable the correction stages."
        )

    known = set(scene.image_names)
    if payload.selected_image_names is not None:
        unknown = [name for name in payload.selected_image_names if name not in known]
        if unknown:
            raise UnprocessableInput(
                f"selected_image_names not present in scene {scene.scene_id}: {unknown[:10]}"
            )
        if payload.mode != "reconstruct" and payload.trajectory is None:
            # artifixer3d.generated_frame_indices (:235-245) needs at least one
            # non-anchor frame, i.e. a strict subset of the source views.
            if len(payload.selected_image_names) >= len(known):
                raise UnprocessableInput(
                    "artifixer3d needs at least one frame to generate: selected_image_names must be "
                    f"a strict subset of the {len(known)} scene images, or supply a trajectory"
                )

    if payload.trajectory is not None:
        frames = len(payload.trajectory.frames)
        if frames > settings.max_trajectory_frames:
            raise PayloadTooLarge(
                f"trajectory has {frames} frames; the limit is {settings.max_trajectory_frames}"
            )


def resolve_training_views(
    selected_image_names: list[str] | None, image_names: list[str]
) -> tuple[list[str], bool]:
    """Decide which images become 3DGRUT training anchors.

    Returns ``(names, auto_holdout)``. When the caller named a subset we use it
    verbatim. Otherwise we hold out every ``VALIDATION_HOLDOUT_INTERVAL``-th image
    so 3DGRUT's validation split is non-empty; see the constant's comment.
    """
    if selected_image_names is not None:
        return list(selected_image_names), False
    training = [
        name
        for index, name in enumerate(image_names)
        if index % VALIDATION_HOLDOUT_INTERVAL != 0
    ]
    if not training:
        # Unreachable for scenes that passed upload validation, which requires at
        # least two images; kept so a future limit change cannot pass an empty set
        # to 3DGRUT.
        raise UnprocessableInput(
            f"Scene has only {len(image_names)} image(s); name the training views "
            "explicitly with selected_image_names"
        )
    return training, True


def _clamp_steps(requested: int | None, default: int, maximum: int, *, field: str) -> int:
    """Resolve a step count against the deployment's ceiling.

    Raises 422, the same status the schema bounds produce, so a caller sees one
    HTTP status for "too many steps" either way. The machine-readable code still
    differs: `validation_error` from the schema bound, `unprocessable_input` here.
    """
    value = default if requested is None else requested
    if value > maximum:
        raise UnprocessableInput(
            f"{field}={value} exceeds the configured maximum of {maximum} for this deployment"
        )
    return value


def _write_job_inputs(paths: pipeline.JobPaths, payload: JobCreate, training_views: list[str]) -> None:
    """Materialize job inputs as the files the repo's CLIs read."""
    inputs_dir = paths.job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths.selected_names_file.write_text("\n".join(training_views) + "\n")
    if payload.trajectory is not None:
        paths.trajectory_file.write_text(
            json.dumps(payload.trajectory.to_transforms_json(), indent=2) + "\n"
        )


@router.post("/v1/jobs", response_model=JobInfo, status_code=202, tags=["jobs"])
async def create_job(
    payload: JobCreate,
    response: Response,
    _: RequireWrite,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> JobInfo:
    """Queue a COLMAP-to-splat pipeline run.

    Returns 202 immediately; poll ``GET /v1/jobs/{job_id}`` for progress and
    ``artifacts`` for the finished splat.
    """
    validate_id(payload.scene_id, kind="scene_id")
    scene = await store.get_scene(payload.scene_id)
    _validate_job_against_scene(payload, scene, settings)

    if payload.client_token is not None:
        existing = await store.find_job_by_client_token(payload.client_token)
        if existing is not None:
            if existing.scene_id != payload.scene_id or existing.mode != payload.mode:
                raise Conflict(
                    "client_token was already used for a different job; use a fresh token"
                )
            response.status_code = 200
            return job_to_info(existing, scheduler)

    reconstruction_steps = _clamp_steps(
        payload.reconstruction_steps,
        settings.reconstruction_steps_default,
        settings.reconstruction_steps_max,
        field="reconstruction_steps",
    )
    artifixer3d_steps = _clamp_steps(
        payload.artifixer3d_steps,
        settings.artifixer3d_steps_default,
        settings.artifixer3d_steps_max,
        field="artifixer3d_steps",
    )

    training_views, auto_holdout = resolve_training_views(
        payload.selected_image_names, list(scene.image_names)
    )

    job_id = new_id("job")
    scene_root = _scene_root(settings, scene)
    stages: list[StageRecord] = build_stage_records(payload.mode, export_ply=payload.export_ply)
    principal: Principal | None = getattr(request.state, "principal", None)

    job = JobRecord(
        job_id=job_id,
        scene_id=scene.scene_id,
        mode=payload.mode,
        state="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
        client_token=payload.client_token,
        api_key_id=principal.key_id if principal else None,
        stages=stages,
        request={
            "mode": payload.mode,
            "scene_root": str(scene_root),
            "reconstruction_steps": reconstruction_steps,
            "artifixer3d_steps": artifixer3d_steps,
            "inference_steps": payload.inference_steps,
            "selected_image_names": training_views,
            "selected_image_count": len(training_views),
            "validation_holdout_auto": auto_holdout,
            "trajectory": payload.trajectory is not None,
            "trajectory_frames": len(payload.trajectory.frames) if payload.trajectory else None,
            "metric_scale": payload.metric_scale,
            "export_ply": payload.export_ply,
            "reconstruction_cache_key": pipeline.reconstruction_cache_key(
                scene_id=scene.scene_id,
                steps=reconstruction_steps,
                selected_image_names=training_views,
            ),
        },
    )

    job_dir = safe_join(settings.jobs_dir, job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    paths = pipeline.JobPaths(
        job_dir=job_dir,
        scene_dir=scene_root,
        scene_id=scene.scene_id,
        reconstruction_steps=reconstruction_steps,
        artifixer3d_steps=artifixer3d_steps,
    )
    inserted = False
    try:
        await asyncio.to_thread(_write_job_inputs, paths, payload, training_views)
        # Re-read the scene: an admin DELETE may have landed while we were writing
        # inputs, and queueing a job against a deleted scene root would fail minutes
        # later inside the prepare stage instead of here.
        await store.get_scene(scene.scene_id)
        await store.insert_job(job)
        inserted = True
        await scheduler.submit(job)
    except BaseException:
        if inserted:
            # Otherwise the row stays `queued` with no queue slot: it would never
            # run, never reach a terminal state, hold its client_token, and block
            # deletion of its scene forever.
            await store.delete_job(job_id)
        await asyncio.to_thread(shutil.rmtree, job_dir, True)
        raise

    logger.info("job %s queued (scene=%s mode=%s)", job_id, scene.scene_id, payload.mode)
    response.headers["Location"] = f"/v1/jobs/{job_id}"
    return job_to_info(job, scheduler)


@router.get("/v1/jobs", response_model=JobList, tags=["jobs"])
async def list_jobs(
    _: RequireRead,
    store: Annotated[AsyncJobStore, Depends(get_store)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    state: Annotated[str | None, Query(pattern="^(queued|running|succeeded|failed|cancelled)$")] = None,
    scene_id: Annotated[str | None, Query(max_length=64)] = None,
    cursor: Annotated[
        str | None,
        Query(max_length=128, description="Opaque next_cursor from a previous response."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JobList:
    if scene_id is not None:
        validate_id(scene_id, kind="scene_id")
    jobs = await store.list_jobs(state=state, limit=limit, before=cursor, scene_id=scene_id)
    # "created_at|job_id": created_at alone is not unique at millisecond resolution,
    # and a page boundary inside a same-millisecond group would skip the remainder.
    next_cursor = f"{jobs[-1].created_at}|{jobs[-1].job_id}" if len(jobs) == limit else None
    return JobList(jobs=[job_to_info(job, scheduler) for job in jobs], next_cursor=next_cursor)


@router.get("/v1/jobs/{job_id}", response_model=JobInfo, tags=["jobs"])
async def get_job(
    job_id: str,
    _: RequireRead,
    store: Annotated[AsyncJobStore, Depends(get_store)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> JobInfo:
    validate_id(job_id, kind="job_id")
    job = await store.get_job(job_id)
    return job_to_info(job, scheduler)


@router.post("/v1/jobs/{job_id}/cancel", response_model=JobInfo, tags=["jobs"])
async def cancel_job(
    job_id: str,
    _: RequireWrite,
    store: Annotated[AsyncJobStore, Depends(get_store)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> JobInfo:
    validate_id(job_id, kind="job_id")
    await scheduler.cancel(job_id)
    job = await store.get_job(job_id)
    return job_to_info(job, scheduler)


@router.get("/v1/jobs/{job_id}/artifacts", response_model=list[ArtifactInfo], tags=["artifacts"])
async def list_artifacts(
    job_id: str, _: RequireRead, store: Annotated[AsyncJobStore, Depends(get_store)]
) -> list[ArtifactInfo]:
    validate_id(job_id, kind="job_id")
    job = await store.get_job(job_id)
    return [_artifact_to_info(job_id, artifact) for artifact in job.artifacts]


@router.get("/v1/jobs/{job_id}/artifacts/{artifact_name:path}", tags=["artifacts"])
async def download_artifact(
    job_id: str,
    artifact_name: str,
    principal: RequireRead,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
) -> FileResponse:
    """Stream one artifact.

    ``FileResponse`` uses ``sendfile`` where the server supports it, so a
    multi-gigabyte splat never passes through Python buffers. The strong
    ``ETag`` is the artifact's own SHA-256, letting clients revalidate cheaply.

    Deliverables need only ``read``. Stage logs additionally need ``write``: they
    are raw subprocess output and carry absolute container paths and tracebacks,
    which a read-only monitoring credential has no reason to see.
    """
    validate_id(job_id, kind="job_id")
    job = await store.get_job(job_id)
    path = artifact_path(settings, job, artifact_name)
    record = next(artifact for artifact in job.artifacts if artifact.name == artifact_name)
    if record.kind == "log":
        principal.require("write")
    headers = {"Content-Disposition": f'attachment; filename="{Path(record.name).name}"'}
    if record.sha256:
        headers["ETag"] = f'"{record.sha256}"'
    media_type = "application/octet-stream"
    if record.name.endswith(".zip"):
        media_type = "application/zip"
    elif record.name.endswith(".mp4"):
        media_type = "video/mp4"
    elif record.name.endswith(".log"):
        media_type = "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media_type, headers=headers)


@router.get("/v1/jobs/{job_id}/logs/{stage}", response_class=PlainTextResponse, tags=["jobs"])
async def get_stage_log(
    job_id: str,
    stage: str,
    _: RequireWrite,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AsyncJobStore, Depends(get_store)],
    tail_bytes: Annotated[int, Query(ge=1024, le=MAX_LOG_TAIL_BYTES)] = 64 * 1024,
) -> PlainTextResponse:
    """Return the tail of a stage log, for live progress during long stages."""
    validate_id(job_id, kind="job_id")
    job = await store.get_job(job_id)
    if stage not in {record.name for record in job.stages}:
        raise NotFound(f"Job {job_id} has no stage {stage!r}")
    log_path = safe_join(settings.jobs_dir, job_id, "logs", f"{stage}.log")
    if not log_path.is_file():
        raise NotFound(f"No log yet for stage {stage!r}")

    def read_tail() -> str:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            if size > tail_bytes:
                handle.seek(size - tail_bytes)
                handle.readline()  # discard the partial first line
            return handle.read().decode("utf-8", errors="replace")

    return PlainTextResponse(await asyncio.to_thread(read_tail))
