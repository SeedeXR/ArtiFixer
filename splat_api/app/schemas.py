# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response models.

All models set ``extra="forbid"``: an unexpected field is a client bug or an
attempt to reach a parameter the API does not intend to expose, and silently
ignoring it is how injection bugs survive code review.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from splat_api.app.errors import BadRequest
from splat_api.app.paths import validate_image_name

PipelineMode = Literal["reconstruct", "artifixer3d", "artifixer3d_plus"]
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]

# Intrinsics required by camera_trajectories.camera_intrinsics_from_mapping
# (data_processing/camera_trajectories.py:55-79).
REQUIRED_INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy")
OPTIONAL_INTRINSIC_KEYS = ("k1", "k2", "p1", "p2")

# Static parse-time ceilings, deliberately above any sane configured limit. They
# exist to bound the cost of validating a hostile body, not to express policy;
# SPLAT_API_MAX_TRAJECTORY_FRAMES and SPLAT_API_MAX_IMAGES are the real limits.
MAX_TRAJECTORY_FRAMES_PARSE = 20_000
MAX_SELECTED_NAMES_PARSE = 20_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrajectoryFrame(StrictModel):
    """One target camera along a novel trajectory.

    ``file_path`` is absent by design: ``assert_target_only_trajectory``
    (data_processing/camera_trajectories.py:138-150) rejects frames that carry
    one, because those look like source/context images rather than render targets.
    """

    transform_matrix: list[list[float]] = Field(
        description="4x4 (or 3x4) OpenGL/NeRFStudio camera-to-world matrix."
    )
    fl_x: float | None = None
    fl_y: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float | None = None
    k2: float | None = None
    p1: float | None = None
    p2: float | None = None

    @field_validator("transform_matrix")
    @classmethod
    def _check_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        # normalize_pose_matrix (camera_trajectories.py:25-28) accepts 3x4 and 4x4.
        if len(value) not in (3, 4) or any(len(row) != 4 for row in value):
            raise ValueError("transform_matrix must be a 3x4 or 4x4 row-major matrix")
        for row in value:
            for entry in row:
                if not isinstance(entry, (int, float)) or isinstance(entry, bool):
                    raise ValueError("transform_matrix entries must be numbers")
                if entry != entry or entry in (float("inf"), float("-inf")):
                    raise ValueError("transform_matrix entries must be finite")
        if len(value) == 4 and value[3] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("transform_matrix bottom row must be [0, 0, 0, 1]")
        return value

    @field_validator("fl_x", "fl_y")
    @classmethod
    def _positive_focal(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("focal length must be positive")
        return value


class Trajectory(StrictModel):
    """A transforms-style novel camera path.

    Mirrors the schema in the ArtiFixer README and the validation in
    ``camera_trajectories.read_camera_trajectory``. Per-frame focal/principal
    overrides are allowed, but the renderer asserts a single resolution across the
    path (threedgrut/render.py:503-506), so ``w``/``h`` are top-level only.
    """

    camera_model: Literal["OPENCV"] = "OPENCV"
    w: int = Field(gt=0, le=8192)
    h: int = Field(gt=0, le=8192)
    fl_x: float = Field(gt=0)
    fl_y: float = Field(gt=0)
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    # Bounded at parse time, not just by SPLAT_API_MAX_TRAJECTORY_FRAMES: pydantic
    # materializes every frame before any handler code runs, so a body full of
    # frames would otherwise cost seconds of event-loop time and gigabytes of RSS
    # before the configured limit could reject it.
    frames: list[TrajectoryFrame] = Field(min_length=1, max_length=MAX_TRAJECTORY_FRAMES_PARSE)

    def to_transforms_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_model": self.camera_model,
            "w": self.w,
            "h": self.h,
            "fl_x": self.fl_x,
            "fl_y": self.fl_y,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "frames": [],
        }
        for frame in self.frames:
            entry: dict[str, Any] = {"transform_matrix": frame.transform_matrix}
            for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"):
                value = getattr(frame, key)
                if value is not None:
                    entry[key] = value
            payload["frames"].append(entry)
        return payload


class JobCreate(StrictModel):
    """Job submission.

    Deliberately *not* exposed: raw Hydra/3DGRUT config overrides, arbitrary
    output paths, and the model id. Those would let a caller reach into config
    composition or the filesystem; see docs/SECURITY.md.
    """

    scene_id: str = Field(description="Scene created by POST /v1/scenes.")
    mode: PipelineMode = Field(
        default="reconstruct",
        description=(
            "reconstruct: 3DGUT reconstruction only (no ArtiFixer checkpoint needed). "
            "artifixer3d: reconstruction + ArtiFixer correction distilled into a new splat. "
            "artifixer3d_plus: adds a second ArtiFixer pass over the ArtiFixer3D renders."
        ),
    )
    # Upper bounds here are a sanity ceiling; the operative limit is
    # SPLAT_API_RECONSTRUCTION_STEPS_MAX / _ARTIFIXER3D_STEPS_MAX, which also
    # rejects with 422.
    reconstruction_steps: int | None = Field(default=None, ge=100, le=1_000_000)
    artifixer3d_steps: int | None = Field(default=None, ge=100, le=1_000_000)
    inference_steps: int = Field(default=4, ge=1, le=50, description="ArtiFixer denoising steps.")
    selected_image_names: (
        list[Annotated[str, Field(min_length=1, max_length=128)]] | None
    ) = Field(
        default=None,
        max_length=MAX_SELECTED_NAMES_PARSE,
        description=(
            "Subset of scene images used as real 3DGRUT anchors. Every other frame becomes an "
            "ArtiFixer target. Required for artifixer3d modes unless a trajectory is supplied."
        ),
    )
    metric_scale: float | None = Field(
        default=None,
        gt=0,
        le=1e6,
        description="Skip MoGe metric alignment by supplying a known scale factor.",
    )
    trajectory: Trajectory | None = Field(
        default=None, description="Optional novel camera path to render and correct."
    )
    export_ply: bool = Field(default=True, description="Export a 3DGS-compatible PLY splat.")
    client_token: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Idempotency key. Re-submitting the same token returns the original job.",
    )

    @field_validator("selected_image_names")
    @classmethod
    def _check_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("selected_image_names must not be empty")
        for name in value:
            try:
                validate_image_name(name)
            except BadRequest as exc:
                # Surface as a pydantic validation error so a malformed body is a
                # 422 like every other schema violation, not a bare 400.
                raise ValueError(str(exc)) from exc
        if len(set(value)) != len(value):
            raise ValueError("selected_image_names contains duplicates")
        return value

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> JobCreate:
        if self.mode != "reconstruct" and self.selected_image_names is None and self.trajectory is None:
            raise ValueError(
                "artifixer3d modes need at least one generated view: supply selected_image_names "
                "(a strict subset of the scene images) or a trajectory"
            )
        if self.mode == "reconstruct" and self.trajectory is not None:
            raise ValueError(
                "mode=reconstruct does not run ArtiFixer, so a trajectory would only be rendered; "
                "use mode=artifixer3d to correct a novel path"
            )
        return self


class SceneRegister(StrictModel):
    """Register a COLMAP scene that already exists on a server-side path.

    Admin-only: it names a filesystem path, so it is an authorization decision,
    not a convenience. The path is still confined to the ``<data_root>/import``
    root.
    """

    path: str = Field(min_length=1, max_length=4096)
    label: str | None = Field(default=None, max_length=128)


class StageInfo(StrictModel):
    name: str
    state: Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled"]
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    message: str | None = None


class ArtifactInfo(StrictModel):
    name: str
    kind: Literal["splat_ply", "splat_checkpoint", "frames", "video", "metadata", "log"]
    size_bytes: int
    sha256: str | None = None
    download_path: str | None = None
    description: str | None = None


class SceneInfo(StrictModel):
    scene_id: str
    created_at: str
    label: str | None = None
    source: Literal["upload", "registered"]
    image_count: int
    camera_count: int
    camera_models: list[str]
    point_count: int
    colmap_width: int
    colmap_height: int
    size_bytes: int
    image_names: list[str] | None = None


class JobInfo(StrictModel):
    job_id: str
    scene_id: str
    state: JobState
    mode: PipelineMode
    progress: float = Field(ge=0.0, le=1.0)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None
    gpu_index: int | None = None
    stages: list[StageInfo]
    artifacts: list[ArtifactInfo]
    error: str | None = None
    request: dict[str, Any]


class JobList(StrictModel):
    jobs: list[JobInfo]
    next_cursor: str | None = None


class Capabilities(StrictModel):
    service_version: str
    modes: list[PipelineMode]
    artifixer_checkpoint_configured: bool
    artifixer_model_id: str
    gpu_count: int
    max_concurrent_jobs: int
    queue_capacity: int
    limits: dict[str, Any]


class HealthStatus(StrictModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, Any]
