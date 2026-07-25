# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage definitions: how a job becomes ArtiFixer command lines.

This module is the single place that knows the repo's CLI contract and output
layout. It is pure: it computes argv lists and paths but never runs anything, so
the whole mapping is unit-testable without a GPU (see tests/test_pipeline.py).

Every path below is derived from the corresponding source location:

* prepared scene layout ....... data_processing/prepare_colmap_artifixer_inputs.py:100-119
* reconstruction checkpoint ... prepare_colmap_artifixer_inputs.py:484-493
* render directory ............ data_processing/render_3dgrut_colmap.py:40-42
* inference output directory .. model_eval/run_inference.py:297-321
* prediction frames ........... model_eval/run_inference.py:585-622
* ArtiFixer3D layout .......... data_processing/artifixer3d.py:207-232
"""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from splat_api.app.config import Settings

# --- constants mirrored from the repo -------------------------------------

# model_eval/reconstructed_colmap_evalsets.py: DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS
DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS = 12
# run_inference defaults we pin explicitly so the derived run-name cannot drift.
NEIGHBOR_SELECTION_MODE = "evenly_spaced"
SINK_SIZE = 7
# prepare_colmap_artifixer_inputs.py:56 DEFAULT_RECON_SUBDIR
RECON_EXPERIMENT = "reconstruction"
# data_processing/artifixer3d.py:37 ARTIFIXER3D_EXPERIMENT
ARTIFIXER3D_EXPERIMENT = "artifixer3d"

STAGE_PREPARE = "prepare"
STAGE_ARTIFIXER = "artifixer"
STAGE_ARTIFIXER3D = "artifixer3d"
STAGE_ARTIFIXER3D_PLUS = "artifixer3d_plus"
STAGE_EXPORT = "export"

STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "reconstruct": (STAGE_PREPARE, STAGE_EXPORT),
    "artifixer3d": (STAGE_PREPARE, STAGE_ARTIFIXER, STAGE_ARTIFIXER3D, STAGE_EXPORT),
    "artifixer3d_plus": (
        STAGE_PREPARE,
        STAGE_ARTIFIXER,
        STAGE_ARTIFIXER3D,
        STAGE_ARTIFIXER3D_PLUS,
        STAGE_EXPORT,
    ),
}

# Environment names forwarded to worker subprocesses. Everything else is dropped:
# the child gets a constructed environment, never the API process's own.
FORWARDED_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "TORCH_EXTENSIONS_DIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_CACHE",
    "MOGE_MODEL_PATH",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
)


@dataclass(frozen=True)
class StageCommand:
    """One subprocess invocation."""

    name: str
    argv: tuple[str, ...]
    description: str
    env: dict[str, str] = field(default_factory=dict)

    def display(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


@dataclass(frozen=True)
class JobPaths:
    """All filesystem locations a job reads or writes.

    ``prepared_root.name`` becomes the prepared scene id
    (prepare_colmap_artifixer_inputs.py:804 uses ``output_root.name``) and is
    interpolated into 3DGRUT Hydra overrides as ``experiment_name``, so it must
    stay within the restricted identifier alphabet enforced by
    ``splat_api.app.paths.validate_id``.
    """

    job_dir: Path
    scene_dir: Path
    scene_id: str
    reconstruction_steps: int
    artifixer3d_steps: int

    @property
    def prepared_root(self) -> Path:
        return self.job_dir / "prep" / self.scene_id

    @property
    def selected_names_file(self) -> Path:
        return self.job_dir / "inputs" / "selected_train_images.txt"

    @property
    def trajectory_file(self) -> Path:
        return self.job_dir / "inputs" / "trajectory.json"

    @property
    def split_path(self) -> Path:
        return self.prepared_root / "split.json"

    @property
    def scale_info_path(self) -> Path:
        return self.prepared_root / "metric_alignment" / "scale_info.txt"

    @property
    def reconstruction_checkpoint(self) -> Path:
        # prepare_colmap_artifixer_inputs.reconstruction_output_dir (:484-486)
        return (
            self.prepared_root
            / "3dgrut_runs"
            / self.scene_id
            / self.scene_id
            / f"ours_{self.reconstruction_steps}"
            / f"ckpt_{self.reconstruction_steps}.pt"
        )

    @property
    def reconstruction_render_dir(self) -> Path:
        # prepare_colmap_artifixer_inputs.prepared_paths (:104)
        return (
            self.prepared_root
            / "recon_results"
            / self.scene_id
            / RECON_EXPERIMENT
            / self.scene_id
            / f"ours_{self.reconstruction_steps}"
        )

    @property
    def artifixer_save_dir(self) -> Path:
        return self.job_dir / "artifixer"

    @property
    def artifixer3d_root(self) -> Path:
        return self.job_dir / "artifixer3d"

    @property
    def artifixer3d_plus_split_path(self) -> Path:
        return self.job_dir / "split_artifixer3d_plus.json"

    @property
    def artifixer3d_plus_save_dir(self) -> Path:
        return self.job_dir / "artifixer3d_plus"

    @property
    def artifixer3d_checkpoint(self) -> Path:
        # artifixer3d.artifixer3d_checkpoint (:230-232)
        return (
            self.artifixer3d_root
            / "runs"
            / self.scene_id
            / self.scene_id
            / f"ours_{self.artifixer3d_steps}"
            / f"ckpt_{self.artifixer3d_steps}.pt"
        )

    @property
    def artifixer3d_render_dir(self) -> Path:
        # artifixer3d.artifixer3d_paths (:216-218)
        return (
            self.artifixer3d_root
            / "recon_results"
            / self.scene_id
            / ARTIFIXER3D_EXPERIMENT
            / self.scene_id
            / f"ours_{self.artifixer3d_steps}"
        )

    @property
    def output_dir(self) -> Path:
        return self.job_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.job_dir / "logs"

    @property
    def splat_ply_path(self) -> Path:
        return self.output_dir / "splat.ply"

    @property
    def frames_archive_path(self) -> Path:
        return self.output_dir / "corrected_frames.zip"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def splat_stats_path(self) -> Path:
        return self.output_dir / "splat_stats.json"


def prepare_phases(mode: str) -> str:
    """Phases for ``prepare_colmap_artifixer_inputs --phases``.

    ``reconstruct`` mode stops after rendering: metric scale (MoGe) and captions
    (Qwen3-VL) exist only to condition ArtiFixer, so running them would download
    tens of GB of weights and add GPU minutes for no effect on the splat.
    """
    if mode == "reconstruct":
        return "prepare,reconstruct,render"
    return "prepare,reconstruct,render,scale,caption"


def render_trajectory_flag(has_trajectory: bool) -> str:
    """``--render_trajectory`` value.

    A trajectory split carries ``target_indices_path``, and
    reconstructed_colmap_eval.py:160-162 rejects ``all_frames`` for such splits;
    conversely ``trajectory`` requires that field (:223-225).
    """
    return "trajectory" if has_trajectory else "all_frames"


def inference_run_name(*, has_trajectory: bool) -> str:
    """Reproduce ``run_inference.get_output_dir``'s run-name component.

    run_inference.py:300-320. ``num_views`` is left unset for
    reconstructed_colmap, which yields the ``auto<N>`` form.
    """
    sink_suffix = f"_sink{SINK_SIZE}" if SINK_SIZE > 0 else ""
    trajectory_suffix = "_trajectory" if has_trajectory else "_all_frames"
    num_views = f"auto{DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS}"
    return (
        f"distilled_views_reconstructed_colmap_{num_views}_"
        f"{NEIGHBOR_SELECTION_MODE}{sink_suffix}{trajectory_suffix}"
    )


def inference_output_dir(save_dir: Path, checkpoint: Path, *, has_trajectory: bool) -> Path:
    """``<save_dir>/<checkpoint stem>/<run name>``.

    Checkpoint-name component: ``checkpoint_loading.checkpoint_output_name``
    (:26-32) uses ``checkpoint_pt.stem``.
    """
    return save_dir / checkpoint.stem / inference_run_name(has_trajectory=has_trajectory)


def predicted_frames_dir(inference_dir: Path, scene_id: str) -> Path:
    """``<inference dir>/<scene id>/frames/batch_0000/pred`` (run_inference.py:470-585)."""
    return inference_dir / scene_id / "frames" / "batch_0000" / "pred"


def reconstruction_cache_key(
    *, scene_id: str, steps: int, selected_image_names: list[str] | None
) -> str:
    """Identify a reusable 3DGUT checkpoint.

    The reconstruction is trained on the selected views only
    (``selected_indices_file`` override), so the view set is part of the identity
    of the checkpoint. Order is normalized because view selection is a set.
    """
    digest = hashlib.sha256()
    digest.update(scene_id.encode())
    digest.update(b"\x00")
    digest.update(str(steps).encode())
    digest.update(b"\x00")
    if selected_image_names is None:
        digest.update(b"__all__")
    else:
        for name in sorted(selected_image_names):
            digest.update(name.encode())
            digest.update(b"\x1f")
    return digest.hexdigest()[:32]


def prepare_command(
    settings: Settings,
    paths: JobPaths,
    *,
    mode: str,
    has_selected_names: bool,
    has_trajectory: bool,
    metric_scale: float | None,
    cached_checkpoint: Path | None,
) -> StageCommand:
    argv = [
        settings.python_executable,
        "-m",
        "data_processing.prepare_colmap_artifixer_inputs",
        "--colmap_dir",
        str(paths.scene_dir),
        "--output_root",
        str(paths.prepared_root),
        "--phases",
        prepare_phases(mode),
        "--reconstruction_steps",
        str(paths.reconstruction_steps),
    ]
    if has_selected_names:
        argv += ["--selected_image_names_file", str(paths.selected_names_file)]
    if has_trajectory:
        argv += ["--trajectory_path", str(paths.trajectory_file)]
    if metric_scale is not None:
        # prepare_colmap_artifixer_inputs.run_metric_alignment (:647-660) writes the
        # scale file directly and skips MoGe when this is supplied.
        argv += ["--metric_scale", repr(float(metric_scale))]
    if cached_checkpoint is not None:
        argv += ["--reconstruction_checkpoint", str(cached_checkpoint)]
    if settings.artifixer_model_id:
        argv += ["--text_encoder_model_id", settings.artifixer_model_id]
    return StageCommand(
        name=STAGE_PREPARE,
        argv=tuple(argv),
        description=(
            "Prepare the COLMAP scene, train the 3DGUT reconstruction, render it"
            + (", estimate metric scale and generate caption embeddings" if mode != "reconstruct" else "")
        ),
    )


def _inference_command(
    settings: Settings,
    *,
    name: str,
    split_path: Path,
    save_dir: Path,
    has_trajectory: bool,
    inference_steps: int,
    description: str,
) -> StageCommand:
    assert settings.artifixer_checkpoint is not None, "ArtiFixer stages require a configured checkpoint"
    argv = (
        settings.python_executable,
        "-m",
        "model_eval.run_inference",
        "--evalset",
        "reconstructed_colmap",
        "--checkpoint_pt",
        str(settings.artifixer_checkpoint),
        "--model_id",
        settings.artifixer_model_id,
        "--save_dir",
        str(save_dir),
        "--split_path",
        str(split_path),
        "--render_trajectory",
        render_trajectory_flag(has_trajectory),
        "--num_inference_steps",
        str(inference_steps),
        "--neighbor_selection_mode",
        NEIGHBOR_SELECTION_MODE,
        "--sink_size",
        str(SINK_SIZE),
        "--save_frame_outputs_only",
    )
    return StageCommand(name=name, argv=argv, description=description)


def artifixer_command(
    settings: Settings, paths: JobPaths, *, has_trajectory: bool, inference_steps: int
) -> StageCommand:
    return _inference_command(
        settings,
        name=STAGE_ARTIFIXER,
        split_path=paths.split_path,
        save_dir=paths.artifixer_save_dir,
        has_trajectory=has_trajectory,
        inference_steps=inference_steps,
        description="Correct the reconstruction renders with the ArtiFixer video model",
    )


def artifixer3d_plus_command(
    settings: Settings, paths: JobPaths, *, has_trajectory: bool, inference_steps: int
) -> StageCommand:
    return _inference_command(
        settings,
        name=STAGE_ARTIFIXER3D_PLUS,
        split_path=paths.artifixer3d_plus_split_path,
        save_dir=paths.artifixer3d_plus_save_dir,
        has_trajectory=has_trajectory,
        inference_steps=inference_steps,
        description="Run ArtiFixer again over the ArtiFixer3D renders (ArtiFixer3D+)",
    )


def artifixer3d_command(
    settings: Settings, paths: JobPaths, *, frames_dir: Path, base_checkpoint: Path | None
) -> StageCommand:
    argv = [
        settings.python_executable,
        "-m",
        "data_processing.run_artifixer3d",
        "--scene_root",
        str(paths.prepared_root),
        "--split_path",
        str(paths.split_path),
        "--scene_id",
        paths.scene_id,
        "--artifixer_frames_dir",
        str(frames_dir),
        "--output_root",
        str(paths.artifixer3d_root),
        "--artifixer3d_plus_inference_split_path",
        str(paths.artifixer3d_plus_split_path),
        "--artifixer3d_steps",
        str(paths.artifixer3d_steps),
        "--no-use_wandb",
    ]
    if base_checkpoint is not None:
        # artifixer3d.train_artifixer3d (:553-556) turns this into a 3DGRUT
        # `resume=` override, warm-starting from the base reconstruction.
        argv += ["--base_checkpoint", str(base_checkpoint)]
    return StageCommand(
        name=STAGE_ARTIFIXER3D,
        argv=tuple(argv),
        description="Distill the corrected frames into a new 3DGRUT splat (ArtiFixer3D)",
    )


def export_command(
    settings: Settings, *, checkpoint: Path, output: Path, stats_path: Path | None = None
) -> StageCommand:
    argv = [
        settings.python_executable,
        "-m",
        "splat_api.app.exporter",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if stats_path is not None:
        argv += ["--stats-json", str(stats_path)]
    return StageCommand(
        name=STAGE_EXPORT,
        argv=tuple(argv),
        description="Export the 3DGRUT checkpoint as a 3DGS-compatible PLY splat",
    )


def stage_sequence(mode: str, *, export_ply: bool) -> tuple[str, ...]:
    stages = STAGE_SEQUENCES[mode]
    if not export_ply:
        stages = tuple(stage for stage in stages if stage != STAGE_EXPORT)
    return stages


def build_subprocess_env(settings: Settings, *, gpu_index: int | None, base_env: dict[str, str]) -> dict[str, str]:
    """Construct a minimal environment for a stage.

    Built from an allowlist rather than inherited, so secrets in the API
    process's environment (other services' tokens, for instance) cannot leak into
    a pipeline that writes user-visible logs.
    """
    env: dict[str, str] = {}
    for name in FORWARDED_ENV:
        value = base_env.get(name)
        if value:
            env[name] = value
    env.update(settings.extra_env)
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the repo importable regardless of how the service itself was started.
    env["PYTHONPATH"] = str(settings.repo_root)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["OMP_NUM_THREADS"] = "8"
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env
