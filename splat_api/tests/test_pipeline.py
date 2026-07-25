# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for the stage/command mapping.

``TestAgainstRepoLayout`` is the drift guard: it compares every path this module
computes against the same path computed by the repo's own helpers. If a future
ArtiFixer release moves a checkpoint or renames a run directory, these tests fail
rather than the service looking for artifacts that are no longer there.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from splat_api.app import pipeline
from splat_api.app.config import Settings


@pytest.fixture
def paths(tmp_path: Path) -> pipeline.JobPaths:
    return pipeline.JobPaths(
        job_dir=tmp_path / "jobs" / "job_abc",
        scene_dir=tmp_path / "scenes" / "scene_xyz",
        scene_id="scene_xyz",
        reconstruction_steps=10_000,
        artifixer3d_steps=30_000,
    )


class TestStageSequences:
    def test_reconstruct_is_prepare_then_export(self) -> None:
        assert pipeline.stage_sequence("reconstruct", export_ply=True) == ("prepare", "export")

    def test_artifixer3d_adds_correction_and_distillation(self) -> None:
        assert pipeline.stage_sequence("artifixer3d", export_ply=True) == (
            "prepare",
            "artifixer",
            "artifixer3d",
            "export",
        )

    def test_plus_adds_a_second_inference_pass(self) -> None:
        assert pipeline.stage_sequence("artifixer3d_plus", export_ply=True) == (
            "prepare",
            "artifixer",
            "artifixer3d",
            "artifixer3d_plus",
            "export",
        )

    def test_export_can_be_omitted(self) -> None:
        assert "export" not in pipeline.stage_sequence("artifixer3d", export_ply=False)

    def test_reconstruct_skips_scale_and_caption_phases(self) -> None:
        """MoGe and Qwen3-VL only condition ArtiFixer, so a plain splat skips them."""
        assert pipeline.prepare_phases("reconstruct") == "prepare,reconstruct,render"
        assert pipeline.prepare_phases("artifixer3d") == "prepare,reconstruct,render,scale,caption"


class TestPrepareCommand:
    def _build(self, settings: Settings, paths: pipeline.JobPaths, **overrides) -> pipeline.StageCommand:
        kwargs: dict = {
            "mode": "artifixer3d",
            "has_selected_names": False,
            "has_trajectory": False,
            "metric_scale": None,
            "cached_checkpoint": None,
        }
        kwargs.update(overrides)
        return pipeline.prepare_command(settings, paths, **kwargs)

    def test_invokes_the_repo_module(self, settings: Settings, paths: pipeline.JobPaths) -> None:
        argv = self._build(settings, paths).argv
        assert argv[1:3] == ("-m", "data_processing.prepare_colmap_artifixer_inputs")
        assert "--colmap_dir" in argv and str(paths.scene_dir) in argv
        assert "--output_root" in argv and str(paths.prepared_root) in argv

    def test_optional_inputs_are_omitted_when_absent(
        self, settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        argv = self._build(settings, paths).argv
        assert "--selected_image_names_file" not in argv
        assert "--trajectory_path" not in argv
        assert "--metric_scale" not in argv
        assert "--reconstruction_checkpoint" not in argv

    def test_optional_inputs_are_passed_when_present(
        self, settings: Settings, paths: pipeline.JobPaths, tmp_path: Path
    ) -> None:
        cached = tmp_path / "cached.pt"
        argv = self._build(
            settings,
            paths,
            has_selected_names=True,
            has_trajectory=True,
            metric_scale=1.5,
            cached_checkpoint=cached,
        ).argv
        assert str(paths.selected_names_file) in argv
        assert str(paths.trajectory_file) in argv
        assert str(cached) in argv
        assert "1.5" in argv

    def test_metric_scale_is_never_a_shell_string(
        self, settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        """Numeric parameters are re-serialized from floats, not echoed."""
        argv = self._build(settings, paths, metric_scale=0.1 + 0.2).argv
        value = argv[argv.index("--metric_scale") + 1]
        assert float(value) == 0.1 + 0.2
        assert all(character not in value for character in ";|&$`\n ")

    def test_argv_has_no_shell_metacharacters(
        self, settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        for part in self._build(settings, paths).argv:
            assert "\n" not in part and "\x00" not in part


class TestInferenceCommands:
    def test_uses_reconstructed_colmap_evalset(
        self, artifixer_settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        argv = pipeline.artifixer_command(
            artifixer_settings, paths, has_trajectory=False, inference_steps=4
        ).argv
        assert argv[1:3] == ("-m", "model_eval.run_inference")
        assert argv[argv.index("--evalset") + 1] == "reconstructed_colmap"
        assert argv[argv.index("--checkpoint_pt") + 1] == str(artifixer_settings.artifixer_checkpoint)
        assert argv[argv.index("--model_id") + 1] == artifixer_settings.artifixer_model_id
        assert "--save_frame_outputs_only" in argv

    def test_render_trajectory_matches_the_split_kind(
        self, artifixer_settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        # reconstructed_colmap_eval.py:160-162 rejects all_frames on a trajectory
        # split, and :223-225 requires target_indices for `trajectory`.
        plain = pipeline.artifixer_command(
            artifixer_settings, paths, has_trajectory=False, inference_steps=4
        ).argv
        traj = pipeline.artifixer_command(
            artifixer_settings, paths, has_trajectory=True, inference_steps=4
        ).argv
        assert plain[plain.index("--render_trajectory") + 1] == "all_frames"
        assert traj[traj.index("--render_trajectory") + 1] == "trajectory"

    def test_plus_pass_reads_the_generated_split(
        self, artifixer_settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        argv = pipeline.artifixer3d_plus_command(
            artifixer_settings, paths, has_trajectory=False, inference_steps=4
        ).argv
        assert argv[argv.index("--split_path") + 1] == str(paths.artifixer3d_plus_split_path)
        assert argv[argv.index("--save_dir") + 1] == str(paths.artifixer3d_plus_save_dir)

    def test_requires_a_configured_checkpoint(
        self, settings: Settings, paths: pipeline.JobPaths
    ) -> None:
        with pytest.raises(AssertionError, match="configured checkpoint"):
            pipeline.artifixer_command(settings, paths, has_trajectory=False, inference_steps=4)


class TestArtifixer3DCommand:
    def test_passes_job_scoped_outputs(
        self, artifixer_settings: Settings, paths: pipeline.JobPaths, tmp_path: Path
    ) -> None:
        frames = tmp_path / "frames"
        argv = pipeline.artifixer3d_command(
            artifixer_settings, paths, frames_dir=frames, base_checkpoint=None
        ).argv
        assert argv[1:3] == ("-m", "data_processing.run_artifixer3d")
        assert argv[argv.index("--scene_root") + 1] == str(paths.prepared_root)
        assert argv[argv.index("--artifixer_frames_dir") + 1] == str(frames)
        # Job-scoped so two jobs on one scene cannot overwrite each other.
        assert argv[argv.index("--output_root") + 1] == str(paths.artifixer3d_root)
        assert argv[argv.index("--artifixer3d_plus_inference_split_path") + 1] == str(
            paths.artifixer3d_plus_split_path
        )
        assert "--no-use_wandb" in argv

    def test_resume_flag_is_opt_in(
        self, artifixer_settings: Settings, paths: pipeline.JobPaths, tmp_path: Path
    ) -> None:
        without = pipeline.artifixer3d_command(
            artifixer_settings, paths, frames_dir=tmp_path, base_checkpoint=None
        ).argv
        with_resume = pipeline.artifixer3d_command(
            artifixer_settings,
            paths,
            frames_dir=tmp_path,
            base_checkpoint=paths.reconstruction_checkpoint,
        ).argv
        assert "--base_checkpoint" not in without
        assert with_resume[with_resume.index("--base_checkpoint") + 1] == str(
            paths.reconstruction_checkpoint
        )


class TestReconstructionCache:
    def test_key_depends_on_steps_and_view_set(self) -> None:
        base = pipeline.reconstruction_cache_key(scene_id="s", steps=100, selected_image_names=None)
        other_steps = pipeline.reconstruction_cache_key(
            scene_id="s", steps=200, selected_image_names=None
        )
        subset = pipeline.reconstruction_cache_key(
            scene_id="s", steps=100, selected_image_names=["a.jpg"]
        )
        other_scene = pipeline.reconstruction_cache_key(
            scene_id="t", steps=100, selected_image_names=None
        )
        assert len({base, other_steps, subset, other_scene}) == 4

    def test_view_order_does_not_matter(self) -> None:
        first = pipeline.reconstruction_cache_key(
            scene_id="s", steps=100, selected_image_names=["a.jpg", "b.jpg"]
        )
        second = pipeline.reconstruction_cache_key(
            scene_id="s", steps=100, selected_image_names=["b.jpg", "a.jpg"]
        )
        assert first == second

    def test_all_views_differs_from_an_equivalent_explicit_list(self) -> None:
        """A cache collision here would train on the wrong views."""
        implicit = pipeline.reconstruction_cache_key(
            scene_id="s", steps=100, selected_image_names=None
        )
        explicit = pipeline.reconstruction_cache_key(
            scene_id="s", steps=100, selected_image_names=["__all__"]
        )
        assert implicit != explicit


class TestSubprocessEnvironment:
    def test_only_allowlisted_names_are_forwarded(self, settings: Settings) -> None:
        base = {"PATH": "/usr/bin", "HF_HOME": "/hf", "AWS_SECRET_ACCESS_KEY": "leak-me"}
        env = pipeline.build_subprocess_env(settings, gpu_index=None, base_env=base)
        assert env["PATH"] == "/usr/bin"
        assert env["HF_HOME"] == "/hf"
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_pins_the_assigned_gpu(self, settings: Settings) -> None:
        env = pipeline.build_subprocess_env(settings, gpu_index=3, base_env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_omits_cuda_visible_devices_when_unassigned(self, settings: Settings) -> None:
        env = pipeline.build_subprocess_env(settings, gpu_index=None, base_env={})
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_repo_is_importable_and_wandb_disabled(self, settings: Settings) -> None:
        env = pipeline.build_subprocess_env(settings, gpu_index=None, base_env={})
        assert env["PYTHONPATH"] == str(settings.repo_root)
        assert env["WANDB_MODE"] == "disabled"


class TestAgainstRepoLayout:
    """Compare our derived paths with the repo's own path helpers."""

    def test_prepared_scene_layout(self, paths: pipeline.JobPaths) -> None:
        prepare = pytest.importorskip(
            "data_processing.prepare_colmap_artifixer_inputs", reason="repo modules unavailable"
        )
        reference = prepare.prepared_paths(
            paths.prepared_root, paths.scene_id, paths.reconstruction_steps
        )
        assert paths.split_path == reference.split_path
        assert paths.reconstruction_render_dir == reference.render_checkpoint_dir
        assert paths.prepared_root / "metric_alignment" == reference.scale_dir

    def test_reconstruction_checkpoint_path(self, paths: pipeline.JobPaths) -> None:
        prepare = pytest.importorskip(
            "data_processing.prepare_colmap_artifixer_inputs", reason="repo modules unavailable"
        )
        import argparse

        reference_paths = prepare.prepared_paths(
            paths.prepared_root, paths.scene_id, paths.reconstruction_steps
        )
        args = argparse.Namespace(
            reconstruction_checkpoint=None, reconstruction_steps=paths.reconstruction_steps
        )
        assert paths.reconstruction_checkpoint == prepare.reconstruction_checkpoint(
            reference_paths, args
        )

    def test_artifixer3d_checkpoint_and_render_paths(self, paths: pipeline.JobPaths) -> None:
        artifixer3d = pytest.importorskip(
            "data_processing.artifixer3d", reason="repo modules unavailable"
        )
        scene = artifixer3d.PreparedScene(
            scene_id=paths.scene_id,
            scene_root=paths.prepared_root,
            transforms_path=paths.prepared_root / "transforms.json",
            colmap_dir=paths.scene_dir,
            prompt_path=paths.prepared_root / "captions" / "caption.h5",
            camera_scale=0.01,
            has_gt=True,
            selected_indices=[0],
            target_indices_path=None,
            reconstruction_checkpoint=paths.reconstruction_checkpoint,
            frame_count=2,
        )
        reference = artifixer3d.artifixer3d_paths(
            scene, paths.artifixer3d_root, paths.artifixer3d_plus_split_path, paths.artifixer3d_steps
        )
        assert paths.artifixer3d_render_dir == reference.render_checkpoint_dir
        assert paths.artifixer3d_checkpoint == artifixer3d.artifixer3d_checkpoint(
            scene, reference, paths.artifixer3d_steps
        )
        assert paths.artifixer3d_plus_split_path == reference.artifixer3d_plus_inference_split_path

    def test_inference_output_directory(self, artifixer_settings: Settings, tmp_path: Path) -> None:
        run_inference = pytest.importorskip(
            "model_eval.run_inference", reason="model_eval requires torch and diffusers"
        )
        import argparse

        save_dir = tmp_path / "save"
        for render_trajectory, has_trajectory in (("all_frames", False), ("trajectory", True)):
            args = argparse.Namespace(
                checkpoint_pt=artifixer_settings.artifixer_checkpoint,
                checkpoint_dir=None,
                sink_size=pipeline.SINK_SIZE,
                render_trajectory=render_trajectory,
                num_views=None,
                evalset="reconstructed_colmap",
                neighbor_selection_mode=pipeline.NEIGHBOR_SELECTION_MODE,
                save_dir=save_dir,
                output_suffix="",
            )
            assert pipeline.inference_output_dir(
                save_dir, artifixer_settings.artifixer_checkpoint, has_trajectory=has_trajectory
            ) == run_inference.get_output_dir(args)

    def test_default_num_views_constant(self) -> None:
        evalsets = pytest.importorskip(
            "model_eval.reconstructed_colmap_evalsets", reason="repo modules unavailable"
        )
        assert (
            pipeline.DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS
            == evalsets.DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS
        )

    def test_recon_experiment_name_constant(self) -> None:
        prepare = pytest.importorskip(
            "data_processing.prepare_colmap_artifixer_inputs", reason="repo modules unavailable"
        )
        assert pipeline.RECON_EXPERIMENT == prepare.DEFAULT_RECON_SUBDIR

    def test_artifixer3d_experiment_name_constant(self) -> None:
        artifixer3d = pytest.importorskip(
            "data_processing.artifixer3d", reason="repo modules unavailable"
        )
        assert pipeline.ARTIFIXER3D_EXPERIMENT == artifixer3d.ARTIFIXER3D_EXPERIMENT

    def test_prepare_phases_are_accepted_by_the_repo(self) -> None:
        prepare = pytest.importorskip(
            "data_processing.prepare_colmap_artifixer_inputs", reason="repo modules unavailable"
        )
        for mode in ("reconstruct", "artifixer3d"):
            parsed = prepare.parse_phases(pipeline.prepare_phases(mode))
            assert parsed  # raises inside parse_phases if a phase name is unknown

    def test_prediction_frames_path_matches_the_readme_contract(self, tmp_path: Path) -> None:
        expected = tmp_path / "scene_a" / "frames" / "batch_0000" / "pred"
        assert pipeline.predicted_frames_dir(tmp_path, "scene_a") == expected
