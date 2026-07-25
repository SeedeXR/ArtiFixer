# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for checkpoint-to-PLY export.

``TestMatchesUpstreamExporter`` is what licenses re-implementing the writer: it
asserts our output is byte-identical to ``threedgrut.export.ply_exporter``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splat_api.app.exporter import (
    ExportError,
    attribute_names,
    build_vertex_array,
    export_checkpoint,
    main,
)
from splat_api.tests.fake_checkpoint import fake_model_parameters, write_fake_checkpoint

torch = pytest.importorskip("torch", reason="exporter needs torch to read checkpoints")


class TestAttributeLayout:
    def test_order_matches_the_3dgs_convention(self) -> None:
        names = attribute_names(3, 45)
        assert names[:6] == ["x", "y", "z", "nx", "ny", "nz"]
        assert names[6:9] == ["f_dc_0", "f_dc_1", "f_dc_2"]
        assert names[9] == "f_rest_0"
        assert names[53] == "f_rest_44"
        assert names[54:] == [
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]

    def test_sh_degree_is_derived_from_the_specular_width(self) -> None:
        for degree in (0, 1, 2, 3):
            payload = fake_model_parameters(num_gaussians=8, sh_degree=degree)
            _, derived = build_vertex_array(payload)
            assert derived == degree

    def test_specular_coefficients_are_channel_major(self) -> None:
        """f_rest must be grouped by channel, matching the upstream transpose."""
        payload = fake_model_parameters(num_gaussians=2, sh_degree=1)
        specular = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9)
        payload["features_specular"] = specular
        vertices, _ = build_vertex_array(payload)
        expected = specular.reshape(2, 3, 3).permute(0, 2, 1).reshape(2, 9).numpy()
        actual = np.stack([vertices[f"f_rest_{index}"] for index in range(9)], axis=1)
        assert np.array_equal(actual, expected)

    def test_values_are_stored_preactivation(self) -> None:
        payload = fake_model_parameters(num_gaussians=4)
        vertices, _ = build_vertex_array(payload)
        assert np.allclose(vertices["opacity"], payload["density"].numpy().ravel())
        assert np.allclose(vertices["scale_0"], payload["scale"][:, 0].numpy())
        assert np.allclose(vertices["rot_0"], payload["rotation"][:, 0].numpy())

    def test_normals_are_the_upstream_placeholder(self) -> None:
        vertices, _ = build_vertex_array(fake_model_parameters(num_gaussians=3))
        assert np.array_equal(vertices["nx"], np.zeros(3, dtype=np.float32))
        assert np.array_equal(vertices["nz"], np.ones(3, dtype=np.float32))


class TestRejections:
    def test_rejects_a_non_gaussian_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.pt"
        torch.save({"state_dict": {"weight": torch.zeros(2)}}, path)
        with pytest.raises(ExportError, match="missing 3DGRUT model parameters"):
            export_checkpoint(path, tmp_path / "out.ply")

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="Checkpoint not found"):
            export_checkpoint(tmp_path / "nope.pt", tmp_path / "out.ply")

    def test_rejects_an_empty_model(self, tmp_path: Path) -> None:
        payload = fake_model_parameters(num_gaussians=0)
        with pytest.raises(ExportError, match="zero Gaussians"):
            build_vertex_array(payload)

    def test_rejects_non_finite_values(self) -> None:
        payload = fake_model_parameters(num_gaussians=4)
        payload["positions"][0, 0] = float("nan")
        with pytest.raises(ExportError, match="non-finite"):
            build_vertex_array(payload)

    def test_rejects_mismatched_row_counts(self) -> None:
        payload = fake_model_parameters(num_gaussians=4)
        payload["rotation"] = torch.zeros((3, 4))
        with pytest.raises(ExportError, match="expected"):
            build_vertex_array(payload)

    def test_rejects_specular_width_not_divisible_by_three(self) -> None:
        payload = fake_model_parameters(num_gaussians=4)
        payload["features_specular"] = torch.zeros((4, 8))
        with pytest.raises(ExportError, match="not divisible by 3"):
            build_vertex_array(payload)


class TestEndToEndExport:
    def test_writes_a_readable_ply(self, tmp_path: Path) -> None:
        plyfile = pytest.importorskip("plyfile")
        checkpoint = write_fake_checkpoint(tmp_path / "ckpt_100.pt", num_gaussians=37)
        output = tmp_path / "out" / "splat.ply"

        stats = export_checkpoint(checkpoint, output)
        assert stats.num_gaussians == 37
        assert stats.sh_degree == 3
        assert stats.global_step == 100
        assert output.stat().st_size == stats.output_bytes

        data = plyfile.PlyData.read(str(output))
        assert data.elements[0].name == "vertex"
        assert len(data.elements[0].data) == 37
        assert data.text is False  # binary PLY, not ASCII

    def test_leaves_no_partial_file_behind(self, tmp_path: Path) -> None:
        checkpoint = write_fake_checkpoint(tmp_path / "ckpt_100.pt", num_gaussians=5)
        output = tmp_path / "splat.ply"
        export_checkpoint(checkpoint, output)
        assert list(tmp_path.glob("*.partial")) == []

    def test_cli_prints_parseable_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        checkpoint = write_fake_checkpoint(tmp_path / "ckpt_100.pt", num_gaussians=11)
        stats_path = tmp_path / "stats.json"
        code = main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(tmp_path / "splat.ply"),
                "--stats-json",
                str(stats_path),
            ]
        )
        assert code == 0
        printed = dict(
            line.split("=", 1) for line in capsys.readouterr().out.splitlines() if "=" in line
        )
        assert printed["num_gaussians"] == "11"
        assert printed["sh_degree"] == "3"
        assert stats_path.is_file()

    def test_cli_returns_nonzero_on_failure(self, tmp_path: Path) -> None:
        assert main(["--checkpoint", str(tmp_path / "absent.pt"), "--output", str(tmp_path / "x.ply")]) == 1


class TestMatchesUpstreamExporter:
    """Byte-for-byte agreement with threedgrut's own PLY writer."""

    def test_identical_output(self, tmp_path: Path) -> None:
        ply_exporter = pytest.importorskip(
            "threedgrut.export.ply_exporter",
            reason="threedgrut is not installed; run inside the service container",
        )
        payload = fake_model_parameters(num_gaussians=23, sh_degree=3)

        class Adapter:
            """Minimal ExportableModel over raw checkpoint tensors."""

            def get_positions(self):
                return payload["positions"]

            def get_max_n_features(self):
                return payload["max_n_features"]

            def get_n_active_features(self):
                return payload["n_active_features"]

            def get_features_albedo(self):
                return payload["features_albedo"]

            def get_features_specular(self):
                return payload["features_specular"]

            def get_density(self, preactivation=False):
                assert preactivation, "upstream exports pre-activation density"
                return payload["density"]

            def get_scale(self, preactivation=False):
                assert preactivation
                return payload["scale"]

            def get_rotation(self, preactivation=False):
                assert preactivation
                return payload["rotation"]

        theirs = tmp_path / "upstream.ply"
        ply_exporter.PLYExporter().export(Adapter(), theirs)

        checkpoint = tmp_path / "ckpt_100.pt"
        torch.save(payload, checkpoint)
        ours = tmp_path / "ours.ply"
        export_checkpoint(checkpoint, ours)

        assert ours.read_bytes() == theirs.read_bytes()

    def test_property_names_match_upstream(self) -> None:
        ply_exporter = pytest.importorskip(
            "threedgrut.export.ply_exporter", reason="threedgrut is not installed"
        )
        albedo = np.zeros((2, 3), dtype=np.float32)
        specular = np.zeros((2, 45), dtype=np.float32)
        scale = np.zeros((2, 3), dtype=np.float32)
        rotation = np.zeros((2, 4), dtype=np.float32)
        assert attribute_names(3, 45) == ply_exporter.PLYExporter._construct_list_of_attributes(
            albedo, specular, scale, rotation
        )
