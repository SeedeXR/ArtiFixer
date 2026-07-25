# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for the COLMAP reader and precondition checks.

The regression class is the important one: it asserts our dependency-free reader
agrees with ``threedgrut.datasets.utils``. If upstream ever changes the binary
layout, that test fails instead of the service silently mis-parsing scenes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from splat_api.app.colmap_input import (
    read_cameras,
    read_images,
    read_point_count,
    validate_scene,
)
from splat_api.app.errors import UnprocessableInput
from splat_api.tests.helpers import (
    FakeCamera,
    FakeImage,
    build_colmap_scene,
    write_image_file,
    write_images_bin,
    write_points3d_bin,
)

MAX_IMAGES = 4000


class TestReader:
    def test_reads_a_synthetic_scene(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path, image_count=3, point_count=17)
        cameras = read_cameras(tmp_path / "sparse" / "0" / "cameras.bin")
        images = read_images(tmp_path / "sparse" / "0" / "images.bin", max_images=MAX_IMAGES)

        assert len(cameras) == 1
        assert cameras[0].model == "PINHOLE"
        assert cameras[0].params == (60.0, 60.0, 32.0, 24.0)
        assert [image.name for image in images] == [
            "frame_0000.jpg",
            "frame_0001.jpg",
            "frame_0002.jpg",
        ]
        assert read_point_count(tmp_path / "sparse" / "0" / "points3D.bin") == 17

    @pytest.mark.parametrize(
        ("model", "params"),
        [
            ("SIMPLE_PINHOLE", (50.0, 32.0, 24.0)),
            ("PINHOLE", (50.0, 51.0, 32.0, 24.0)),
            ("SIMPLE_RADIAL", (50.0, 32.0, 24.0, 0.01)),
            ("RADIAL", (50.0, 32.0, 24.0, 0.01, 0.02)),
            ("OPENCV", (50.0, 51.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0)),
        ],
    )
    def test_reads_every_supported_camera_model(
        self, tmp_path: Path, model: str, params: tuple[float, ...]
    ) -> None:
        camera = FakeCamera(model=model, params=params)
        build_colmap_scene(tmp_path, camera=camera)
        summary = validate_scene(tmp_path, max_images=MAX_IMAGES)
        assert summary.camera_models == (model,)

    def test_detects_truncation(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path)
        path = tmp_path / "sparse" / "0" / "cameras.bin"
        path.write_bytes(path.read_bytes()[:12])
        with pytest.raises(UnprocessableInput, match="truncated"):
            read_cameras(path)

    def test_rejects_unknown_camera_model_id(self, tmp_path: Path) -> None:
        path = tmp_path / "cameras.bin"
        path.write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 99, 64, 48))
        with pytest.raises(UnprocessableInput, match="unknown COLMAP camera model"):
            read_cameras(path)

    def test_rejects_unterminated_image_name(self, tmp_path: Path) -> None:
        path = tmp_path / "images.bin"
        path.write_bytes(
            struct.pack("<Q", 1) + struct.pack("<idddddddi", 1, 1, 0, 0, 0, 0, 0, 0, 1) + b"noterm"
        )
        with pytest.raises(UnprocessableInput, match="unterminated"):
            read_images(path, max_images=MAX_IMAGES)


class TestValidation:
    def test_accepts_a_valid_scene(self, tmp_path: Path) -> None:
        names = build_colmap_scene(tmp_path, image_count=6, point_count=42)
        summary = validate_scene(tmp_path, max_images=MAX_IMAGES)
        assert summary.image_count == 6
        assert summary.point_count == 42
        assert summary.width == 64
        assert summary.height == 48
        assert list(summary.image_names) == names

    def test_rejects_missing_images_directory(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path)
        for path in (tmp_path / "images").iterdir():
            path.unlink()
        (tmp_path / "images").rmdir()
        with pytest.raises(UnprocessableInput, match="missing the images/ directory"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_empty_points3d(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path)
        write_points3d_bin(tmp_path / "sparse" / "0" / "points3D.bin", 0)
        with pytest.raises(UnprocessableInput, match="points3D.bin is empty"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_image_referenced_but_absent(self, tmp_path: Path) -> None:
        names = build_colmap_scene(tmp_path)
        (tmp_path / "images" / names[1]).unlink()
        with pytest.raises(UnprocessableInput, match="absent from images/"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_unsupported_camera_model(self, tmp_path: Path) -> None:
        camera = FakeCamera(model="OPENCV_FISHEYE", params=(50.0, 51.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0))
        build_colmap_scene(tmp_path, camera=camera)
        with pytest.raises(UnprocessableInput, match="Unsupported COLMAP camera models"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_duplicate_basenames(self, tmp_path: Path) -> None:
        camera = FakeCamera()
        build_colmap_scene(tmp_path, image_count=2)
        images = [
            FakeImage(image_id=1, name="a/dup.jpg", camera_id=camera.camera_id),
            FakeImage(image_id=2, name="b/dup.jpg", camera_id=camera.camera_id),
        ]
        write_images_bin(tmp_path / "sparse" / "0" / "images.bin", images)
        with pytest.raises(UnprocessableInput, match="basenames must be unique"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_mixed_image_sizes_for_one_camera(self, tmp_path: Path) -> None:
        names = build_colmap_scene(tmp_path, image_count=3)
        write_image_file(tmp_path / "images" / names[2], (32, 24))
        with pytest.raises(UnprocessableInput, match="different sizes"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_accepts_uniformly_downscaled_images(self, tmp_path: Path) -> None:
        """Half-resolution images with full-resolution intrinsics are supported.

        prepare_colmap_artifixer_inputs.scale_colmap_scene_to_images rescales the
        camera when the ratio is uniform, so we must not reject this case.
        """
        camera = FakeCamera(width=128, height=96, params=(120.0, 120.0, 64.0, 48.0))
        build_colmap_scene(tmp_path, camera=camera, image_size=(64, 48))
        summary = validate_scene(tmp_path, max_images=MAX_IMAGES)
        assert (summary.width, summary.height) == (64, 48)

    def test_rejects_non_uniform_rescale(self, tmp_path: Path) -> None:
        camera = FakeCamera(width=128, height=96, params=(120.0, 120.0, 64.0, 48.0))
        build_colmap_scene(tmp_path, camera=camera, image_size=(64, 72))
        with pytest.raises(UnprocessableInput, match="uniform rescaling"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_two_different_calibrations(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path, image_count=2)
        from splat_api.tests.helpers import write_cameras_bin

        write_cameras_bin(
            tmp_path / "sparse" / "0" / "cameras.bin",
            [
                FakeCamera(camera_id=1, params=(60.0, 60.0, 32.0, 24.0)),
                FakeCamera(camera_id=2, params=(80.0, 80.0, 32.0, 24.0)),
            ],
        )
        write_images_bin(
            tmp_path / "sparse" / "0" / "images.bin",
            [
                FakeImage(image_id=1, name="frame_0000.jpg", camera_id=1),
                FakeImage(image_id=2, name="frame_0001.jpg", camera_id=2),
            ],
        )
        with pytest.raises(UnprocessableInput, match="one shared intrinsic calibration"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_camera_id_not_in_cameras_bin(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path, image_count=2)
        write_images_bin(
            tmp_path / "sparse" / "0" / "images.bin",
            [
                FakeImage(image_id=1, name="frame_0000.jpg", camera_id=1),
                FakeImage(image_id=2, name="frame_0001.jpg", camera_id=7),
            ],
        )
        with pytest.raises(UnprocessableInput, match="camera ids absent"):
            validate_scene(tmp_path, max_images=MAX_IMAGES)

    def test_rejects_image_count_over_limit(self, tmp_path: Path) -> None:
        build_colmap_scene(tmp_path, image_count=4)
        with pytest.raises(UnprocessableInput, match="the limit is 2"):
            validate_scene(tmp_path, max_images=2)


class TestReaderAgreesWithThreedgrut:
    """Regression guard against upstream COLMAP-format drift."""

    @staticmethod
    def _upstream():
        return pytest.importorskip(
            "threedgrut.datasets.utils",
            reason="threedgrut is not installed; run inside the service container",
        )

    def test_cameras_match_upstream(self, tmp_path: Path) -> None:
        upstream = self._upstream()
        camera = FakeCamera(model="OPENCV", params=(50.0, 51.0, 32.0, 24.0, 0.1, 0.2, 0.3, 0.4))
        build_colmap_scene(tmp_path, camera=camera)
        path = tmp_path / "sparse" / "0" / "cameras.bin"

        theirs = upstream.read_colmap_intrinsics_binary(path)
        ours = {entry.camera_id: entry for entry in read_cameras(path)}
        assert set(theirs) == set(ours)
        for camera_id, reference in theirs.items():
            mine = ours[camera_id]
            assert mine.model == reference.model
            assert (mine.width, mine.height) == (int(reference.width), int(reference.height))
            assert list(mine.params) == [float(value) for value in reference.params]

    def test_images_match_upstream(self, tmp_path: Path) -> None:
        upstream = self._upstream()
        build_colmap_scene(tmp_path, image_count=5)
        path = tmp_path / "sparse" / "0" / "images.bin"

        theirs = upstream.read_colmap_extrinsics_binary(path)
        ours = read_images(path, max_images=MAX_IMAGES)
        assert [entry.name for entry in ours] == [entry.name for entry in theirs]
        assert [entry.camera_id for entry in ours] == [entry.camera_id for entry in theirs]
        assert [entry.image_id for entry in ours] == [entry.id for entry in theirs]

    def test_point_count_matches_upstream(self, tmp_path: Path) -> None:
        upstream = self._upstream()
        build_colmap_scene(tmp_path, point_count=23)
        path = tmp_path / "sparse" / "0" / "points3D.bin"
        xyzs, _, _ = upstream.read_colmap_points3D_binary(path)
        assert read_point_count(path) == len(xyzs) == 23

    def test_supported_models_match_prepare_module(self) -> None:
        prepare = pytest.importorskip(
            "data_processing.prepare_colmap_artifixer_inputs",
            reason="repo modules unavailable",
        )
        from splat_api.app.colmap_input import SUPPORTED_CAMERA_MODELS

        assert set(SUPPORTED_CAMERA_MODELS) == set(prepare.SUPPORTED_CAMERA_MODELS)


class TestRealScene:
    """Runs only when a real COLMAP capture is provided via the environment."""

    def test_validates_a_real_capture(self) -> None:
        import os

        scene_root = os.environ.get("SPLAT_API_TEST_COLMAP_SCENE")
        if not scene_root:
            pytest.skip("set SPLAT_API_TEST_COLMAP_SCENE to a real COLMAP scene directory")
        summary = validate_scene(Path(scene_root), max_images=100_000)
        assert summary.image_count > 1
        assert summary.point_count > 0
        assert summary.width > 0
