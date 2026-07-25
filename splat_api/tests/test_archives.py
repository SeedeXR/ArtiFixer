# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ZIP ingestion, including adversarial archives."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from splat_api.app.archives import extract_colmap_archive, inspect_archive
from splat_api.app.config import Settings
from splat_api.app.errors import BadRequest, PayloadTooLarge, UnprocessableInput
from splat_api.tests.helpers import (
    build_colmap_scene,
    build_colmap_zip,
    make_zip,
    set_encrypted_flag,
    zip_directory,
)


def _extract(archive: Path, tmp_path: Path, settings: Settings) -> tuple[Path, object]:
    destination = tmp_path / "extracted"
    destination.mkdir()
    report = extract_colmap_archive(archive, destination, settings)
    return destination, report


class TestHappyPath:
    def test_extracts_images_and_sparse_model(self, tmp_path: Path, settings: Settings) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=5)
        destination, report = _extract(archive, tmp_path, settings)

        assert report.image_count == 5
        assert sorted(path.name for path in (destination / "images").iterdir()) == sorted(names)
        assert (destination / "sparse" / "0" / "cameras.bin").is_file()
        assert (destination / "sparse" / "0" / "points3D.bin").is_file()
        assert report.written_bytes > 0

    def test_strips_the_scene_prefix(self, tmp_path: Path, settings: Settings) -> None:
        archive, _ = build_colmap_zip(tmp_path, prefix="captures/truck")
        destination, report = _extract(archive, tmp_path, settings)
        assert report.scene_prefix == "captures/truck"
        assert (destination / "images").is_dir()

    def test_flat_archive_without_prefix(self, tmp_path: Path, settings: Settings) -> None:
        archive, _ = build_colmap_zip(tmp_path, prefix="")
        _, report = _extract(archive, tmp_path, settings)
        assert report.scene_prefix == "<root>"

    def test_falls_back_to_downsampled_image_directory(self, tmp_path: Path, settings: Settings) -> None:
        scene = tmp_path / "scene"
        build_colmap_scene(scene, image_dir_name="images_4")
        archive = zip_directory(scene, tmp_path / "scene.zip")
        _, report = _extract(archive, tmp_path, settings)
        assert report.image_dir_name == "images_4"

    def test_ignores_unrelated_members(self, tmp_path: Path, settings: Settings) -> None:
        scene = tmp_path / "scene"
        build_colmap_scene(scene)
        (scene / "sparse" / "0" / "project.ini").write_text("[General]\n")
        (scene / "notes.txt").write_text("hello")
        (scene / "run.sh").write_text("#!/bin/sh\nrm -rf /\n")
        archive = zip_directory(scene, tmp_path / "scene.zip")
        destination, report = _extract(archive, tmp_path, settings)

        assert report.skipped_members == 3
        assert not (destination / "notes.txt").exists()
        assert not (destination / "run.sh").exists()
        assert not (destination / "sparse" / "0" / "project.ini").exists()

    def test_nested_image_paths_are_flattened(self, tmp_path: Path, settings: Settings) -> None:
        scene = tmp_path / "scene"
        names = build_colmap_scene(scene)
        nested = scene / "images" / "sub"
        nested.mkdir()
        (scene / "images" / names[0]).rename(nested / names[0])
        archive = zip_directory(scene, tmp_path / "scene.zip")
        destination, report = _extract(archive, tmp_path, settings)
        assert report.image_count == len(names)
        assert (destination / "images" / names[0]).is_file()

    def test_parallel_extraction_matches_serial(self, tmp_path: Path, settings: Settings) -> None:
        # 20 images exercises the thread-pool branch; content must be identical.
        archive, names = build_colmap_zip(tmp_path, image_count=20)
        destination, report = _extract(archive, tmp_path, settings)
        assert report.image_count == 20
        with zipfile.ZipFile(archive) as bundle:
            for name in names:
                expected = bundle.read(f"scene/images/{name}")
                assert (destination / "images" / name).read_bytes() == expected


class TestRejections:
    def test_rejects_non_zip(self, tmp_path: Path, settings: Settings) -> None:
        bogus = tmp_path / "not.zip"
        bogus.write_bytes(b"this is not a zip file")
        with pytest.raises(BadRequest, match="not a valid ZIP"):
            _extract(bogus, tmp_path, settings)

    def test_rejects_zip_slip(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(tmp_path / "evil.zip", {"../../evil.jpg": b"x", "sparse/0/cameras.bin": b"x"})
        with pytest.raises(BadRequest, match="escapes"):
            _extract(archive, tmp_path, settings)

    def test_rejects_absolute_member(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(tmp_path / "abs.zip", {"/etc/cron.d/x": b"x", "sparse/0/cameras.bin": b"x"})
        with pytest.raises(BadRequest, match="relative path"):
            _extract(archive, tmp_path, settings)

    def test_rejects_symlink_member(self, tmp_path: Path, settings: Settings) -> None:
        archive_path = tmp_path / "link.zip"
        with zipfile.ZipFile(archive_path, "w") as bundle:
            info = zipfile.ZipInfo("images/link.jpg")
            info.create_system = 3
            info.external_attr = (0o120777 & 0xFFFF) << 16
            bundle.writestr(info, "/etc/passwd")
            bundle.writestr("sparse/0/cameras.bin", b"x")
        with pytest.raises(BadRequest, match="symlink"):
            _extract(archive_path, tmp_path, settings)

    def test_rejects_too_many_members(self, tmp_path: Path, settings: Settings) -> None:
        tiny = replace(settings, max_archive_members=4)
        archive, _ = build_colmap_zip(tmp_path, image_count=6)
        with pytest.raises(PayloadTooLarge, match="members"):
            _extract(archive, tmp_path, tiny)

    def test_rejects_oversized_expansion(self, tmp_path: Path, settings: Settings) -> None:
        tiny = replace(settings, max_uncompressed_bytes=2048)
        archive = make_zip(
            tmp_path / "big.zip",
            {"images/a.jpg": b"a" * 4096, "sparse/0/cameras.bin": b"c"},
        )
        with pytest.raises(PayloadTooLarge, match="expands to"):
            _extract(archive, tmp_path, tiny)

    def test_rejects_compression_bomb_ratio(self, tmp_path: Path, settings: Settings) -> None:
        # 32 MB of zeros deflates to ~32 KB: past the 4096-byte floor where the
        # ratio guard engages, and far past the configured ratio.
        bomb = replace(settings, max_compression_ratio=10.0, max_uncompressed_bytes=1 << 30)
        archive = make_zip(tmp_path / "bomb.zip", {"images/a.jpg": b"\0" * (32 << 20)})
        with pytest.raises(PayloadTooLarge, match="compression ratio"):
            _extract(archive, tmp_path, bomb)

    def test_small_archives_bypass_the_ratio_guard(self, tmp_path: Path, settings: Settings) -> None:
        """A legitimate tiny scene must not trip the bomb heuristic."""
        strict = replace(settings, max_compression_ratio=2.0)
        archive, _ = build_colmap_zip(tmp_path, image_count=2)
        _, report = _extract(archive, tmp_path, strict)
        assert report.image_count == 2

    def test_rejects_missing_sparse_model(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(tmp_path / "no_sparse.zip", {"images/a.jpg": b"x", "images/b.jpg": b"y"})
        with pytest.raises(UnprocessableInput, match="sparse/0/cameras.bin"):
            _extract(archive, tmp_path, settings)

    def test_rejects_partial_sparse_model(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(
            tmp_path / "partial.zip",
            {"images/a.jpg": b"x", "images/b.jpg": b"y", "sparse/0/cameras.bin": b"c"},
        )
        with pytest.raises(UnprocessableInput, match="missing required sparse model files"):
            _extract(archive, tmp_path, settings)

    def test_rejects_missing_image_directory(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(
            tmp_path / "no_images.zip",
            {
                "sparse/0/cameras.bin": b"c",
                "sparse/0/images.bin": b"i",
                "sparse/0/points3D.bin": b"p",
            },
        )
        with pytest.raises(UnprocessableInput, match="no image directory"):
            _extract(archive, tmp_path, settings)

    def test_rejects_two_colmap_models(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(
            tmp_path / "two.zip",
            {"a/sparse/0/cameras.bin": b"c", "b/sparse/0/cameras.bin": b"c"},
        )
        with pytest.raises(UnprocessableInput, match="more than one COLMAP model"):
            _extract(archive, tmp_path, settings)

    def test_rejects_duplicate_image_basenames(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(
            tmp_path / "dupe.zip",
            {
                "images/a.jpg": b"x",
                "images/nested/a.jpg": b"y",
                "sparse/0/cameras.bin": b"c",
                "sparse/0/images.bin": b"i",
                "sparse/0/points3D.bin": b"p",
            },
        )
        with pytest.raises(UnprocessableInput, match="duplicate image basename"):
            _extract(archive, tmp_path, settings)

    def test_rejects_single_image_scene(self, tmp_path: Path, settings: Settings) -> None:
        archive, _ = build_colmap_zip(tmp_path, image_count=1)
        with pytest.raises(UnprocessableInput, match="at least 2"):
            _extract(archive, tmp_path, settings)

    def test_rejects_image_count_over_limit(self, tmp_path: Path, settings: Settings) -> None:
        capped = replace(settings, max_images=3)
        archive, _ = build_colmap_zip(tmp_path, image_count=5)
        with pytest.raises(PayloadTooLarge, match="more than 3 images"):
            _extract(archive, tmp_path, capped)

    def test_rejects_encrypted_archive(self, tmp_path: Path, settings: Settings) -> None:
        archive = make_zip(tmp_path / "enc.zip", {"images/a.jpg": b"x" * 64})
        set_encrypted_flag(archive)
        with pytest.raises(BadRequest, match="Encrypted"):
            inspect_archive(archive, settings)

    def test_nothing_is_written_when_planning_fails(self, tmp_path: Path, settings: Settings) -> None:
        """A rejected archive must leave no files behind."""
        archive = make_zip(
            tmp_path / "partial.zip",
            {"images/a.jpg": b"x", "images/b.jpg": b"y", "sparse/0/cameras.bin": b"c"},
        )
        destination = tmp_path / "extracted"
        destination.mkdir()
        with pytest.raises(UnprocessableInput):
            extract_colmap_archive(archive, destination, settings)
        assert list(destination.rglob("*")) == []
