# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for path containment and identifier validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from splat_api.app.errors import BadRequest
from splat_api.app.paths import (
    directory_size,
    is_within,
    new_id,
    safe_join,
    sanitize_archive_member,
    validate_id,
    validate_image_name,
)


class TestSafeJoin:
    def test_joins_relative_components(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "jobs", "job_abc") == (tmp_path / "jobs" / "job_abc").resolve()

    def test_allows_paths_that_do_not_exist_yet(self, tmp_path: Path) -> None:
        target = safe_join(tmp_path, "not", "created", "yet.ply")
        assert not target.exists()
        assert is_within(tmp_path, target.parent.parent)

    @pytest.mark.parametrize("component", ["..", ".", "", "/etc", "a\x00b"])
    def test_rejects_hostile_components(self, tmp_path: Path, component: str) -> None:
        with pytest.raises(BadRequest):
            safe_join(tmp_path, component)

    def test_rejects_escape_through_symlinked_directory(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(BadRequest):
            safe_join(root, "link", "secret.txt")

    def test_root_itself_is_permitted(self, tmp_path: Path) -> None:
        nested = tmp_path / "root"
        nested.mkdir()
        assert safe_join(nested, "x").parent == nested.resolve()


class TestSanitizeArchiveMember:
    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "C:/windows/system32",
            "images/../../escape.jpg",
            "images\\..\\..\\escape.jpg",
            "a\x00b",
        ],
    )
    def test_rejects_traversal(self, name: str) -> None:
        with pytest.raises(BadRequest):
            sanitize_archive_member(name)

    def test_normalizes_separators_and_dot_segments(self) -> None:
        assert sanitize_archive_member("scene\\images\\./a.jpg") == "scene/images/a.jpg"

    def test_rejects_directory_entries(self) -> None:
        with pytest.raises(BadRequest):
            sanitize_archive_member("images/")


class TestImageNames:
    @pytest.mark.parametrize("name", ["frame_0001.jpg", "IMG-2.PNG", "a.tiff"])
    def test_accepts_plain_basenames(self, name: str) -> None:
        assert validate_image_name(name) == name

    @pytest.mark.parametrize(
        "name",
        ["../a.jpg", "sub/dir.jpg", "a.exe", ".hidden.jpg", "a.jpg;rm -rf /", "a b.jpg", "a.jpg\n"],
    )
    def test_rejects_anything_else(self, name: str) -> None:
        with pytest.raises(BadRequest):
            validate_image_name(name)


class TestIdentifiers:
    def test_generated_ids_validate(self) -> None:
        for prefix in ("scene", "job"):
            generated = new_id(prefix)
            assert validate_id(generated, kind=prefix) == generated

    @pytest.mark.parametrize(
        "value",
        ["", "Scene_1", "a" * 64, "scene.1", "scene=1", "scene,1", "../x", "-leading"],
    )
    def test_rejects_unsafe_ids(self, value: str) -> None:
        # '=' and ',' matter specifically: identifiers are interpolated into Hydra
        # overrides such as experiment_name=<scene_id>.
        with pytest.raises(BadRequest):
            validate_id(value, kind="scene_id")


def test_directory_size_ignores_symlinks(tmp_path: Path) -> None:
    (tmp_path / "real.bin").write_bytes(b"x" * 100)
    external = tmp_path.parent / "external.bin"
    external.write_bytes(b"y" * 5000)
    (tmp_path / "link.bin").symlink_to(external)
    assert directory_size(tmp_path) == 100
