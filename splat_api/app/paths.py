# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path containment and identifier helpers.

Every filesystem path the service builds from caller-supplied data goes through
:func:`safe_join`. The pipeline writes symlink-heavy trees (the ArtiFixer prep
step symlinks source images and ``points3D.bin``), so containment checks must
distinguish "resolves outside the root" from "is a symlink", and only the former
is fatal for reads.
"""

from __future__ import annotations

import os
import re
import secrets
import unicodedata
from pathlib import Path

from splat_api.app.errors import BadRequest

# Identifiers appear in filesystem paths and in 3DGRUT/Hydra overrides, so the
# alphabet is deliberately narrower than the filesystem allows: no dots (no
# traversal, no hidden files), no '=' or ',' (Hydra override separators), and no
# whitespace.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
IMAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Extensions PIL can open and that COLMAP realistically emits.
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
ALLOWED_SPARSE_NAMES = frozenset({"cameras.bin", "images.bin", "points3D.bin"})


def new_id(prefix: str) -> str:
    """Return a collision-resistant, URL-safe, lowercase identifier."""
    return f"{prefix}_{secrets.token_hex(12)}"


def validate_id(value: str, *, kind: str) -> str:
    if not ID_PATTERN.match(value):
        raise BadRequest(f"Invalid {kind}: must match {ID_PATTERN.pattern}")
    return value


def safe_join(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and refuse anything that escapes it.

    ``root`` is resolved; the candidate is resolved with ``strict=False`` so the
    check also covers paths that do not exist yet (job artifact targets). This
    catches ``..`` segments, absolute components, and symlinked directories that
    point outside the root.
    """
    resolved_root = root.resolve()
    for part in parts:
        if part in ("", ".", ".."):
            raise BadRequest("Path components must not be empty or relative markers")
        if part.startswith("/") or "\x00" in part:
            raise BadRequest("Path components must be relative and NUL-free")
    candidate = resolved_root.joinpath(*parts).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise BadRequest("Resolved path escapes its permitted root")
    return candidate


def is_within(root: Path, candidate: Path) -> bool:
    """Non-raising containment test used for read-time validation."""
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
    except OSError:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def sanitize_archive_member(name: str) -> str:
    """Return a safe relative path for a zip member, or raise.

    Rejects absolute paths, drive letters, traversal segments, NUL bytes, and
    Unicode forms that normalize into a traversal. Backslashes are treated as
    separators because Windows-produced archives use them and a naive check
    would otherwise let ``..\\..\\x`` through as a single "filename".
    """
    if not name or name.endswith("/"):
        raise BadRequest("Archive contains an entry with an empty name")
    if "\x00" in name:
        raise BadRequest("Archive member name contains a NUL byte")

    normalized = unicodedata.normalize("NFC", name).replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise BadRequest(f"Archive member must be a relative path: {name!r}")

    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise BadRequest(f"Archive member escapes the extraction root: {name!r}")
    if not parts:
        raise BadRequest(f"Archive member has no usable path: {name!r}")
    return "/".join(parts)


def validate_image_name(name: str) -> str:
    """Validate a caller-supplied image basename.

    Used for ``selected_image_names``: these end up in a file consumed by
    ``prepare_colmap_artifixer_inputs``, which matches on basenames. Rejecting
    separators here means a caller cannot smuggle a path.
    """
    if "/" in name or "\\" in name:
        raise BadRequest(f"Image name must be a bare filename, got {name!r}")
    if not IMAGE_NAME_PATTERN.match(name):
        raise BadRequest(f"Image name {name!r} contains unsupported characters")
    if Path(name).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise BadRequest(f"Image name {name!r} does not have a supported image extension")
    return name


def directory_size(path: Path) -> int:
    """Total size of regular files under ``path``, ignoring symlinks.

    Symlinks are skipped so the reported size reflects bytes this job actually
    owns; the prep stage links source images rather than copying them.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not Path(dirpath, name).is_symlink()]
        for filename in filenames:
            file_path = Path(dirpath, filename)
            if file_path.is_symlink():
                continue
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total
