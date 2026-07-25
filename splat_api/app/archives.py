# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardened extraction of caller-supplied COLMAP archives.

Only ZIP is accepted. TAR is deliberately unsupported because tar members can
carry device nodes, hard links and setuid bits, none of which ``tarfile`` filters
by default on every supported Python.

The extractor is an allowlist, not a denylist: it writes *only* the files the
pipeline needs (``images/<name>`` and ``sparse/0/{cameras,images,points3D}.bin``)
and silently skips everything else. That removes arbitrary-file-write as a class
of bug rather than trying to enumerate dangerous names.
"""

from __future__ import annotations

import contextlib
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from splat_api.app.config import Settings
from splat_api.app.errors import BadRequest, PayloadTooLarge, UnprocessableInput
from splat_api.app.paths import (
    ALLOWED_IMAGE_SUFFIXES,
    ALLOWED_SPARSE_NAMES,
    sanitize_archive_member,
)

# Image directories COLMAP/3DGS tooling commonly emits. Anything else is ignored.
IMAGE_DIR_NAMES = ("images", "images_2", "images_4", "images_8")

SPARSE_MARKER = "sparse/0/cameras.bin"
# Stored and deflated only. The others (bzip2, LZMA, PPMd) either need optional
# modules — raising NotImplementedError from deep inside the read, which is a 500,
# not a 400 — or have far worse worst-case expansion behaviour.
ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_UNIX_SYMLINK_MODE = 0o120000
_UNIX_TYPE_MASK = 0o170000
_COPY_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ExtractionReport:
    """What was written, and what was ignored."""

    scene_prefix: str
    image_dir_name: str
    image_count: int
    written_bytes: int
    skipped_members: int

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_scene_prefix": self.scene_prefix,
            "archive_image_dir": self.image_dir_name,
            "image_count": self.image_count,
            "extracted_bytes": self.written_bytes,
            "ignored_members": self.skipped_members,
        }


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    if info.create_system != 3:  # 3 == Unix; other producers have no mode bits.
        return False
    mode = info.external_attr >> 16
    return bool(mode) and (mode & _UNIX_TYPE_MASK) == _UNIX_SYMLINK_MODE


def _find_scene_prefix(names: list[str]) -> str:
    """Locate the directory that holds ``sparse/0/cameras.bin``.

    Archives are commonly rooted at a scene folder (``truck/sparse/0/...``), so
    we anchor on the marker file rather than assuming a flat layout. Multiple
    candidates are rejected: guessing which scene the caller meant would be
    worse than an explicit error.
    """
    candidates = sorted(
        name[: -len(SPARSE_MARKER)].rstrip("/") for name in names if name.endswith(SPARSE_MARKER)
    )
    if not candidates:
        raise UnprocessableInput(
            "Archive does not contain sparse/0/cameras.bin. Expected layout: "
            "images/ and sparse/0/{cameras,images,points3D}.bin"
        )
    if len(candidates) > 1:
        raise UnprocessableInput(
            "Archive contains more than one COLMAP model: "
            f"{', '.join(candidate or '<root>' for candidate in candidates[:5])}. "
            "Upload one scene per request."
        )
    return candidates[0]


def _relative_to_prefix(name: str, prefix: str) -> str | None:
    if not prefix:
        return name
    marker = prefix + "/"
    return name[len(marker) :] if name.startswith(marker) else None


def _pick_image_dir(relative_names: list[str]) -> str:
    present = {
        name.split("/", 1)[0]
        for name in relative_names
        if "/" in name and name.split("/", 1)[0] in IMAGE_DIR_NAMES
    }
    for candidate in IMAGE_DIR_NAMES:
        if candidate in present:
            return candidate
    raise UnprocessableInput(
        f"Archive has no image directory. Expected one of: {', '.join(IMAGE_DIR_NAMES)}"
    )


def inspect_archive(zip_path: Path, settings: Settings) -> zipfile.ZipFile:
    """Open ``zip_path`` and reject structurally hostile archives up front.

    Runs before a single byte is written, using only the central directory, so a
    zip bomb is rejected without doing the work of decompressing it.
    """
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise BadRequest(f"Uploaded file is not a valid ZIP archive: {exc}") from exc

    # Any rejection from here on must close the handle, including the ones raised
    # by sanitize_archive_member.
    try:
        infos = archive.infolist()
        if len(infos) > settings.max_archive_members:
            raise PayloadTooLarge(
                f"Archive has {len(infos)} members; the limit is {settings.max_archive_members}"
            )

        total_uncompressed = 0
        total_compressed = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise BadRequest("Encrypted ZIP archives are not supported")
            if _member_is_symlink(info):
                raise BadRequest(f"Archive contains a symlink member: {info.filename!r}")
            if info.is_dir():
                continue
            if info.compress_type not in ALLOWED_COMPRESSION:
                raise BadRequest(
                    f"Archive member {info.filename!r} uses unsupported compression method "
                    f"{info.compress_type}; use stored or deflate"
                )
            sanitize_archive_member(info.filename)
            total_uncompressed += info.file_size
            total_compressed += info.compress_size

        if total_uncompressed > settings.max_uncompressed_bytes:
            raise PayloadTooLarge(
                f"Archive expands to {total_uncompressed} bytes; the limit is "
                f"{settings.max_uncompressed_bytes}"
            )
        # Guard the classic bomb: enormous expansion from a tiny payload. Only
        # meaningful once the payload is big enough that the ratio is not noise.
        if (
            total_compressed > 4096
            and total_uncompressed / total_compressed > settings.max_compression_ratio
        ):
            raise PayloadTooLarge(
                f"Archive compression ratio {total_uncompressed / total_compressed:.1f}x exceeds the "
                f"{settings.max_compression_ratio:.1f}x limit"
            )
    except BaseException:
        archive.close()
        raise
    return archive


def _write_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, budget: int) -> int:
    """Stream one member to ``target``, stopping if it exceeds ``budget``.

    The declared ``file_size`` in the central directory is not trusted: we cap on
    bytes actually produced by the decompressor.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with archive.open(info, "r") as source, target.open("wb") as sink:
        while True:
            chunk = source.read(_COPY_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > budget:
                sink.close()
                target.unlink(missing_ok=True)
                raise PayloadTooLarge(
                    f"Archive member {info.filename!r} is larger than the remaining extraction budget"
                )
            sink.write(chunk)
    target.chmod(0o640)
    return written


@dataclass(frozen=True)
class _PlannedMember:
    member_name: str
    target: Path
    declared_size: int


def _plan_extraction(
    archive: zipfile.ZipFile, dest_dir: Path, settings: Settings
) -> tuple[list[_PlannedMember], ExtractionReport]:
    """Decide what to write before writing anything.

    Planning from the central directory alone lets every structural rejection
    (missing sparse files, duplicate basenames, too many images) happen before the
    first byte hits the disk, and it produces an independent work list that the
    write phase can execute in parallel.
    """
    file_infos = [info for info in archive.infolist() if not info.is_dir()]
    names = [sanitize_archive_member(info.filename) for info in file_infos]
    prefix = _find_scene_prefix(names)
    relative_names = [rel for rel in (_relative_to_prefix(name, prefix) for name in names) if rel]
    image_dir_name = _pick_image_dir(relative_names)

    planned: list[_PlannedMember] = []
    claimed_images: set[str] = set()
    sparse_planned: set[str] = set()
    skipped = 0

    for info, name in zip(file_infos, names):
        relative = _relative_to_prefix(name, prefix)
        if relative is None:
            skipped += 1
            continue
        head, _, tail = relative.partition("/")
        if head == image_dir_name and tail:
            # Flatten nested image folders: COLMAP matches on basenames, and
            # prepare_colmap_artifixer_inputs requires unique basenames (:174-176).
            basename = Path(tail).name
            if Path(basename).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                skipped += 1
                continue
            if basename in claimed_images:
                raise UnprocessableInput(
                    f"Archive contains duplicate image basename {basename!r}; "
                    "COLMAP image basenames must be unique"
                )
            if len(claimed_images) >= settings.max_images:
                raise PayloadTooLarge(f"Archive contains more than {settings.max_images} images")
            claimed_images.add(basename)
            planned.append(
                _PlannedMember(info.filename, dest_dir / "images" / basename, info.file_size)
            )
        elif relative.startswith("sparse/0/") and Path(relative).name in ALLOWED_SPARSE_NAMES:
            basename = Path(relative).name
            if basename in sparse_planned:
                raise UnprocessableInput(f"Archive contains two copies of sparse/0/{basename}")
            sparse_planned.add(basename)
            planned.append(
                _PlannedMember(info.filename, dest_dir / "sparse" / "0" / basename, info.file_size)
            )
        else:
            skipped += 1

    missing = sorted(ALLOWED_SPARSE_NAMES - sparse_planned)
    if missing:
        raise UnprocessableInput(f"Archive is missing required sparse model files: {missing}")
    if len(claimed_images) < 2:
        raise UnprocessableInput(
            f"Archive contains {len(claimed_images)} image(s); a reconstruction needs at least 2"
        )

    report = ExtractionReport(
        scene_prefix=prefix or "<root>",
        image_dir_name=image_dir_name,
        image_count=len(claimed_images),
        written_bytes=0,
        skipped_members=skipped,
    )
    return planned, report


def _extraction_workers(planned_count: int) -> int:
    # Inflating hundreds of JPEGs is zlib work that releases the GIL, so threads
    # scale. Cap low enough that extraction cannot monopolize a shared box.
    return max(1, min(8, (os.cpu_count() or 2), planned_count))


def extract_colmap_archive(zip_path: Path, dest_dir: Path, settings: Settings) -> ExtractionReport:
    """Extract the COLMAP subset of ``zip_path`` into ``dest_dir``.

    ``dest_dir`` must already exist and be owned by the caller (a fresh scene
    directory). On any failure the partially written tree is left for the caller
    to remove, which keeps this function free of destructive behaviour.

    Members are written by a small thread pool, each thread holding its own
    ``ZipFile`` handle because ``ZipFile`` is not safe for concurrent reads.
    """
    archive = inspect_archive(zip_path, settings)
    try:
        planned, report = _plan_extraction(archive, dest_dir, settings)
    finally:
        archive.close()

    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    budget = settings.max_uncompressed_bytes
    worker_count = _extraction_workers(len(planned))
    local = threading.local()
    written_total = 0
    lock = threading.Lock()

    handles: list[zipfile.ZipFile] = []

    def open_handle() -> zipfile.ZipFile:
        handle = getattr(local, "handle", None)
        if handle is None:
            handle = zipfile.ZipFile(zip_path)
            local.handle = handle
            # Register at creation, not after a successful write: a member that
            # raises mid-read would otherwise leave its handle untracked and closed
            # only whenever the garbage collector got round to it.
            with lock:
                handles.append(handle)
        return handle

    def write_one(item: _PlannedMember) -> int:
        nonlocal written_total
        handle = open_handle()
        info = handle.getinfo(item.member_name)
        with lock:
            # Each worker reserves against the shared total before writing, so
            # concurrent writers cannot jointly exceed the budget.
            remaining = budget - written_total
            written_total += info.file_size
        try:
            written = _write_member(handle, info, item.target, remaining)
        except BaseException:
            with lock:
                written_total -= info.file_size
            raise
        with lock:
            # Replace the reservation with what was actually produced.
            written_total += written - info.file_size
        return written

    try:
        if worker_count == 1:
            for item in planned:
                write_one(item)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="unzip") as pool:
                for future in as_completed([pool.submit(write_one, item) for item in planned]):
                    future.result()
    except zipfile.BadZipFile as exc:
        # Corrupt member data (bad CRC, truncated stream) is bad input.
        raise BadRequest(f"Archive data is corrupt: {exc}") from exc
    except NotImplementedError as exc:
        raise BadRequest(f"Archive uses an unsupported ZIP feature: {exc}") from exc
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()

    return ExtractionReport(
        scene_prefix=report.scene_prefix,
        image_dir_name=report.image_dir_name,
        image_count=report.image_count,
        written_bytes=written_total,
        skipped_members=report.skipped_members,
    )
