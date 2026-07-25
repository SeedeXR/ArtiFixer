# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures: synthetic COLMAP models and archives.

The binary writers here mirror the formats read by
``thirdparty/3DGRUT-ArtiFixer/threedgrut/datasets/utils.py``. ``points3D.bin`` is
written in full (the repo's own writer symlinks it instead) so tests can build a
complete, self-contained scene without any real capture data.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# model_id, num_params per CAMERA_MODELS in threedgrut/datasets/utils.py:223-235.
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": (0, 3),
    "PINHOLE": (1, 4),
    "SIMPLE_RADIAL": (2, 4),
    "RADIAL": (3, 5),
    "OPENCV": (4, 8),
    "OPENCV_FISHEYE": (5, 8),
}


@dataclass
class FakeCamera:
    camera_id: int = 1
    model: str = "PINHOLE"
    width: int = 64
    height: int = 48
    params: tuple[float, ...] = (60.0, 60.0, 32.0, 24.0)


@dataclass
class FakeImage:
    image_id: int
    name: str
    camera_id: int = 1
    qvec: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    observations: list[tuple[float, float, int]] = field(default_factory=list)


def write_cameras_bin(path: Path, cameras: list[FakeCamera]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            model_id, num_params = CAMERA_MODEL_IDS[camera.model]
            assert len(camera.params) == num_params, f"{camera.model} needs {num_params} params"
            handle.write(struct.pack("<iiQQ", camera.camera_id, model_id, camera.width, camera.height))
            handle.write(struct.pack("<" + "d" * num_params, *camera.params))


def write_images_bin(path: Path, images: list[FakeImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            handle.write(
                struct.pack("<idddddddi", image.image_id, *image.qvec, *image.tvec, image.camera_id)
            )
            handle.write(image.name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(image.observations)))
            for x, y, point_id in image.observations:
                handle.write(struct.pack("<ddq", x, y, point_id))


def write_points3d_bin(path: Path, count: int) -> None:
    """Write ``count`` points, each with a single-observation track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", count))
        for index in range(count):
            handle.write(
                struct.pack(
                    "<QdddBBBd",
                    index + 1,
                    float(index) * 0.01,
                    float(index) * 0.02,
                    float(index) * 0.03,
                    128,
                    128,
                    128,
                    0.5,
                )
            )
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<ii", 1, index))


def write_image_file(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, color=(120, 130, 140)).save(path)


def build_colmap_scene(
    root: Path,
    *,
    image_count: int = 4,
    camera: FakeCamera | None = None,
    point_count: int = 32,
    image_size: tuple[int, int] | None = None,
    image_dir_name: str = "images",
    extension: str = ".jpg",
) -> list[str]:
    """Create a minimal but fully valid COLMAP scene. Returns image basenames."""
    camera = camera or FakeCamera()
    image_size = image_size or (camera.width, camera.height)
    names = [f"frame_{index:04d}{extension}" for index in range(image_count)]
    images = [
        FakeImage(
            image_id=index + 1,
            name=name,
            camera_id=camera.camera_id,
            tvec=(float(index) * 0.1, 0.0, 1.0),
            observations=[(1.0 + index, 2.0 + index, index + 1)],
        )
        for index, name in enumerate(names)
    ]
    for name in names:
        write_image_file(root / image_dir_name / name, image_size)
    write_cameras_bin(root / "sparse" / "0" / "cameras.bin", [camera])
    write_images_bin(root / "sparse" / "0" / "images.bin", images)
    write_points3d_bin(root / "sparse" / "0" / "points3D.bin", point_count)
    return names


def zip_directory(source: Path, archive_path: Path, *, prefix: str = "") -> Path:
    """Zip ``source`` recursively, optionally under a top-level ``prefix``."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                bundle.write(path, arcname=f"{prefix}/{relative}" if prefix else relative)
    return archive_path


def build_colmap_zip(tmp_path: Path, *, prefix: str = "scene", **scene_kwargs) -> tuple[Path, list[str]]:
    """Build a scene directory and return ``(archive_path, image_names)``."""
    scene_dir = tmp_path / "raw_scene"
    names = build_colmap_scene(scene_dir, **scene_kwargs)
    archive = zip_directory(scene_dir, tmp_path / "scene.zip", prefix=prefix)
    return archive, names


def set_encrypted_flag(archive_path: Path) -> Path:
    """Flip the "encrypted" general-purpose bit in an existing archive.

    ``ZipFile.writestr`` recomputes ``flag_bits``, so an encrypted archive cannot
    be produced through the public API. Patching the raw headers is the only way
    to build this adversarial input. Offsets are from the ZIP specification:
    bit flag at +6 in a local file header, +8 in a central directory record.
    """
    data = bytearray(archive_path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        position = 0
        while True:
            position = data.find(signature, position)
            if position < 0:
                break
            index = position + flag_offset
            data[index] |= 0x01
            position += 4
    archive_path.write_bytes(bytes(data))
    return archive_path


def make_zip(archive_path: Path, members: dict[str, bytes]) -> Path:
    """Write an archive with exact member names, for adversarial-input tests."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return archive_path
