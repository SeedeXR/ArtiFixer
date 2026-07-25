# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free COLMAP sparse-model reader and precondition checks.

The API process must validate an uploaded scene in the request path, so it uses a
self-contained reader instead of importing ``threedgrut``/``torch`` (seconds of
startup and hundreds of MB of RSS in a process that never runs a model).

The binary layout mirrors ``threedgrut/datasets/utils.py``
(``read_colmap_intrinsics_binary``, ``read_colmap_extrinsics_binary``,
``read_colmap_points3D_binary``) and ``tests/test_colmap_input.py`` asserts this
reader agrees with those functions on a real scene.

The checks here intentionally duplicate the assertions in
``data_processing/prepare_colmap_artifixer_inputs.py`` so a bad scene is a clean
HTTP 422 at submit time rather than a subprocess traceback minutes into a job.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from splat_api.app.errors import UnprocessableInput
from splat_api.app.paths import ALLOWED_IMAGE_SUFFIXES, IMAGE_NAME_PATTERN

# model_id -> (name, num_params); mirrors CAMERA_MODELS in threedgrut/datasets/utils.py:223-235.
CAMERA_MODEL_IDS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

# prepare_colmap_artifixer_inputs.py:58 SUPPORTED_CAMERA_MODELS.
SUPPORTED_CAMERA_MODELS = frozenset({"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"})

# Model files are read whole into memory, so the cap has to be a size we are happy
# to allocate in the API process. A 4000-image images.bin with dense observations
# is a few hundred MB at most; anything larger is not a scene we can serve.
_MAX_MODEL_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImageEntry:
    image_id: int
    camera_id: int
    name: str


@dataclass(frozen=True)
class ColmapSummary:
    """Facts about a validated scene, safe to return to the caller."""

    image_count: int
    camera_count: int
    camera_models: tuple[str, ...]
    point_count: int
    width: int
    height: int
    image_names: tuple[str, ...]

    def as_dict(self, *, include_names: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "image_count": self.image_count,
            "camera_count": self.camera_count,
            "camera_models": list(self.camera_models),
            "point_count": self.point_count,
            "colmap_width": self.width,
            "colmap_height": self.height,
        }
        if include_names:
            payload["image_names"] = list(self.image_names)
        return payload


class _Reader:
    """Bounds-checked cursor over an in-memory model file."""

    def __init__(self, path: Path) -> None:
        size = path.stat().st_size
        if size > _MAX_MODEL_BYTES:
            raise UnprocessableInput(f"{path.name} is {size} bytes, which exceeds the supported limit")
        self._data = path.read_bytes()
        self._offset = 0
        self._name = path.name

    def unpack(self, fmt: str) -> tuple[object, ...]:
        fmt = "<" + fmt
        size = struct.calcsize(fmt)
        end = self._offset + size
        if end > len(self._data):
            raise UnprocessableInput(f"{self._name} is truncated")
        values = struct.unpack_from(fmt, self._data, self._offset)
        self._offset = end
        return values

    def skip(self, count: int) -> None:
        if count < 0 or self._offset + count > len(self._data):
            raise UnprocessableInput(f"{self._name} is truncated")
        self._offset += count

    def read_cstring(self, limit: int = 4096) -> str:
        end = self._data.find(b"\x00", self._offset)
        if end < 0 or end - self._offset > limit:
            raise UnprocessableInput(f"{self._name} contains an unterminated name")
        raw = self._data[self._offset : end]
        self._offset = end + 1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnprocessableInput(f"{self._name} contains a non-UTF-8 image name") from exc

    @property
    def remaining(self) -> int:
        return len(self._data) - self._offset


def read_cameras(path: Path) -> list[ColmapCamera]:
    reader = _Reader(path)
    (count,) = reader.unpack("Q")
    if not 0 < count <= 100_000:
        raise UnprocessableInput(f"cameras.bin declares {count} cameras, which is not plausible")
    cameras: list[ColmapCamera] = []
    for _ in range(int(count)):
        camera_id, model_id, width, height = reader.unpack("iiQQ")
        entry = CAMERA_MODEL_IDS.get(int(model_id))
        if entry is None:
            raise UnprocessableInput(f"cameras.bin uses unknown COLMAP camera model id {model_id}")
        model, num_params = entry
        params = reader.unpack("d" * num_params)
        cameras.append(
            ColmapCamera(
                camera_id=int(camera_id),
                model=model,
                width=int(width),
                height=int(height),
                params=tuple(float(value) for value in params),
            )
        )
    return cameras


def read_images(path: Path, *, max_images: int) -> list[ColmapImageEntry]:
    reader = _Reader(path)
    (count,) = reader.unpack("Q")
    if count == 0:
        raise UnprocessableInput("images.bin contains no registered images")
    if count > max_images:
        raise UnprocessableInput(f"images.bin declares {count} images; the limit is {max_images}")
    images: list[ColmapImageEntry] = []
    for _ in range(int(count)):
        values = reader.unpack("idddddddi")
        image_id = int(values[0])
        camera_id = int(values[8])
        name = reader.read_cstring()
        (num_points2d,) = reader.unpack("Q")
        # Each 2D observation is (x, y, point3D_id) == 8 + 8 + 8 bytes.
        reader.skip(int(num_points2d) * 24)
        images.append(ColmapImageEntry(image_id=image_id, camera_id=camera_id, name=name))
    return images


def read_point_count(path: Path) -> int:
    """Read only the point count.

    ``artifixer3d.copy_source_points3d`` rejects an empty ``points3D.bin``
    (data_processing/artifixer3d.py:449-451), and 3DGRUT initializes Gaussians
    from these points, so a zero count is fatal for the whole pipeline.
    """
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise UnprocessableInput("points3D.bin is truncated")
    return int(struct.unpack("<Q", header)[0])


def resolve_scene_paths(scene_dir: Path) -> tuple[Path, Path]:
    """Mirror ``prepare_colmap_artifixer_inputs.resolve_colmap_paths``."""
    image_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    if not image_dir.is_dir():
        raise UnprocessableInput("Scene is missing the images/ directory")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).is_file():
            raise UnprocessableInput(f"Scene is missing sparse/0/{name}")
    return image_dir, sparse_dir


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image as PILImage  # Local import: Pillow is only needed here.
    from PIL import UnidentifiedImageError

    try:
        with PILImage.open(path) as image:
            return image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # A file with an image extension that Pillow cannot decode is bad input,
        # not a server fault.
        raise UnprocessableInput(
            f"{path.name} is not a decodable image: {type(exc).__name__}"
        ) from exc


def validate_scene(scene_dir: Path, *, max_images: int) -> ColmapSummary:
    """Validate an extracted COLMAP scene against every pipeline precondition.

    Raises :class:`UnprocessableInput` with an actionable message, or returns a
    summary. Ordering matters: cheap structural checks run before per-image
    decoding so a bad upload is rejected without touching thousands of files.
    """
    image_dir, sparse_dir = resolve_scene_paths(scene_dir)
    cameras = read_cameras(sparse_dir / "cameras.bin")
    images = read_images(sparse_dir / "images.bin", max_images=max_images)
    point_count = read_point_count(sparse_dir / "points3D.bin")
    if point_count == 0:
        raise UnprocessableInput(
            "sparse/0/points3D.bin is empty. 3DGRUT initializes Gaussians from these points, "
            "so the reconstruction cannot start."
        )

    # prepare_colmap_artifixer_inputs.require_unique_basenames (:174-176).
    basenames = [Path(image.name).name for image in images]
    if len(set(basenames)) != len(basenames):
        duplicates = sorted({name for name in basenames if basenames.count(name) > 1})
        raise UnprocessableInput(f"COLMAP image basenames must be unique; duplicates: {duplicates[:10]}")

    for basename in basenames:
        if Path(basename).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise UnprocessableInput(f"Unsupported image type referenced by COLMAP: {basename!r}")
        # Names read out of images.bin are caller-controlled too. They end up in the
        # newline-delimited selected-views file the prepare CLI parses, so a name
        # containing a newline or a separator would corrupt that file.
        if not IMAGE_NAME_PATTERN.match(basename):
            raise UnprocessableInput(
                f"COLMAP image name {basename!r} contains characters this service does not accept"
            )

    cameras_by_id = {camera.camera_id: camera for camera in cameras}
    used_camera_ids = sorted({image.camera_id for image in images})
    missing_cameras = [camera_id for camera_id in used_camera_ids if camera_id not in cameras_by_id]
    if missing_cameras:
        raise UnprocessableInput(f"images.bin references camera ids absent from cameras.bin: {missing_cameras}")

    used_models = sorted({cameras_by_id[camera_id].model for camera_id in used_camera_ids})
    unsupported = sorted(set(used_models) - SUPPORTED_CAMERA_MODELS)
    if unsupported:
        raise UnprocessableInput(
            f"Unsupported COLMAP camera models: {unsupported}. Supported: "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}"
        )

    # Missing image files: prepare's source_image_path raises FileNotFoundError
    # per image (:179-183). Report them together instead of one at a time.
    missing_files = [basename for basename in basenames if not (image_dir / basename).is_file()]
    if missing_files:
        raise UnprocessableInput(
            f"{len(missing_files)} image(s) referenced by COLMAP are absent from images/: "
            f"{missing_files[:10]}"
        )

    # prepare_colmap_artifixer_inputs.scale_colmap_scene_to_images (:211-259):
    # all images sharing a camera must have identical size, and the
    # image/camera scale factor must be uniform.
    sizes_by_camera: dict[int, tuple[int, int]] = {}
    for image, basename in zip(images, basenames):
        size = _image_size(image_dir / basename)
        previous = sizes_by_camera.setdefault(image.camera_id, size)
        if previous != size:
            raise UnprocessableInput(
                f"COLMAP camera {image.camera_id} is shared by images with different sizes: "
                f"{previous[0]}x{previous[1]} and {size[0]}x{size[1]}"
            )

    for camera_id, (width, height) in sizes_by_camera.items():
        camera = cameras_by_id[camera_id]
        if (width, height) == (camera.width, camera.height):
            continue
        scale_x = width / camera.width
        scale_y = height / camera.height
        # Same tolerance as the np.isclose(sx, sy) upstream uses
        # (prepare_colmap_artifixer_inputs.py:241): rejecting a scene the pipeline
        # would have accepted would be worse than a slightly late failure.
        if not math.isclose(scale_x, scale_y, rel_tol=1e-5, abs_tol=1e-8):
            raise UnprocessableInput(
                f"COLMAP camera {camera_id} is {camera.width}x{camera.height} but its images are "
                f"{width}x{height}; only uniform rescaling is supported"
            )

    # prepare_colmap_artifixer_inputs.shared_camera_intrinsics (:338-347) requires
    # one shared calibration across every used camera.
    reference = None
    for camera_id in used_camera_ids:
        camera = cameras_by_id[camera_id]
        signature = (camera.model, camera.width, camera.height, camera.params)
        if reference is None:
            reference = signature
        elif signature != reference:
            raise UnprocessableInput(
                "ArtiFixer COLMAP preparation expects one shared intrinsic calibration, but "
                f"camera {camera_id} differs from camera {used_camera_ids[0]}"
            )

    first_size = sizes_by_camera[used_camera_ids[0]]
    return ColmapSummary(
        image_count=len(images),
        camera_count=len(used_camera_ids),
        camera_models=tuple(used_models),
        point_count=point_count,
        width=first_size[0],
        height=first_size[1],
        image_names=tuple(basenames),
    )
