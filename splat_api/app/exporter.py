# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a 3DGRUT checkpoint into a 3DGS-compatible PLY splat.

Attribute layout, ordering and activation conventions are those of
``threedgrut.export.ply_exporter.PLYExporter``
(thirdparty/3DGRUT-ArtiFixer/threedgrut/export/ply_exporter.py:33-84):

    x y z nx ny nz f_dc_0..2 f_rest_0..M opacity scale_0..2 rot_0..3

with ``opacity``/``scale``/``rot`` stored **pre-activation** (the raw optimized
parameters), matching the inria 3DGS convention that every splat viewer expects.

This module re-implements the writer instead of importing 3DGRUT so that export
needs neither CUDA nor a Hydra config, and so it runs in ~1s instead of importing
the whole tracer stack. ``tests/test_exporter_regression.py`` asserts byte-level
agreement with the upstream exporter, which is what makes that trade safe.

Run as a module (the API never exposes this as an endpoint)::

    python -m splat_api.app.exporter --checkpoint ckpt_30000.pt --output splat.ply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keys written by MixtureOfGaussians.get_model_parameters
# (threedgrut/model/model.py:111-139).
REQUIRED_KEYS = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be turned into a splat."""


@dataclass(frozen=True)
class SplatStats:
    num_gaussians: int
    sh_degree: int
    global_step: int | None
    output_path: Path
    output_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_gaussians": self.num_gaussians,
            "sh_degree": self.sh_degree,
            "global_step": self.global_step,
            "ply_bytes": self.output_bytes,
        }


def _to_numpy(value: Any, name: str) -> np.ndarray:
    """Detach any tensor-like checkpoint entry into a float32 numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ExportError(f"Checkpoint entry {name!r} must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ExportError(f"Checkpoint entry {name!r} contains non-finite values")
    return array


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a 3DGRUT checkpoint.

    3DGRUT stores its OmegaConf config inside the checkpoint, so
    ``weights_only=True`` cannot deserialize it. We try the safe mode first and
    only fall back to a full unpickle, which is acceptable because this module is
    invoked exclusively by the scheduler on checkpoints the pipeline just wrote
    inside the job directory. It is never pointed at caller-supplied bytes.
    """
    import torch

    if not checkpoint_path.is_file():
        raise ExportError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ExportError(f"Checkpoint {checkpoint_path.name} does not contain a parameter dict")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ExportError(
            f"Checkpoint {checkpoint_path.name} is missing 3DGRUT model parameters: {missing}. "
            "Only 3DGRUT Gaussian checkpoints can be exported."
        )
    return payload


def attribute_names(num_albedo: int, num_specular: int) -> list[str]:
    """PLY property order; mirrors ``PLYExporter._construct_list_of_attributes``."""
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(num_albedo)]
    names += [f"f_rest_{index}" for index in range(num_specular)]
    names.append("opacity")
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    return names


def build_vertex_array(payload: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Assemble the structured vertex array and return it with the SH degree."""
    positions = _to_numpy(payload["positions"], "positions")
    rotation = _to_numpy(payload["rotation"], "rotation")
    scale = _to_numpy(payload["scale"], "scale")
    density = _to_numpy(payload["density"], "density")
    albedo = _to_numpy(payload["features_albedo"], "features_albedo")
    specular = _to_numpy(payload["features_specular"], "features_specular")

    count = positions.shape[0]
    if count == 0:
        raise ExportError("Checkpoint contains zero Gaussians")
    for name, array, width in (
        ("positions", positions, 3),
        ("rotation", rotation, 4),
        ("scale", scale, 3),
        ("density", density, 1),
        ("features_albedo", albedo, 3),
    ):
        if array.shape != (count, width):
            raise ExportError(f"Checkpoint entry {name!r} has shape {array.shape}, expected {(count, width)}")
    if specular.shape[0] != count:
        raise ExportError(
            f"features_specular has {specular.shape[0]} rows but there are {count} Gaussians"
        )

    # features_specular is [N, 3 * ((degree+1)^2 - 1)]; see
    # threedgrut/utils/misc.py:114-116 sh_degree_to_specular_dim.
    specular_width = specular.shape[1]
    if specular_width % 3 != 0:
        raise ExportError(f"features_specular width {specular_width} is not divisible by 3")
    num_specular_coeffs = specular_width // 3
    sh_degree = int(round((num_specular_coeffs + 1) ** 0.5)) - 1
    if (sh_degree + 1) ** 2 - 1 != num_specular_coeffs:
        raise ExportError(
            f"features_specular width {specular_width} does not correspond to a whole SH degree"
        )

    # Upstream reshapes to (N, coeffs, 3) then transposes to channel-major before
    # flattening, so f_rest is ordered channel-major exactly like inria 3DGS.
    specular_channel_major = (
        specular.reshape(count, num_specular_coeffs, 3).transpose(0, 2, 1).reshape(count, specular_width)
    )
    normals = np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), repeats=count, axis=0)

    attributes = np.concatenate(
        (positions, normals, albedo, specular_channel_major, density, scale, rotation), axis=1
    )
    names = attribute_names(albedo.shape[1], specular_width)
    if attributes.shape[1] != len(names):
        raise ExportError(
            f"Assembled {attributes.shape[1]} columns but derived {len(names)} property names"
        )

    vertices = np.empty(count, dtype=[(name, "f4") for name in names])
    for index, name in enumerate(names):
        vertices[name] = attributes[:, index]
    return vertices, sh_degree


def write_ply(vertices: np.ndarray, output_path: Path) -> int:
    """Write ``vertices`` as a binary little-endian PLY, atomically.

    The temporary file lives in the destination directory so the final rename is
    atomic on the same filesystem: a consumer never observes a partial splat.
    """
    from plyfile import PlyData, PlyElement

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".partial")
    element = PlyElement.describe(vertices, "vertex")
    PlyData([element]).write(str(temp_path))
    size = temp_path.stat().st_size
    os.replace(temp_path, output_path)
    return size


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> SplatStats:
    payload = load_checkpoint(checkpoint_path)
    vertices, sh_degree = build_vertex_array(payload)
    size = write_ply(vertices, output_path)
    global_step = payload.get("global_step")
    return SplatStats(
        num_gaussians=int(vertices.shape[0]),
        sh_degree=sh_degree,
        global_step=int(global_step) if isinstance(global_step, (int, float)) else None,
        output_path=output_path,
        output_bytes=size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="3DGRUT checkpoint (.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Destination .ply path")
    parser.add_argument(
        "--stats-json", type=Path, default=None, help="Optional path for machine-readable export stats"
    )
    args = parser.parse_args(argv)

    try:
        stats = export_checkpoint(args.checkpoint.resolve(), args.output.resolve())
    except ExportError as exc:
        print(f"export_error={exc}", file=sys.stderr, flush=True)
        return 1

    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats.as_dict(), indent=2) + "\n")
    print(f"ply_path={stats.output_path}", flush=True)
    print(f"num_gaussians={stats.num_gaussians}", flush=True)
    print(f"sh_degree={stats.sh_degree}", flush=True)
    print(f"ply_bytes={stats.output_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
