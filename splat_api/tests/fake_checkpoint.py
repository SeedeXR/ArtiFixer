# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a 3DGRUT-shaped Gaussian checkpoint without running 3DGRUT.

Keys and shapes match ``MixtureOfGaussians.get_model_parameters``
(threedgrut/model/model.py:111-139) for the ``sh`` feature type, which is what the
exporter consumes. Imported both by tests and by the fake pipeline stages, so it
lives in its own module rather than inside a fixture.
"""

from __future__ import annotations

from pathlib import Path


def fake_model_parameters(*, num_gaussians: int = 64, sh_degree: int = 3) -> dict:
    import torch

    generator = torch.Generator().manual_seed(1234)
    specular_dim = 3 * ((sh_degree + 1) ** 2 - 1)
    return {
        "positions": torch.rand((num_gaussians, 3), generator=generator) * 2 - 1,
        "rotation": torch.rand((num_gaussians, 4), generator=generator),
        "scale": torch.rand((num_gaussians, 3), generator=generator) - 3.0,
        "density": torch.rand((num_gaussians, 1), generator=generator),
        "features_albedo": torch.rand((num_gaussians, 3), generator=generator),
        "features_specular": torch.rand((num_gaussians, specular_dim), generator=generator),
        "n_active_features": sh_degree,
        "max_n_features": sh_degree,
        "progressive_training": False,
        "scene_extent": 1.0,
        "global_step": 100,
        "background": {},
    }


def write_fake_checkpoint(path: Path, *, num_gaussians: int = 64, sh_degree: int = 3) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fake_model_parameters(num_gaussians=num_gaussians, sh_degree=sh_degree), path)
    return path
