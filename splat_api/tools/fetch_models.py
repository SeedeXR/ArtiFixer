#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch every weight the pipeline needs, so a container can be self-sufficient.

Four artifacts, all from Hugging Face:

===========================  ==============================================
ArtiFixer release checkpoint  ``nvidia/ArtiFixer`` -> ``artifixer-{variant}.pt``
Wan2.1 base model             architecture config + UMT5 text encoder
Qwen3-VL captioner            scene captions for the ArtiFixer stages
MoGe v2                       monocular depth for metric-scale alignment
===========================  ==============================================

Only the ArtiFixer checkpoint and the matching Wan base are needed for
``mode=reconstruct``; that mode does not run captioning or metric alignment at
all. The captioner is by far the largest download, so ``--skip captioner`` is a
reasonable choice on a reconstruct-only deployment.

Idempotent: Hugging Face caches by content hash, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHECKPOINT_REPO = "nvidia/ArtiFixer"

# README.md: "artifixer-14b.pt -> Wan-AI/Wan2.1-T2V-14B-Diffusers",
# "artifixer-1.3b.pt -> Wan-AI/Wan2.1-T2V-1.3B-Diffusers". Loading a checkpoint
# against the wrong base fails with shape mismatches, so the pairing is fixed.
VARIANTS = {
    "1.3b": ("artifixer-1.3b.pt", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    "14b": ("artifixer-14b.pt", "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
}

# prepare_colmap_artifixer_inputs.py:54 DEFAULT_CAPTIONING_MODEL_ID
CAPTIONING_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# sparse_recon/metric_alignment.py:28 default MoGe variant
MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"

COMPONENTS = ("checkpoint", "base_model", "captioner", "moge")


def log(message: str) -> None:
    print(f"[fetch_models] {message}", flush=True)


def fetch_checkpoint(variant: str, destination: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename, _ = VARIANTS[variant]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.is_file():
        log(f"checkpoint already present: {target}")
        return target
    log(f"downloading {CHECKPOINT_REPO}/{filename}")
    path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=filename, local_dir=str(destination))
    log(f"checkpoint ready: {path}")
    return Path(path)


def fetch_repo_snapshot(repo_id: str, *, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    log(f"downloading {repo_id}")
    snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    log(f"ready: {repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="1.3b")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/data/artifixer-checkpoints"))
    parser.add_argument(
        "--skip",
        action="append",
        choices=COMPONENTS,
        default=[],
        help="Component to skip; repeatable.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=COMPONENTS,
        default=[],
        help="Fetch only these components; repeatable.",
    )
    args = parser.parse_args(argv)

    wanted = set(args.only) if args.only else set(COMPONENTS)
    wanted -= set(args.skip)

    _, base_model = VARIANTS[args.variant]
    try:
        if "checkpoint" in wanted:
            fetch_checkpoint(args.variant, args.checkpoint_dir)
        if "base_model" in wanted:
            # The transformer weights come from the release checkpoint; from the
            # base repo we need the configs, tokenizer, VAE and text encoder.
            fetch_repo_snapshot(base_model)
        if "captioner" in wanted:
            fetch_repo_snapshot(CAPTIONING_MODEL)
        if "moge" in wanted:
            fetch_repo_snapshot(MOGE_MODEL)
    except Exception as exc:  # noqa: BLE001 - the message is the useful part here
        print(f"[fetch_models] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    log("all requested components are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
