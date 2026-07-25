#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint.
#
#   serve      (default) optionally fetch models, then run the API
#   fetch      fetch models and exit
#   test       run the service test suite and exit
#   <command>  exec anything else verbatim

set -euo pipefail

REPO_ROOT="${SPLAT_API_REPO_ROOT:-/workspace/artifixer}"
cd "${REPO_ROOT}"

log() { printf '{"level":"INFO","logger":"entrypoint","message":"%s"}\n' "$1"; }
fail() { printf '{"level":"ERROR","logger":"entrypoint","message":"%s"}\n' "$1" >&2; exit 1; }

fetch_models() {
    # Opt-in so a normal restart never re-downloads tens of gigabytes, and so an
    # air-gapped deployment is never surprised by an outbound request.
    if [[ "${SPLAT_API_AUTO_DOWNLOAD:-0}" != "1" ]]; then
        log "SPLAT_API_AUTO_DOWNLOAD is not 1; skipping model download"
        return 0
    fi
    python -m splat_api.tools.fetch_models \
        --variant "${SPLAT_API_CHECKPOINT_VARIANT:-1.3b}" \
        --checkpoint-dir "${SPLAT_API_CHECKPOINT_DIR:-/data/artifixer-checkpoints}"
}

case "${1:-serve}" in
    serve)
        fetch_models
        # Resolve the checkpoint here rather than in the image so a volume that
        # gains a checkpoint later needs only a restart, not a rebuild.
        if [[ -z "${SPLAT_API_ARTIFIXER_CHECKPOINT:-}" ]]; then
            variant="${SPLAT_API_CHECKPOINT_VARIANT:-1.3b}"
            candidate="${SPLAT_API_CHECKPOINT_DIR:-/data/artifixer-checkpoints}/artifixer-${variant}.pt"
            if [[ -f "${candidate}" ]]; then
                export SPLAT_API_ARTIFIXER_CHECKPOINT="${candidate}"
                log "using ArtiFixer checkpoint ${candidate}"
            else
                log "no ArtiFixer checkpoint found; only mode=reconstruct will be offered"
            fi
        fi
        if [[ -n "${SPLAT_API_ARTIFIXER_CHECKPOINT:-}" && -z "${SPLAT_API_ARTIFIXER_MODEL_ID:-}" ]]; then
            # A 1.3B checkpoint loaded against the default 14B config fails with
            # shape mismatches, so pair the two automatically.
            case "${SPLAT_API_ARTIFIXER_CHECKPOINT}" in
                *1.3b*) export SPLAT_API_ARTIFIXER_MODEL_ID="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" ;;
                *) export SPLAT_API_ARTIFIXER_MODEL_ID="Wan-AI/Wan2.1-T2V-14B-Diffusers" ;;
            esac
            log "inferred model id ${SPLAT_API_ARTIFIXER_MODEL_ID}"
        fi
        exec python -m splat_api.app.main
        ;;
    fetch)
        SPLAT_API_AUTO_DOWNLOAD=1 fetch_models
        ;;
    test)
        exec python -m pytest splat_api/tests -q
        ;;
    *)
        exec "$@"
        ;;
esac
