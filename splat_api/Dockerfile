# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# COLMAP-to-splat API service.
#
# Layered on the ArtiFixer runtime image, which already carries CUDA, torch,
# 3DGRUT (with its compiled tracer and slang kernels), diffusers/transformers and
# MoGe. Rebuilding that stack here would duplicate a long, fragile build.
#
#   # 1. Build the base image once, from the repo root:
#   docker build -f Dockerfile.cuda12 -t artifixer:cuda12 .
#
#   # 2. Build the service:
#   docker build -f splat_api/Dockerfile -t splat-api:latest .
#
# Use Dockerfile.cuda13 / Dockerfile.cuda13-aarch64 as the base on Blackwell or
# Grace hosts and pass --build-arg BASE_IMAGE accordingly.
ARG BASE_IMAGE=artifixer:cuda12
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- service dependencies -------------------------------------------------
COPY splat_api/requirements.txt /tmp/splat_api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/splat_api-requirements.txt && \
    rm /tmp/splat_api-requirements.txt

# --- application ----------------------------------------------------------
WORKDIR /workspace/artifixer
COPY . /workspace/artifixer

# --- runtime user ---------------------------------------------------------
# The service never needs root. It owns only its data root, which is a volume, so
# the image itself can be mounted read-only.
ARG SERVICE_UID=10001
ARG SERVICE_GID=10001
RUN groupadd --gid ${SERVICE_GID} splat && \
    useradd --uid ${SERVICE_UID} --gid ${SERVICE_GID} --create-home --shell /usr/sbin/nologin splat && \
    mkdir -p /data/splat-api /data/artifixer-checkpoints /data/hf-cache && \
    chown -R splat:splat /data/splat-api /data/artifixer-checkpoints /data/hf-cache && \
    chmod -R g-w,o-rwx /workspace/artifixer

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/artifixer \
    HF_HOME=/data/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled \
    SPLAT_API_DATA_ROOT=/data/splat-api \
    SPLAT_API_REPO_ROOT=/workspace/artifixer \
    SPLAT_API_HOST=0.0.0.0 \
    SPLAT_API_PORT=8000

EXPOSE 8000
VOLUME ["/data/splat-api", "/data/artifixer-checkpoints", "/data/hf-cache"]

# Liveness only: /readyz additionally checks disk and the database, and is the
# right probe for an orchestrator's readiness gate.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('SPLAT_API_PORT','8000')+'/healthz', timeout=4).status==200 else 1)"

USER splat
ENTRYPOINT ["/workspace/artifixer/splat_api/docker/entrypoint.sh"]
CMD ["serve"]
