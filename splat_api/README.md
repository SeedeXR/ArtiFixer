<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# COLMAP to Gaussian Splat API

An HTTP service that takes a COLMAP sparse reconstruction and returns a Gaussian splat.

Upload a ZIP containing `images/` and `sparse/0/*.bin`, get back a `.ply` splat that loads in
any 3DGS viewer. The service drives this repository's real pipeline — 3DGUT reconstruction via
3DGRUT, optionally corrected by the ArtiFixer video diffusion model and re-distilled
(ArtiFixer3D / ArtiFixer3D+) — as isolated subprocesses, one job per GPU.

| Document | Contents |
| --- | --- |
| [docs/API.md](docs/API.md) | Full HTTP reference, error codes, curl walkthrough |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Process flow, per-stage detail, on-disk layout |
| [docs/SECURITY.md](docs/SECURITY.md) | Threat model and every control, with residual risks |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Generated import map of the surrounding repository |

## Quick start

```bash
# 1. Build the ArtiFixer runtime image (CUDA, torch, 3DGRUT, diffusers, MoGe), then the service.
docker build -f Dockerfile.cuda12 -t artifixer:cuda12 .
docker build -f splat_api/Dockerfile -t splat-api:latest .

# 2. Run it. The secret must be at least 24 characters.
export SPLAT_API_KEYS="ci:$(openssl rand -hex 24):read+write"
docker compose -f splat_api/docker-compose.yml up -d

# 3. Reconstruct a scene.
curl -sS -X POST http://127.0.0.1:8000/v1/scenes \
     -H "Authorization: Bearer <secret>" \
     -H 'Content-Type: application/zip' \
     --data-binary @my_scene.zip
# -> {"scene_id":"scene_ab12...","image_count":225,"point_count":184311,...}

curl -sS -X POST http://127.0.0.1:8000/v1/jobs \
     -H "Authorization: Bearer <secret>" -H 'Content-Type: application/json' \
     -d '{"scene_id":"scene_ab12...","mode":"reconstruct"}'
# -> 202 {"job_id":"job_cd34...","state":"queued","stages":[...]}

curl -sS http://127.0.0.1:8000/v1/jobs/job_cd34... -H "Authorization: Bearer <secret>"
# poll until "state":"succeeded"

curl -sS -o splat.ply \
     http://127.0.0.1:8000/v1/jobs/job_cd34.../artifacts/splat.ply \
     -H "Authorization: Bearer <secret>"
```

Running without a container, from the repository root:

```bash
pip install -r splat_api/requirements.txt
export SPLAT_API_DATA_ROOT=/data/splat-api
export SPLAT_API_KEYS="dev:$(openssl rand -hex 24):admin"
python -m splat_api.app.main
```

## Input contract

```text
my_scene.zip
  images/                       # or images_2 / images_4 / images_8
    000001.jpg
    ...
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

A single top-level scene folder (`my_scene/images/...`) is fine; the extractor locates the model
by its `sparse/0/cameras.bin` marker and strips the prefix. Only these files are extracted —
anything else in the archive is ignored rather than written to disk.

Validation happens during the upload request, so a bad scene is a `422` in seconds rather than a
subprocess traceback minutes into a GPU job. The checks mirror the assertions in
`data_processing/prepare_colmap_artifixer_inputs.py`:

- all three sparse binaries present and parseable, `points3D.bin` non-empty
- every image referenced by `images.bin` present in `images/`, with unique basenames
- camera model in `{SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV}`
- one shared intrinsic calibration across all used cameras
- images sharing a camera have identical dimensions, and any camera/image size difference is a
  uniform scale

That last rule rejects a common real-world case: a scene whose `cameras.bin` says 1957x1091 while
`images/` holds 979x546 downsampled copies. The two axes round differently (0.50026 vs 0.50046),
which `prepare_colmap_artifixer_inputs` also refuses. Either ship the full-resolution images or
re-run COLMAP at the resolution you are shipping.

### Faster alternatives to uploading a ZIP

ZIP upload is the portable path and is what most clients should use. Two faster options exist:

| Option | When to use | Cost |
| --- | --- | --- |
| `POST /v1/scenes` with `Content-Type: application/zip` | Default. Raw body streams straight to disk, no multipart framing, hashed while writing. | One disk write plus one extraction pass; extraction is threaded. |
| Same endpoint, `dedupe=true` (default) | Re-submitting a scene. An identical archive returns the existing `scene_id` with `200` and `X-Scene-Deduplicated: true`. | Upload only; no extraction, no validation, no new copy on disk. |
| `POST /v1/scenes/register` (admin) | The capture is already on server storage under `<data_root>/import`. | Zero copy. Validation only. This is the fastest possible ingest and the right choice for large captures. |

## Pipeline modes

| Mode | Stages | Needs an ArtiFixer checkpoint | Output |
| --- | --- | --- | --- |
| `reconstruct` | prepare -> export | no | 3DGUT splat |
| `artifixer3d` | prepare -> artifixer -> artifixer3d -> export | yes | Splat distilled from ArtiFixer-corrected views |
| `artifixer3d_plus` | adds a second `artifixer` pass over the ArtiFixer3D renders | yes | Same splat plus twice-corrected frames |

`reconstruct` deliberately skips the metric-scale (MoGe) and captioning (Qwen3-VL) phases: those
exist only to condition ArtiFixer, so running them would download tens of gigabytes of weights and
burn GPU time without changing the splat.

The `artifixer3d` modes need at least one frame to generate, so either pass
`selected_image_names` as a **strict subset** of the scene images (the rest become ArtiFixer
targets) or pass a `trajectory` of novel cameras. Requesting them with every image selected and no
trajectory is a `422`, because `data_processing/artifixer3d.py` has nothing to distill.

### Automatic validation holdout

When you omit `selected_image_names`, the API holds back every 8th image and passes the rest as
3DGRUT training anchors; the job response reports `validation_holdout_auto: true` and the resulting
`selected_image_count`. This is not a preference — 3DGRUT always builds a validation dataset
alongside the training one, and with a selected-indices file the validation split is
`setdiff1d(all, selected)`. Selecting every image leaves it empty and training dies with
`IndexError: min(): Expected reduction dim 0 to have non-zero size`. Since
`prepare_colmap_artifixer_inputs` always passes that file, holding views back is the only route
through the public CLI. The interval matches the engine's own `test_split_interval: 8`. Pass
`selected_image_names` to choose the split yourself.

See [docs/PIPELINE.md](docs/PIPELINE.md) for the flow diagram, the exact command line each stage
runs, and the on-disk layout.

## Artifacts

| Name | Kind | Notes |
| --- | --- | --- |
| `splat.ply` | `splat_ply` | Binary 3DGS-compatible PLY: `x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*` |
| `splat_checkpoint.pt` | `splat_checkpoint` | The 3DGRUT checkpoint behind the splat |
| `splat_stats.json` | `metadata` | Gaussian count, SH degree, training step |
| `corrected_frames.zip` | `frames` | ArtiFixer-corrected PNGs (correction modes only) |
| `reconstruction_preview.mp4`, `artifixer3d_preview.mp4` | `video` | Renderer turntables |
| `manifest.json` | `metadata` | Full run record: request, stage timings, exact commands |
| `logs/<stage>.log` | `log` | stdout/stderr per stage. Requires the `write` scope: raw subprocess output carries container paths and tracebacks. |

Every artifact carries a SHA-256 that is also served as a strong `ETag`. Downloads use
`sendfile`, so a multi-gigabyte splat never passes through Python buffers. `manifest.json`
is downloadable, so it carries the same field projection as the HTTP job response and
replaces deployment paths with `<data_root>`/`<repo_root>` placeholders.

## Performance design

- **One worker per GPU.** A stage is a full CUDA process; admitting more concurrent jobs than
  devices only causes OOM. Each worker pins its children with `CUDA_VISIBLE_DEVICES`, and the
  queue is the admission control. `SPLAT_API_MAX_CONCURRENT_JOBS` cannot exceed the visible GPUs.
- **Reconstruction cache.** The base 3DGUT checkpoint is keyed by `(scene, steps, view set)` and
  hardlinked into a cache. A second job on the same scene and views skips reconstruction entirely
  via `--reconstruction_checkpoint`, which is the single biggest saving available.
- **Nothing heavy in the API process.** It never imports torch or `threedgrut`; COLMAP parsing is
  a dependency-free reader. Startup is sub-second and RSS stays small.
- **Threaded extraction.** Archive members are inflated by a small thread pool (zlib releases the
  GIL), each thread with its own `ZipFile` handle.
- **No event-loop blocking.** All disk and database work is dispatched to threads; stage output is
  streamed to log files as it is produced.
- **Cheap artifact publication.** Large binaries are hardlinked, not copied; frame bundles use
  `ZIP_STORED` because PNG is already compressed.
- **Durable queue.** Jobs live in SQLite (WAL). A restart re-admits interrupted jobs, and because
  every upstream stage skips work whose outputs already exist, they resume rather than repeat.

## Configuration

Every setting is read once at startup from `SPLAT_API_*` environment variables; nothing in the
request path reads the environment. Startup fails loudly on a bad value. The full table is in
[docs/API.md](docs/API.md) and [docs/PIPELINE.md](docs/PIPELINE.md); the ones that matter most:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPLAT_API_KEYS` | — | `id:secret:scopes` entries separated by `\|`. Required unless auth is disabled. |
| `SPLAT_API_DATA_ROOT` | `/data/splat-api` | Scenes, jobs, artifacts, database |
| `SPLAT_API_ARTIFIXER_CHECKPOINT` | unset | Enables the correction modes |
| `SPLAT_API_ARTIFIXER_MODEL_ID` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Must match the checkpoint variant |
| `SPLAT_API_MAX_CONCURRENT_JOBS` | GPU count | Workers, one GPU each |
| `SPLAT_API_REQUIRE_AUTH` | `true` | Set `false` only on a trusted network; admin stays gated |

Secrets may be supplied pre-hashed as `id:sha256:<hex>:scopes` so a live credential never appears
in the environment.

## Self-sufficient container

With `SPLAT_API_AUTO_DOWNLOAD=1` the entrypoint fetches everything the pipeline needs before
serving, into the mounted volumes, and pairs the checkpoint with its base model automatically:

```bash
docker compose -f splat_api/docker-compose.yml run --rm \
  -e SPLAT_API_AUTO_DOWNLOAD=1 -e SPLAT_API_CHECKPOINT_VARIANT=1.3b splat-api fetch
```

That pulls the ArtiFixer release checkpoint, the matching Wan2.1 base (config, tokenizer, VAE,
UMT5 text encoder), the Qwen3-VL captioner and MoGe v2. It is opt-in so a routine restart never
re-downloads tens of gigabytes and an air-gapped deployment is never surprised by egress. For a
`reconstruct`-only deployment, `--skip captioner --skip moge` avoids the largest downloads:

```bash
python -m splat_api.tools.fetch_models --variant 1.3b --skip captioner --skip moge
```

### Size the volumes first

Everything lands in the mounted volumes, not in the image, so the image stays small but the volumes
do not. Measured sizes for the 1.3B configuration:

| Component | Size | Needed by |
| --- | --- | --- |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 58 GB | `caption` phase (correction modes only) |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 27 GB | ArtiFixer inference and caption embeddings |
| `nvidia/ArtiFixer` `artifixer-1.3b.pt` | 6.3 GB | ArtiFixer inference |
| `Ruicheng/moge-2-vitl-normal` | 1.3 GB | `scale` phase (correction modes only) |
| **Total** | **~93 GB** | |

Provision at least 150 GB for the weight volumes plus whatever the job outputs need. A
`reconstruct`-only deployment needs none of it — that mode runs no captioning, no metric alignment
and no ArtiFixer inference — so `--skip captioner --skip moge` cuts the download to ~33 GB, and
skipping the checkpoint entirely cuts it to zero.

The fetcher is idempotent: it skips a checkpoint that is already on disk and lets the Hugging Face
cache short-circuit the rest, so a restart re-checks metadata rather than re-downloading.

### Optional: bind an existing cache instead of downloading

For local testing on a machine that already holds these weights, each mount also accepts an
absolute host path in place of its named volume, which avoids a second copy:

```bash
SPLAT_API_HF_CACHE_VOLUME=/root/.cache/huggingface \
SPLAT_API_CHECKPOINT_VOLUME=/data/artifixer-checkpoints \
docker compose -f splat_api/docker-compose.yml up -d
```

Leave `SPLAT_API_AUTO_DOWNLOAD` at `0` in that case. The entrypoint still finds the checkpoint in
the mounted directory and pairs it with the matching base model id.

## Tests

```bash
python -m pytest splat_api/tests -q          # 275 tests, ~3 minutes, no GPU required
```

Three layers:

- **Unit** — `test_paths`, `test_archives`, `test_colmap_input`, `test_security`, `test_jobstore`,
  `test_exporter`. Includes adversarial archives: zip-slip, symlink members, encrypted entries,
  unsupported compression methods, compression bombs, duplicate basenames, truncated binaries.
- **Regression against upstream** — the drift guards. `test_colmap_input` asserts the built-in
  COLMAP reader agrees with `threedgrut.datasets.utils`; `test_exporter` asserts the PLY writer is
  byte-identical to `threedgrut.export.ply_exporter`; `test_pipeline` recomputes every derived path
  with the repo's own helpers (`prepared_paths`, `reconstruction_checkpoint`, `artifixer3d_paths`,
  `run_inference.get_output_dir`) and compares. If an upstream release renames a directory, these
  fail instead of the service silently looking in the wrong place.
- **Defect regression** — `test_hardening` pins each of the 18 findings from an adversarial
  review of this service, most importantly that a stage emitting 250 KB without a newline
  completes instead of permanently wedging its GPU worker. Each test's docstring records the
  original failure.
- **Integration** — `test_api` drives the real ASGI app, middleware, auth, SQLite store,
  subprocess execution, artifact collection and downloads. Only the four heavy stage commands are
  substituted with Python stand-ins that write the same files at the same paths, so the service
  itself is never mocked. It parses the delivered PLY and asserts the correction modes deliver the
  distilled checkpoint rather than the base one.

Both paths have been run for real on one A100-80GB against the Deep Blending `playroom`
scene (225 images at 1264x832, 1,000 steps):

| Mode | Wall clock, end to end over HTTP | Result |
| --- | --- | --- |
| `reconstruct` | 263 s | 44,977 Gaussians, 62 PLY properties, 11.2 MB |
| `artifixer3d` (1.3B checkpoint, 200 anchors / 25 targets) | 849 s | Splat from the distilled checkpoint, plus 25 corrected frames |

Those were short runs to exercise the whole path; the release defaults are 10,000
reconstruction and 30,000 ArtiFixer3D steps. See
[docs/PIPELINE.md](docs/PIPELINE.md#measured-runtime-a-verified-reconstruct-run) for the
per-stage breakdown.

Real-GPU acceptance, nothing stubbed — starts the service, talks HTTP, runs actual 3DGRUT:

```bash
python -m splat_api.tools.e2e_real \
    --colmap-scene /data/colmap-scenes/playroom \
    --mode reconstruct --reconstruction-steps 1000

python -m splat_api.tools.e2e_real \
    --colmap-scene /data/colmap-scenes/playroom --mode artifixer3d \
    --checkpoint /data/artifixer-checkpoints/artifixer-1.3b.pt \
    --selected-views 24 --reconstruction-steps 1000 --artifixer3d-steps 1000
```

Point the reader test at a real capture to include it in the suite:

```bash
SPLAT_API_TEST_COLMAP_SCENE=/data/colmap-scenes/playroom python -m pytest splat_api/tests -q
```

## Layout

```text
splat_api/
  app/
    main.py          ASGI factory, lifespan, error handlers, health, metrics
    routes.py        Scenes, jobs, artifacts, logs, capabilities
    schemas.py       Request/response models (extra="forbid")
    security.py      Auth, scopes, request context, body limits, rate limiting
    config.py        Environment parsing and validation
    archives.py      Hardened ZIP extraction (allowlist, threaded)
    colmap_input.py  Dependency-free COLMAP reader and precondition checks
    jobstore.py      SQLite scene/job records, keyset pagination, restart recovery
    scheduler.py     GPU workers, subprocess stages, artifacts, caching
    pipeline.py      The only module that knows the repo's CLI contract
    exporter.py      3DGRUT checkpoint -> 3DGS PLY
  docker/entrypoint.sh
  tools/             fetch_models, e2e_real, repo_map
  tests/
  docs/
```
