<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# splat_api HTTP reference

`splat_api` turns a COLMAP sparse reconstruction into a Gaussian splat using 3DGUT,
optionally corrected by the ArtiFixer video diffusion model (ArtiFixer3D /
ArtiFixer3D+). All pipeline work is asynchronous: you upload a scene, create a job,
poll it, and download artifacts.

- Base path for the resource API: `/v1`
- Media type: `application/json` unless noted
- Service version is reported by `GET /v1/capabilities` (`service_version`)

Contents: [Quick start](#quick-start) · [Authentication](#authentication) ·
[Errors](#errors) · [Rate limiting](#rate-limiting) · [Endpoints](#endpoints) ·
[Configuration](#configuration-http-surface)

Two behaviours surprise first-time callers and are worth reading before you submit
anything: [automatic validation holdout](#automatic-validation-holdout) (omitting
`selected_image_names` does **not** train on every image) and the
[non-uniform camera/image scale limitation](#known-input-limitation-non-uniform-cameraimage-scale-ratio)
(which rejects some published datasets as shipped).

---

## Quick start

The walkthrough below uploads a COLMAP scene, runs a reconstruct-only job, and
downloads `splat.ply`.

The archive must contain `images/` and `sparse/0/{cameras,images,points3D}.bin`
(see [Input contract](#input-contract)):

```bash
export SPLAT_API=https://splat.internal.example
export SPLAT_KEY=sk_live_your_write_scoped_key

# 1. Upload the scene (raw body is the fast path; no multipart framing).
scene_id=$(curl -sS -X POST "$SPLAT_API/v1/scenes?label=truck" \
  -H "Authorization: Bearer $SPLAT_KEY" \
  -H "Content-Type: application/zip" \
  --data-binary @truck.zip | jq -r .scene_id)

# 2. Create a job. 202 Accepted; Location points at the job resource.
job_id=$(curl -sS -X POST "$SPLAT_API/v1/jobs" \
  -H "Authorization: Bearer $SPLAT_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"scene_id\":\"$scene_id\",\"mode\":\"reconstruct\",\"client_token\":\"truck-run-0001\"}" \
  | jq -r .job_id)

# 3. Poll until terminal. progress is (succeeded+skipped)/total stages.
while :; do
  state=$(curl -sS "$SPLAT_API/v1/jobs/$job_id" \
    -H "Authorization: Bearer $SPLAT_KEY" | jq -r .state)
  echo "state=$state"
  case "$state" in succeeded|failed|cancelled) break;; esac
  sleep 30
done

# 4. Download the splat.
curl -sS -o splat.ply \
  "$SPLAT_API/v1/jobs/$job_id/artifacts/splat.ply" \
  -H "Authorization: Bearer $SPLAT_KEY"
```

Useful while polling: `GET /v1/jobs/{job_id}/logs/prepare` streams the tail of the
current stage's stdout/stderr.

---

## Authentication

Two header forms are accepted, checked in this order:

| Header | Format | Notes |
| --- | --- | --- |
| `Authorization` | `Bearer <secret>` | Any other scheme, or an empty value, is `401 unauthorized`. |
| `X-API-Key` | `<secret>` | Used only when `Authorization` is absent. |

Keys are configured out-of-band in `SPLAT_API_KEYS` as `|`-separated entries:

```text
SPLAT_API_KEYS='ci:sk_live_at_least_24_chars_long:write|ops:sha256:9f86d0...:admin'
```

- `key_id:secret:scopes` — the secret must be at least 24 characters and cannot
  contain `:`.
- `key_id:sha256:<64 hex>:scopes` — pre-hashed, so no live credential need appear
  in the environment.
- `key_id` must match `^[A-Za-z0-9_.-]{1,64}$`; duplicate ids or duplicate secrets
  are a startup error.
- `scopes` is `+`-separated from `read`, `write`, `admin`. Granting `admin`
  implicitly grants `read` and `write`.

Only the SHA-256 digest of a secret is held in memory. Comparison is
constant-time and walks the whole key list, so neither the value nor the number of
configured keys is observable through response timing.

### Scopes

| Scope | Grants |
| --- | --- |
| `read` | `GET /v1/capabilities`, all scene/job/artifact/log reads |
| `write` | `POST /v1/scenes`, `POST /v1/jobs`, `POST /v1/jobs/{id}/cancel` |
| `admin` | `POST /v1/scenes/register`, `DELETE /v1/scenes/{id}` |

`admin` is the only scope with an implication: a key whose spec contains `admin`
is expanded to `read+write+admin` at parse time, because an admin key that cannot
read the resources it manages is never useful. There is no other hierarchy — a
handler requiring `read` rejects a `write`-only key with `403 forbidden`, so grant
`read+write` explicitly for a client that both submits and polls.

When `SPLAT_API_REQUIRE_AUTH=false`, a request with no credential is treated as
principal `anonymous` with `read` and `write` — never `admin`, because admin
endpoints name server-side filesystem paths. A credential that *is* offered is
still validated, and a wrong one is rejected rather than silently downgraded.

`/healthz`, `/readyz`, `/metrics`, `/docs`, and `/openapi.json` require no
credential.

### Response headers on every response

`X-Request-Id` (echoed from a client-supplied `X-Request-Id` only when it is
alphanumeric and at most 64 characters, otherwise generated), plus
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`,
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'`,
and `Cache-Control: no-store`.

---

## Errors

Every error is a JSON envelope. `request_id` matches the `X-Request-Id` response
header and the server log record.

```json
{
  "error": {
    "code": "unprocessable_input",
    "message": "Archive does not contain sparse/0/cameras.bin. Expected layout: images/ and sparse/0/{cameras,images,points3D}.bin",
    "request_id": "3f1c8a90b2d4e7f1"
  }
}
```

`details` is added when a code carries structured context (currently:
`validation_error`, which puts the pydantic error list under
`details.errors`).

| Code | HTTP | Meaning |
| --- | --- | --- |
| `bad_request` | 400 | Malformed request: a `Content-Type` on an upload that is none of `application/zip`, `application/x-zip-compressed`, `application/octet-stream`, `multipart/form-data`; a `multipart/form-data` upload with no `file` part; a non-integer `Content-Length`; blank `label`; empty body; invalid id or image name; a ZIP that is not a ZIP, an encrypted ZIP, a symlink member, or a path component that escapes its root. |
| `unauthorized` | 401 | Missing credential, non-`Bearer` `Authorization` scheme, or an unknown key. |
| `forbidden` | 403 | Authenticated, but the credential lacks the scope the endpoint requires. |
| `not_found` | 404 | Unknown `scene_id`/`job_id`, unknown artifact name, an artifact whose file is gone, an unknown stage name, no log written yet, or a `register` path that does not exist. |
| `conflict` | 409 | `client_token` already used for a different scene/mode, cancelling an already-finished job, or deleting a scene that still has queued/running jobs. |
| `payload_too_large` | 413 | Declared `Content-Length` above `SPLAT_API_MAX_JSON_BYTES` (for `application/json`) or above `SPLAT_API_MAX_REQUEST_BYTES` (everything else), upload bytes above `SPLAT_API_MAX_UPLOAD_BYTES`, archive member count / uncompressed size / compression ratio / image count above their caps, or a trajectory with more frames than `SPLAT_API_MAX_TRAJECTORY_FRAMES`. |
| `unprocessable_input` | 422 | The request parsed but the referenced data is unusable: archive layout wrong, sparse files missing or truncated, fewer than 2 images, duplicate image basenames, empty `points3D.bin`, unsupported camera model, non-uniform image/camera scaling, more than one intrinsic calibration, a step count above the deployment's configured maximum, `selected_image_names` not present in the scene, `selected_image_names` not a strict subset for an `artifixer3d` mode, no `<data_root>/import` directory, or a registered scene directory that has disappeared. |
| `validation_error` | 422 | Request body failed pydantic validation (unknown field, wrong type, out-of-range value, failed model validator). Emitted by the framework handler, with `details.errors`. |
| `rate_limited` | 429 | Token bucket exhausted. Carries `Retry-After`. |
| `internal_error` | 500 | Unhandled server-side failure. The message is always the literal `"Internal server error"`; correlate with logs via `request_id`. |
| `service_unavailable` | 503 | `mode` requires an ArtiFixer checkpoint that is not configured, the job queue is full, or the service is shutting down. |

Notes:

- A non-integer `Content-Length` header returns `400 bad_request` with the message
  `"Malformed Content-Length header"`. Although the check runs in middleware rather
  than a handler, the envelope carries `request_id` like every other error, because
  the request-context middleware runs outermost and has already assigned one.
- `GET /readyz` returns `503` with a `HealthStatus` body, not an error envelope.

---

## Rate limiting

An in-process token bucket, applied before any handler work.

| Property | Value |
| --- | --- |
| Refill rate | `SPLAT_API_RATE_LIMIT_PER_MINUTE / 60` tokens per second (default 120/min) |
| Bucket capacity | `SPLAT_API_RATE_LIMIT_BURST` (default 40) |
| Cost | 1 token per request |
| Bucket key | `key:<first 32 hex of sha256(credential header)>` when `Authorization` or `X-API-Key` is present, otherwise `ip:<client host>` |
| Exempt paths | `/healthz`, `/readyz` |
| Disabled when | `SPLAT_API_RATE_LIMIT_PER_MINUTE=0` |

On rejection: `429` with code `rate_limited` and a `Retry-After` header in whole
seconds, always at least `1`, so a client that honours it makes progress. Buckets
are per process; the service runs a single API process by design.

### Body size pre-filter

A separate middleware rejects an oversized request from its declared
`Content-Length` alone, with `413`, before any handler work. Two limits apply, and
which one is used is decided by the request's `Content-Type`:

| `Content-Type` | Limit | Default |
| --- | --- | --- |
| `application/json` | `SPLAT_API_MAX_JSON_BYTES` | 32 MiB |
| anything else (including uploads) | `SPLAT_API_MAX_REQUEST_BYTES` | 32 GiB |

The smaller JSON cap exists because a job request is a few kilobytes: there is no
reason to let a caller stream gigabytes into the pydantic parser. The type is
matched on the media type only, so `application/json; charset=utf-8` still gets the
JSON limit.

This is a pre-filter only. Upload handlers additionally cap on bytes actually
received against `SPLAT_API_MAX_UPLOAD_BYTES` (default 16 GiB), because chunked
uploads declare no length at all.

---

## Endpoints

### Operational

| Method | Path | Scope | Success |
| --- | --- | --- | --- |
| `GET` | `/healthz` | none | `200` |
| `GET` | `/readyz` | none | `200` healthy / `503` degraded |
| `GET` | `/metrics` | none | `200 text/plain` (only when `SPLAT_API_METRICS_ENABLED=true`, the default) |
| `GET` | `/docs`, `/openapi.json` | none | `200` (only when `SPLAT_API_DOCS_ENABLED=true`, the default) |

`GET /healthz` is liveness only and never touches the GPU:

```json
{ "status": "ok", "version": "1.0.0", "checks": { "process": "ok" } }
```

`GET /readyz` reports storage and database health. It is `ok` only when
`database == "ok"`, `data_root_writable` is true, and `disk_free_ratio > 0.02`;
otherwise `degraded` with HTTP `503`.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "checks": {
    "database": "ok",
    "data_root_writable": true,
    "disk_free_bytes": 8241934663680,
    "disk_free_ratio": 0.4137,
    "gpu_count": 8,
    "queue_depth": 2,
    "artifixer_checkpoint_configured": true,
    "jobs_by_state": { "succeeded": 91, "queued": 2, "running": 1 }
  }
}
```

`GET /metrics` exposes Prometheus text: `splat_api_jobs_total{state="…"}` for
`queued|running|succeeded|failed|cancelled`, `splat_api_queue_depth`, and
`splat_api_workers`.

---

### `GET /v1/capabilities`

Scope: `read`. What this deployment can actually run, and the limits your client
should respect.

`modes` is `["reconstruct"]` unless an ArtiFixer checkpoint is configured *and
present on disk*, in which case `artifixer3d` and `artifixer3d_plus` are added.

`200 OK`:

```json
{
  "service_version": "1.0.0",
  "modes": ["reconstruct", "artifixer3d", "artifixer3d_plus"],
  "artifixer_checkpoint_configured": true,
  "artifixer_model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
  "gpu_count": 8,
  "max_concurrent_jobs": 8,
  "queue_capacity": 256,
  "limits": {
    "max_upload_bytes": 17179869184,
    "max_images": 4000,
    "max_archive_members": 20000,
    "max_uncompressed_bytes": 68719476736,
    "max_trajectory_frames": 2000,
    "reconstruction_steps_max": 100000,
    "artifixer3d_steps_max": 200000,
    "rate_limit_per_minute": 120
  }
}
```

Errors: `401`, `403`, `429`.

---

### `POST /v1/scenes`

Scope: `write`. Upload a COLMAP scene as a ZIP archive.

Query parameters:

| Name | Type | Default | Constraint |
| --- | --- | --- | --- |
| `label` | string | – | `maxLength` 128; must not be blank if present |
| `dedupe` | boolean | `true` | When true, an archive whose bytes hash to an existing scene returns that scene instead of re-ingesting |

Body — one of:

| `Content-Type` | Body |
| --- | --- |
| `application/zip`, `application/x-zip-compressed`, `application/octet-stream` | The raw ZIP bytes. Fast path: no multipart framing, no spool file. |
| `multipart/form-data` | A part named `file` containing the ZIP. At most 1 file and 8 fields. |

Any other content type is `400 bad_request`, and the message names every accepted
value: `application/zip`, `application/x-zip-compressed`, `application/octet-stream`
(raw body), or `multipart/form-data` with a `file` part.

#### Input contract

The archive is extracted by allowlist: **only** `images/<name>` and
`sparse/0/{cameras.bin,images.bin,points3D.bin}` are written, everything else is
silently ignored (the count appears in server logs as `ignored_members`).

| Rule | Detail |
| --- | --- |
| Anchor | The directory containing `sparse/0/cameras.bin` becomes the scene root, so `truck/images/...` works. Two such directories is `422` — one scene per request. |
| Image directory | The first present of `images`, `images_2`, `images_4`, `images_8`. |
| Image extensions | `.jpg .jpeg .png .bmp .tif .tiff .webp`; other files are ignored. |
| Nested image folders | Flattened to basenames; a duplicate basename is `422`. |
| Minimum images | 2. |
| Required sparse files | All three; a missing one is `422`. |
| Archive format | ZIP only. TAR is deliberately unsupported. Encrypted archives and symlink members are `400`. |
| Caps | `max_archive_members`, `max_uncompressed_bytes`, `max_compression_ratio`, `max_images` — all `413`. |

The extracted scene is then validated against every pipeline precondition, all of
which are `422` on failure: `points3D.bin` must be non-empty; COLMAP image
basenames must be unique and have a supported extension; every camera id
referenced by `images.bin` must exist in `cameras.bin`; camera models must be one
of `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`; every image
referenced by COLMAP must be present in `images/`; images sharing a camera must
have identical dimensions; the image-to-camera scale factor must be uniform in x
and y; and all used cameras must share one intrinsic calibration.

#### Known input limitation: non-uniform camera/image scale ratio

A scene whose `cameras.bin` resolution and shipped `images/` resolution differ by a
ratio that is not the *same* in x and y is rejected with `422`, even when the
difference is only rounding.

This bites real published datasets. The Tanks-and-Temples `truck` scene from the
3DGS release ships `cameras.bin` at 1957x1091 alongside 979x546 images. The per-axis
ratios are `979/1957 = 0.500255` and `546/1091 = 0.500458`, which differ by more
than a rounding tolerance, so the scene is refused.

This is a **pre-existing pipeline constraint, not an API restriction**:
`prepare_colmap_artifixer_inputs.scale_colmap_scene_to_images` asserts
`np.isclose(sx, sy)` on the same two ratios and would abort the `prepare` stage
minutes into a GPU job. The API duplicates the check so the rejection is a clean
`422` in the request path instead.

Remedies:

- ship the full-resolution images that match the `cameras.bin` dimensions, or
- re-run COLMAP at the resolution you actually intend to ship, so the camera and
  the images agree.

The Deep Blending `playroom` and `drjohnson` scenes match exactly (no rescale at
all) and work unmodified.

`201 Created` (or `200 OK` with `X-Scene-Deduplicated: true` on a dedupe hit):

```json
{
  "scene_id": "scene_5b1d9c4e2a7f0361d8ab5c92",
  "created_at": "2026-07-25T14:02:11.418Z",
  "label": "truck",
  "source": "upload",
  "image_count": 251,
  "camera_count": 1,
  "camera_models": ["OPENCV"],
  "point_count": 138204,
  "colmap_width": 1957,
  "colmap_height": 1091,
  "size_bytes": 641203712,
  "image_names": ["000001.jpg", "000002.jpg", "000003.jpg"]
}
```

Errors: `400`, `401`, `403`, `413`, `422`, `429`.

---

### `POST /v1/scenes/register`

Scope: `admin`. Zero-copy ingestion of a scene already on server storage — nothing
is copied, which makes it the fastest way to onboard very large captures. Admin-only
because the caller names a filesystem path.

Registration is confined to a **single hardcoded import root**, `<data_root>/import`.
There is no environment variable to move it, add a second root, or point it
elsewhere: the path is derived from `data_root` in the handler. The operator creates
that directory (the service does not create it at startup) and places scenes under
it; if it does not exist, every registration attempt is `422` with
`"No import root configured."`.

Body (`extra` fields forbidden):

| Field | Type | Required | Constraint |
| --- | --- | --- | --- |
| `path` | string | yes | 1–4096 chars; interpreted relative to `<data_root>/import`, leading `/` stripped, containment enforced |
| `label` | string \| null | no | `maxLength` 128 |

```bash
curl -sS -X POST "$SPLAT_API/v1/scenes/register" \
  -H "Authorization: Bearer $SPLAT_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"path": "captures/warehouse-a", "label": "warehouse-a"}'
```

`201 Created`: a `SceneInfo` with `"source": "registered"` and `image_names`
populated. Registered scenes are never deleted from disk by
`DELETE /v1/scenes/{scene_id}`.

Errors: `400` (blank path, escaping path), `401`, `403`, `404` (no such directory
under the import root), `422` (`<data_root>/import` does not exist, or the scene
fails COLMAP validation), `429`.

---

### `GET /v1/scenes`

Scope: `read`. Newest first.

| Name | Type | Default | Constraint |
| --- | --- | --- | --- |
| `limit` | integer | 50 | 1–200 |

`200 OK`: an array of `SceneInfo`. `image_names` is `null` in the list projection;
fetch a single scene to get it.

Errors: `401`, `403`, `422` (`limit` out of range), `429`.

---

### `GET /v1/scenes/{scene_id}`

Scope: `read`. `scene_id` must match `^[a-z0-9][a-z0-9_-]{0,62}$`.

`200 OK`: `SceneInfo` including `image_names` — this is the authoritative list for
building `selected_image_names`.

Errors: `400` (malformed id), `401`, `403`, `404`, `429`.

---

### `DELETE /v1/scenes/{scene_id}`

Scope: `admin`. Removes the record, and for `source == "upload"` also the scene
directory. Registered scenes keep their bytes: they belong to whoever placed them
under the import root.

`204 No Content` (empty body).

Errors: `400`, `401`, `403`, `404`, `409` (the scene has queued or running jobs),
`429`.

---

### `POST /v1/jobs`

Scope: `write`. Queue a COLMAP-to-splat pipeline run. Returns immediately; poll
for progress.

Body (`extra` fields forbidden):

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `scene_id` | string | required | An id from `POST /v1/scenes` |
| `mode` | `reconstruct` \| `artifixer3d` \| `artifixer3d_plus` | `reconstruct` | Non-`reconstruct` needs a configured ArtiFixer checkpoint, else `503` |
| `reconstruction_steps` | integer \| null | server default (`SPLAT_API_RECONSTRUCTION_STEPS_DEFAULT`, 10000) | `100 … 1000000` sanity ceiling in the schema; the operative limit is `SPLAT_API_RECONSTRUCTION_STEPS_MAX` (default 100000). Either limit rejects with `422` |
| `artifixer3d_steps` | integer \| null | server default (`SPLAT_API_ARTIFIXER3D_STEPS_DEFAULT`, 30000) | `100 … 1000000` sanity ceiling in the schema; the operative limit is `SPLAT_API_ARTIFIXER3D_STEPS_MAX` (default 200000). Either limit rejects with `422` |
| `inference_steps` | integer | `4` | `1 … 50`; ArtiFixer denoising steps |
| `selected_image_names` | array of string \| null | `null` | Non-empty, no duplicates; each 1–128 chars matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` with a supported image extension and no path separators; each must exist in the scene. **Omitting it does not mean "train on everything"** — see [Automatic validation holdout](#automatic-validation-holdout) |
| `metric_scale` | number \| null | `null` | `> 0`, `<= 1e6`; supplying it skips MoGe metric alignment |
| `trajectory` | `Trajectory` \| null | `null` | See below; forbidden when `mode == "reconstruct"` |
| `export_ply` | boolean | `true` | When false the `export` stage is dropped from the sequence and no `splat.ply` is produced |
| `client_token` | string \| null | `null` | 8–128 chars matching `^[A-Za-z0-9_.:-]+$`; idempotency key |

Step counts have two ceilings and both answer with HTTP `422`: the pydantic bound
of 1,000,000 (as `validation_error`, with `details.errors`) and the deployment's
`SPLAT_API_*_STEPS_MAX` (as `unprocessable_input`, message
`"<field>=<value> exceeds the configured maximum of <max> for this deployment"`).
Since `*_STEPS_MAX` is normally far below 1,000,000, the deployment limit is the one
you will hit. Read `limits.reconstruction_steps_max` and
`limits.artifixer3d_steps_max` from `GET /v1/capabilities` rather than guessing.

Cross-field rules:

- `mode != "reconstruct"` requires `selected_image_names` **or** `trajectory`
  (`validation_error`, 422).
- `mode == "reconstruct"` with a `trajectory` is rejected (`validation_error`,
  422): nothing would correct it.
- For `artifixer3d` modes with `selected_image_names` and no `trajectory`, the
  selection must be a *strict* subset of the scene's images
  (`unprocessable_input`, 422) — ArtiFixer3D needs at least one non-anchor frame
  to generate. See [PIPELINE.md](PIPELINE.md#why-artifixer3d-needs-a-strict-subset-or-a-trajectory).
- `trajectory.frames` longer than `SPLAT_API_MAX_TRAJECTORY_FRAMES` is `413`.

#### Automatic validation holdout

**When you omit `selected_image_names`, the API does not train on every image.** It
holds out every 8th image (indices 0, 8, 16, … of the scene's `image_names` order)
and passes the remaining ~87.5% as the 3DGRUT training anchors. The job response
tells you exactly what happened:

| Response field | Meaning |
| --- | --- |
| `request.validation_holdout_auto` | `true` when the API chose the training views, `false` when you named them |
| `request.selected_image_count` | How many images actually became training anchors |

So a 225-image scene submitted with no `selected_image_names` reports
`validation_holdout_auto: true` and `selected_image_count: 196` (225 minus the 29
held-out views).

**Why this is not optional.** 3DGRUT always constructs a *validation*
`ColmapDataset` alongside the training one
(`thirdparty/3DGRUT-ArtiFixer/threedgrut/datasets/__init__.py:46-56`), and when a
selected-indices file is present the validation split is defined as
`setdiff1d(all_indices, selected_indices)`
(`threedgrut/datasets/dataset_colmap.py:101-104`). Selecting every image therefore
leaves the validation split *empty*, and training dies during dataset construction
with:

```text
IndexError: min(): Expected reduction dim 0 to have non-zero size.
```

raised from `compute_spatial_extents`. This was verified empirically against the
real engine, not inferred. Because
`data_processing/prepare_colmap_artifixer_inputs` always writes and passes a
selected-indices file, there is no way to ask the public CLI for "no split at all" —
holding views back is the only route through it.

The interval of 8 is not arbitrary: it matches the engine's own
`test_split_interval: 8` (`thirdparty/3DGRUT-ArtiFixer/configs/dataset/colmap.yaml:3`),
so the holdout is the split 3DGRUT would have chosen for itself.

Consequences worth knowing:

- The delivered splat is trained on ~87.5% of your views, not 100%. Naming
  `selected_image_names` yourself does not escape this — it only moves the decision
  to you, and naming *every* image is what triggers the crash above. Always leave at
  least one image out.
- The held-out views are still rendered. The render pass builds its dataset with
  `split="test"`, which upstream comments as "test mode to ensure we render all the
  images regardless of the selected indices"
  (`threedgrut/datasets/__init__.py:87-96`), so a 225-image scene still yields 225
  rendered frames. The holdout affects which views are *optimization targets*, not
  which are rendered.
- The reconstruction cache key is computed from the resolved training view set, so
  an auto-holdout job and an explicit job that names the same views share a cached
  base reconstruction.
- For `artifixer3d` modes the schema already requires `selected_image_names` or a
  `trajectory`, so the automatic holdout applies to `mode=reconstruct` and to
  `artifixer3d` runs driven by a `trajectory` alone.

#### `Trajectory` object

Top-level intrinsics apply to the whole path; the renderer asserts a single
resolution, so `w`/`h` are top-level only.

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `camera_model` | `"OPENCV"` | `"OPENCV"` | Only value accepted |
| `w`, `h` | integer | required | `> 0`, `<= 8192` |
| `fl_x`, `fl_y` | number | required | `> 0` |
| `cx`, `cy` | number | required | – |
| `k1`, `k2`, `p1`, `p2` | number | `0.0` | – |
| `frames` | array of `TrajectoryFrame` | required | `minItems` 1 |

`TrajectoryFrame`:

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `transform_matrix` | array of arrays of number | required | 3×4 or 4×4 row-major OpenGL/NeRFStudio camera-to-world; all entries finite numbers (no booleans); a 4×4 must have bottom row `[0,0,0,1]` |
| `fl_x`, `fl_y` | number \| null | `null` | Per-frame override; must be positive |
| `cx`, `cy`, `k1`, `k2`, `p1`, `p2` | number \| null | `null` | Per-frame override |

`file_path` is deliberately absent: frames carrying one look like source/context
images rather than render targets, and the upstream trajectory validator rejects
them.

Example:

```bash
curl -sS -X POST "$SPLAT_API/v1/jobs" \
  -H "Authorization: Bearer $SPLAT_KEY" -H "Content-Type: application/json" -d '{
  "scene_id": "scene_5b1d9c4e2a7f0361d8ab5c92",
  "mode": "artifixer3d_plus",
  "reconstruction_steps": 10000,
  "artifixer3d_steps": 30000,
  "inference_steps": 4,
  "selected_image_names": ["000001.jpg", "000013.jpg", "000025.jpg"],
  "client_token": "warehouse-a-plus-0007"
}'
```

`202 Accepted`, with `Location: /v1/jobs/{job_id}`:

```json
{
  "job_id": "job_2c9f0a71e5b3d846fa10c7e2",
  "scene_id": "scene_5b1d9c4e2a7f0361d8ab5c92",
  "state": "queued",
  "mode": "artifixer3d_plus",
  "progress": 0.0,
  "created_at": "2026-07-25T14:07:44.902Z",
  "started_at": null,
  "finished_at": null,
  "queue_position": 3,
  "gpu_index": null,
  "stages": [
    { "name": "prepare", "state": "pending", "started_at": null, "finished_at": null,
      "duration_seconds": null, "exit_code": null, "message": null },
    { "name": "artifixer", "state": "pending", "started_at": null, "finished_at": null,
      "duration_seconds": null, "exit_code": null, "message": null },
    { "name": "artifixer3d", "state": "pending", "started_at": null, "finished_at": null,
      "duration_seconds": null, "exit_code": null, "message": null },
    { "name": "artifixer3d_plus", "state": "pending", "started_at": null, "finished_at": null,
      "duration_seconds": null, "exit_code": null, "message": null },
    { "name": "export", "state": "pending", "started_at": null, "finished_at": null,
      "duration_seconds": null, "exit_code": null, "message": null }
  ],
  "artifacts": [],
  "error": null,
  "request": {
    "mode": "artifixer3d_plus",
    "reconstruction_steps": 10000,
    "artifixer3d_steps": 30000,
    "inference_steps": 4,
    "selected_image_count": 3,
    "validation_holdout_auto": false,
    "trajectory_frames": null,
    "metric_scale": null,
    "export_ply": true
  }
}
```

`request` is a filtered projection: only `mode`, `reconstruction_steps`,
`artifixer3d_steps`, `inference_steps`, `selected_image_count`,
`validation_holdout_auto`, `trajectory_frames`, `metric_scale`, and `export_ply`
are echoed. Server-side paths and cache keys stored with the job never leave the
process, and the resolved training view list itself is not echoed — only its count
and whether the API chose it.

Idempotency: re-submitting the same `client_token` returns the original job with
`200 OK` instead of `202`. Reusing a token with a different `scene_id` or `mode`
is `409 conflict`.

Errors: `400` (malformed `scene_id`), `401`, `403`, `404` (unknown scene), `409`,
`413`, `422` (schema violation, step count over the configured maximum, unknown or
non-strict-subset `selected_image_names`), `429`, `503` (mode needs a checkpoint /
queue full / shutting down).

---

### `GET /v1/jobs`

Scope: `read`. Keyset pagination on `created_at`, newest first.

| Name | Type | Default | Constraint |
| --- | --- | --- | --- |
| `state` | string | – | One of `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `scene_id` | string | – | `maxLength` 64, must be a valid id |
| `cursor` | string | – | `maxLength` 64; pass the `created_at` of the last seen job |
| `limit` | integer | 50 | 1–200 |

`200 OK`:

```json
{
  "jobs": [ { "job_id": "job_2c9f0a71e5b3d846fa10c7e2", "state": "running", "…": "…" } ],
  "next_cursor": "2026-07-25T13:11:02.774Z"
}
```

`next_cursor` is non-null only when the page was full (`len(jobs) == limit`).

Errors: `400`, `401`, `403`, `422`, `429`.

---

### `GET /v1/jobs/{job_id}`

Scope: `read`. The polling endpoint.

`200 OK`: a `JobInfo`. Fields worth calling out:

| Field | Meaning |
| --- | --- |
| `state` | `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `progress` | `(succeeded + skipped stages) / total stages`, rounded to 4 dp; `1.0` for a terminal job with no stages |
| `queue_position` | Populated only while `state == "queued"`: `1`-based position in the wait queue, `0` if it is already executing, `null` if unknown |
| `gpu_index` | The CUDA device the job's stages are pinned to, once running |
| `stages[].state` | `pending`, `running`, `succeeded`, `failed`, `skipped`, `cancelled` |
| `stages[].exit_code` | Subprocess exit code once the stage finished |
| `stages[].message` | Short reason, e.g. `Exited with code 1`, `Timed out after 86400s`, or `Interrupted by a service restart; will be retried` |
| `artifacts[]` | See below; empty until the job succeeds |
| `error` | Failure summary with up to 12 lines of the failing stage's log tail; `Cancelled by request` for cancellations |

A succeeded `artifixer3d_plus` job:

```json
{
  "job_id": "job_2c9f0a71e5b3d846fa10c7e2",
  "scene_id": "scene_5b1d9c4e2a7f0361d8ab5c92",
  "state": "succeeded",
  "mode": "artifixer3d_plus",
  "progress": 1.0,
  "created_at": "2026-07-25T14:07:44.902Z",
  "started_at": "2026-07-25T14:07:45.310Z",
  "finished_at": "2026-07-25T19:52:03.117Z",
  "queue_position": null,
  "gpu_index": 3,
  "stages": [
    { "name": "prepare", "state": "succeeded", "started_at": "2026-07-25T14:07:45.311Z",
      "finished_at": "2026-07-25T15:41:12.884Z", "duration_seconds": 5607.573,
      "exit_code": 0, "message": null }
  ],
  "artifacts": [
    {
      "name": "splat.ply",
      "kind": "splat_ply",
      "size_bytes": 512884736,
      "sha256": "6b3f0c1d9e77a2418c5d0be4f1a9d3c8e02b74f5a6d1c9083be27f45a1c6d0e9",
      "download_path": "/v1/jobs/job_2c9f0a71e5b3d846fa10c7e2/artifacts/splat.ply",
      "description": "Gaussian splat in 3DGS-compatible binary PLY format"
    },
    {
      "name": "logs/export.log",
      "kind": "log",
      "size_bytes": 1841,
      "sha256": null,
      "download_path": "/v1/jobs/job_2c9f0a71e5b3d846fa10c7e2/artifacts/logs/export.log",
      "description": "stdout/stderr of the export stage"
    }
  ],
  "error": null,
  "request": { "mode": "artifixer3d_plus", "export_ply": true, "…": "…" }
}
```

Errors: `400`, `401`, `403`, `404`, `429`.

---

### `POST /v1/jobs/{job_id}/cancel`

Scope: `write`. Cancels a queued or running job. A running job's stage is stopped
by signalling its whole process group (`SIGTERM`, then `SIGKILL` after a 20 s
grace period). A queued job is marked terminal immediately.

`200 OK`: the updated `JobInfo`, with `state` becoming `cancelled`, pending stages
`cancelled`, and `error` set to `Cancelled before execution` or
`Cancelled by request`.

Errors: `400`, `401`, `403`, `404`, `409` (already `succeeded`/`failed`/
`cancelled`), `429`.

---

### `GET /v1/jobs/{job_id}/artifacts`

Scope: `read`. The same array as `JobInfo.artifacts`.

`ArtifactInfo` fields: `name`, `kind`, `size_bytes`, `sha256`, `download_path`,
`description`. `kind` is one of:

| `kind` | Typical `name` | Content |
| --- | --- | --- |
| `splat_ply` | `splat.ply` | Gaussian splat, 3DGS-compatible binary PLY |
| `splat_checkpoint` | `splat_checkpoint.pt` | 3DGRUT checkpoint for the delivered splat |
| `frames` | `corrected_frames.zip` | ArtiFixer-corrected PNG frames, stored (uncompressed) so clients can seek |
| `video` | `reconstruction_preview.mp4`, `artifixer3d_preview.mp4` | Turntable renders, when the renderer produced them |
| `metadata` | `manifest.json`, `splat_stats.json` | Machine-readable records of the run; see below |
| `log` | `logs/<stage>.log` | Full stdout/stderr of one stage |

`sha256` is `null` for logs (never hashed) and for any file larger than 8 GiB.

#### `metadata` artifacts

Both are published, downloadable artifacts like any other — fetch them from
`GET /v1/jobs/{job_id}/artifacts/manifest.json` and
`…/artifacts/splat_stats.json`.

`manifest.json` is the full record of the run, written by the scheduler and sorted
by key:

| Key | Content |
| --- | --- |
| `job_id`, `scene_id`, `mode`, `created_at`, `finished_at` | Job identity and timing |
| `request` | The **complete stored** request, not the filtered `JobInfo.request` projection: it includes the resolved `selected_image_names` list, `validation_holdout_auto`, the absolute `scene_root`, and the `reconstruction_cache_key` |
| `metrics` | Harvested numbers, including `stage_seconds` (wall-clock seconds per stage), `stage_outputs` (the allowlisted `key=value` lines each stage printed), `inference_dirs`, and `reconstruction_checkpoint` |
| `stages[]` | Per stage: `name`, `state`, `duration_seconds`, and the fully-quoted `command` that was executed |
| `artifacts[]` | `name`, `kind`, `size_bytes`, `sha256` for every other artifact |

Because the manifest lists the other artifacts, it is written **last** and then
registered as an artifact itself, so `artifacts[]` inside it never contains an entry
for `manifest.json` and it never carries its own digest. The `ArtifactInfo` record
returned by the API *does* carry the manifest's `sha256`.

Note that the manifest deliberately exposes what the HTTP projection withholds
(absolute server paths, the cache key, the full view list). It is written into the
job directory, so treat it as operator-and-`read`-scope-visible detail in the same
category as stage logs.

`splat_stats.json` is written by the `export` stage, which the service invokes with
`--stats-json`. It has exactly four keys and describes the delivered splat:

| Key | Type | Meaning |
| --- | --- | --- |
| `num_gaussians` | integer | Vertex count in the PLY |
| `sh_degree` | integer | Spherical-harmonic degree, derived from the checkpoint's `features_specular` width |
| `global_step` | integer \| null | Training step recorded in the checkpoint; `null` when it recorded none |
| `ply_bytes` | integer | Size of the written `splat.ply` |

For a concrete measured example, see the runtime data point in
[PIPELINE.md](PIPELINE.md#measured-runtime-a-verified-reconstruct-run). This
artifact is absent when `export_ply=false`, since the `export` stage is then
dropped.

Errors: `400`, `401`, `403`, `404`, `429`.

---

### `GET /v1/jobs/{job_id}/artifacts/{artifact_name}`

Scope: `read`. Streams one artifact. `artifact_name` is a path-style parameter, so
nested names such as `logs/prepare.log` work; only names present in the job's
artifact list resolve, and the resolved path is re-checked for containment inside
the job directory.

`200 OK` with:

- `Content-Disposition: attachment; filename="<basename>"`
- `ETag: "<sha256>"` when the artifact was hashed (a strong validator, so clients
  can revalidate cheaply)
- `Content-Type`: `application/zip` for `.zip`, `video/mp4` for `.mp4`,
  `text/plain; charset=utf-8` for `.log`, otherwise `application/octet-stream`

Large files are sent with `sendfile` where the platform supports it, so a
multi-gigabyte splat never passes through Python buffers.

Errors: `400`, `401`, `403`, `404` (unknown name, or the file is no longer on
disk), `429`.

---

### `GET /v1/jobs/{job_id}/logs/{stage}`

Scope: `read`. The tail of one stage's log, for live progress during long stages.
`stage` must be one of the job's stage names.

| Name | Type | Default | Constraint |
| --- | --- | --- | --- |
| `tail_bytes` | integer | 65536 | 1024 – 262144 |

`200 OK`, `text/plain`. The first partial line is discarded when the log is longer
than `tail_bytes`. Undecodable bytes are replaced, never rejected.

Errors: `400`, `401`, `403`, `404` (unknown job, unknown stage, or no log written
yet), `422` (`tail_bytes` out of range), `429`.

---

## Configuration: HTTP surface

Every knob is read once at startup into an immutable settings object; nothing in
the request path reads the environment, so a request can never influence process
configuration. A malformed value fails startup loudly. Blank or whitespace-only
values are treated as unset. Booleans accept `1/true/yes/on` and `0/false/no/off`.

Pipeline, scheduler and filesystem variables are documented in
[PIPELINE.md](PIPELINE.md#configuration-pipeline-and-scheduler).

| Variable | Default | Validation |
| --- | --- | --- |
| `SPLAT_API_KEYS` | `""` | `key_id:secret:scopes` or `key_id:sha256:<hex>:scopes`, `|`-separated. See [Authentication](#authentication). |
| `SPLAT_API_REQUIRE_AUTH` | `true` | Boolean. `true` with no keys configured is a startup error. |
| `SPLAT_API_TRUSTED_HOSTS` | `*` | Comma-separated. Any value other than the single `*` installs host checking. |
| `SPLAT_API_CORS_ORIGINS` | *(empty)* | Comma-separated. CORS middleware is installed only when non-empty: methods `GET, POST, DELETE`; headers `authorization, x-api-key, content-type`; credentials disallowed; preflight cached 600 s. |
| `SPLAT_API_MAX_REQUEST_BYTES` | 34359738368 (32 GiB) | Integer `>= 1024`. Enforced against declared `Content-Length` for every request whose media type is **not** `application/json`. See [Body size pre-filter](#body-size-pre-filter). |
| `SPLAT_API_RATE_LIMIT_PER_MINUTE` | `120` | Integer `>= 0`; `0` disables rate limiting. |
| `SPLAT_API_RATE_LIMIT_BURST` | `40` | Integer `>= 1`. |
| `SPLAT_API_MAX_UPLOAD_BYTES` | 17179869184 (16 GiB) | Integer `>= 1024`. Enforced on bytes actually received. |
| `SPLAT_API_MAX_ARCHIVE_MEMBERS` | `20000` | Integer `>= 4`. |
| `SPLAT_API_MAX_UNCOMPRESSED_BYTES` | 68719476736 (64 GiB) | Integer `>= 1024`. Also the running extraction budget. |
| `SPLAT_API_MAX_COMPRESSION_RATIO` | `200.0` | Float `>= 1.0`. Applied only when total compressed size exceeds 4096 bytes. |
| `SPLAT_API_MAX_IMAGES` | `4000` | Integer `>= 2`. Caps both archive images and `images.bin` entries. |
| `SPLAT_API_MAX_TRAJECTORY_FRAMES` | `2000` | Integer `>= 1`. |
| `SPLAT_API_MAX_JSON_BYTES` | 33554432 (32 MiB) | Integer `>= 1024`. The smaller `Content-Length` cap applied when the media type is `application/json`, so a JSON body cannot be streamed at upload scale into the pydantic parser. See [Body size pre-filter](#body-size-pre-filter). |
| `SPLAT_API_LOG_LEVEL` | `INFO` | Upper-cased; unknown names fall back to `INFO`. Logs are one JSON object per line with `request_id` promoted. |
| `SPLAT_API_METRICS_ENABLED` | `true` | Boolean. Controls `/metrics`. |
| `SPLAT_API_DOCS_ENABLED` | `true` | Boolean. Controls `/docs` and `/openapi.json`; ReDoc is always off. |

Read directly by the entry point (not through the settings object):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPLAT_API_HOST` | `0.0.0.0` | uvicorn bind address |
| `SPLAT_API_PORT` | `8000` | uvicorn port |
| `SPLAT_API_FORWARDED_ALLOW_IPS` | `127.0.0.1` | Peers whose `X-Forwarded-*` headers are trusted |

The service runs a single API worker process on purpose: the scheduler, its queue,
and GPU assignment are process-local state, and the bottleneck is the GPU, not the
event loop. Responses larger than 1024 bytes are gzipped when the client accepts
it.
