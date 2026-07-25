<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# splat_api security model

The service accepts untrusted archives and untrusted JSON, then runs GPU
subprocesses that read and write files on a shared machine. This document states
what is defended, how, and what is left to the operator.

Contents: [Threat model](#threat-model) · [Authentication](#authentication-and-authorization) ·
[Archive hardening](#archive-hardening) · [Path containment](#path-containment) ·
[Subprocess safety](#subprocess-safety) · [Not exposed](#what-is-deliberately-not-exposed) ·
[Error hygiene](#error-message-hygiene) · [Rate limiting](#rate-limiting-and-resource-caps) ·
[Container](#container-hardening-expectations) ·
[Residual risks](#residual-risks-and-operator-responsibilities)

---

## Threat model

**Trusted:** the operator, the repo checkout, the ArtiFixer checkpoint and base
model weights, anything an admin places under `<data_root>/import`, and the network
path between a TLS terminator and the service.

**Untrusted:** the uploaded ZIP and every byte in it; all request bodies, query
strings and headers, including `X-Request-Id`; and every `key=value` line the
pipeline's own CLIs print (they are parsed but only allowlisted keys are kept).

**Assets:** the checkpoint and any HF token in the environment; other tenants'
scenes, jobs and artifacts; the filesystem outside `<data_root>`; and the GPUs
(availability).

**Adversary:** a caller who holds a `read`+`write` API key, or, when
`SPLAT_API_REQUIRE_AUTH=false`, anyone who can reach the port. Goals defended
against: arbitrary file write via archive extraction, arbitrary file read via path
or artifact-name manipulation, arbitrary command or config injection into the
pipeline, credential extraction from responses or logs, credential brute-force, and
resource exhaustion (disk, memory, GPU, queue).

---

## Authentication and authorization

- **API keys, hashed at rest.** `SPLAT_API_KEYS` is parsed once at startup; only
  the SHA-256 digest of each secret is retained, so a heap dump or a log of the
  settings object cannot yield a usable credential. Deployments can avoid live
  credentials in the environment entirely by supplying
  `key_id:sha256:<64 hex>:scopes`.
- **Plain SHA-256, deliberately.** API keys are high-entropy random strings, not
  user-chosen passwords, so a slow KDF would buy nothing and add per-request
  latency. A minimum secret length of 24 characters is enforced at parse time to
  keep that assumption true. Duplicate key ids and duplicate secrets are startup
  errors; key ids are constrained to `^[A-Za-z0-9_.-]{1,64}$` because they appear
  in log records.
- **Constant-time comparison.** `hmac.compare_digest` against every configured
  key, and the loop does not break on a match, so neither the credential value nor
  the number of configured keys is observable through response timing.
- **Two header forms.** `Authorization: Bearer <key>` or `X-API-Key: <key>`. A
  present-but-non-`Bearer` `Authorization` header is rejected rather than ignored,
  so a client cannot accidentally fall through to unauthenticated handling.
- **Scopes `read` / `write` / `admin`, checked per endpoint.** `admin` is the one
  scope with an implication: `_expand_scopes` turns any spec containing `admin` into
  `read+write+admin` at parse time, because an admin key that cannot read the
  resources it manages is never useful. **An `admin` key therefore does have full
  read and write access**, and that is intentional. There is no other hierarchy —
  `read` and `write` do not imply each other, so a `write`-only key gets `403` from a
  `read` endpoint. Scope-to-endpoint mapping is in [API.md](API.md#scopes).
- **`admin` gates path-naming operations.** `POST /v1/scenes/register` lets a
  caller name a server-side directory, so it is an authorization decision, not a
  convenience; `DELETE /v1/scenes/{id}` deletes bytes. Both require `admin`.
- **Auth-disabled mode is still not admin.** With
  `SPLAT_API_REQUIRE_AUTH=false`, an uncredentialed caller becomes principal
  `anonymous` with `read`+`write` only. A credential that *is* offered is still
  validated and a wrong one is rejected, rather than silently downgrading to
  anonymous.
- **Startup refuses the unsafe default.** `REQUIRE_AUTH=true` with no keys
  configured is a `ConfigError`, so the service cannot come up accidentally open.
- **The credential never lands in a bucket key.** The rate limiter buckets on
  `sha256(credential)[:32]`, because bucket keys end up in memory dumps and must
  not be replayable credentials.
- **Job attribution.** Each job records the `api_key_id` that created it, for audit;
  the id, never the secret.

---

## Archive hardening

Only ZIP is accepted. TAR is deliberately unsupported: tar members can carry device
nodes, hard links and setuid bits, and `tarfile` does not filter those by default
on every supported Python.

Extraction is an **allowlist, not a denylist**: the extractor writes only
`images/<basename>` and `sparse/0/{cameras.bin,images.bin,points3D.bin}` and
silently skips everything else. That removes arbitrary-file-write as a class of bug
rather than trying to enumerate dangerous names. The full plan is computed from the
central directory before a single byte is written, so structurally hostile archives
are rejected without paying for decompression.

| Control | Behaviour | Failure |
| --- | --- | --- |
| Allowlist extraction | Only the four path shapes above are written; image files must have an extension in `.jpg .jpeg .png .bmp .tif .tiff .webp`; the count of ignored members is logged | – |
| Zip-slip / traversal | Member names are NFC-normalized, backslashes treated as separators (Windows producers), then rejected for absolute paths, `X:` drive prefixes, any `..` segment, NUL bytes, empty names, and names that reduce to nothing | `400 bad_request` |
| Symlink members | Unix-created members whose mode bits say `S_IFLNK` are rejected outright, before extraction | `400 bad_request` |
| Encrypted archives | Any member with the encryption flag bit set | `400 bad_request` |
| Member cap | `len(infolist()) > SPLAT_API_MAX_ARCHIVE_MEMBERS` (default 20000) | `413 payload_too_large` |
| Uncompressed cap | Declared total `> SPLAT_API_MAX_UNCOMPRESSED_BYTES` (default 64 GiB) | `413 payload_too_large` |
| Compression-ratio cap | `uncompressed / compressed > SPLAT_API_MAX_COMPRESSION_RATIO` (default 200×), evaluated only once the compressed payload exceeds 4096 bytes so the ratio is not noise | `413 payload_too_large` |
| Image cap | More than `SPLAT_API_MAX_IMAGES` (default 4000) distinct image basenames; `images.bin` is capped identically | `413` / `422` |
| Duplicate basenames | Nested image directories are flattened to basenames because COLMAP matches on basenames; a collision is rejected rather than resolved by overwrite. Two copies of the same `sparse/0/*.bin` is also rejected | `422 unprocessable_input` |
| Declared sizes not trusted | Per-member writing caps on bytes *actually produced by the decompressor* against a running budget, and deletes the partial file when a member exceeds it | `413 payload_too_large` |
| Multiple COLMAP models | More than one directory containing `sparse/0/cameras.bin` — guessing which scene was meant would be worse than an explicit error | `422 unprocessable_input` |
| Upload size | Bytes actually received `> SPLAT_API_MAX_UPLOAD_BYTES` (default 16 GiB); enforced against received bytes, not `Content-Length`, because chunked uploads declare nothing. A separate middleware pre-filter also rejects an oversized declared `Content-Length` — against `SPLAT_API_MAX_JSON_BYTES` (default 32 MiB) when the media type is `application/json`, against `SPLAT_API_MAX_REQUEST_BYTES` (default 32 GiB) otherwise | `413 payload_too_large` |
| Empty upload | Zero bytes received | `400 bad_request` |
| Upload content type | Anything other than `application/zip`, `application/x-zip-compressed`, `application/octet-stream` or `multipart/form-data` (with a `file` part) is refused before any body is read | `400 bad_request` |
| Bounded model parsing | The sparse-model reader refuses files over 4 GiB, bounds-checks every read, caps declared camera counts at 100000, caps image-name length at 4096 bytes, and rejects non-UTF-8 names — a truncated or hostile `.bin` produces `422`, never an unbounded allocation or a crash | `422 unprocessable_input` |

Extraction runs in a small thread pool capped at `min(8, cpu_count)` workers, each
holding its own `ZipFile` handle, so extraction cannot monopolize a shared box.
Extracted files are `0640`; staged uploads are written under
`<data_root>/uploads/<scene_id>/` and moved into place only after validation
passes. Any failure removes the staging tree.

The service also validates the extracted COLMAP model against every pipeline
precondition before accepting the scene. That is a robustness measure rather than a
security boundary, but it means a malformed scene is a `422` in the request path
instead of a subprocess traceback minutes into a GPU job. Rules are listed in
[API.md](API.md#input-contract).

---

## Path containment

Every filesystem path built from caller-supplied data goes through `safe_join`,
which resolves the root, rejects components that are empty, `.`, `..`, absolute, or
contain NUL, then resolves the candidate with `strict=False` and requires the root
to be one of its parents. Resolving non-existent paths matters because job artifact
targets do not exist yet, and resolution is what catches symlinked directories
pointing outside the root.

Containment is applied at:

- scene staging, scene directories, and job directories under `<data_root>`;
- `POST /v1/scenes/register`, where the caller-supplied path is stripped of a
  leading `/`, split into parts, and joined under `<data_root>/import`. That root is
  hardcoded — derived from `data_root` in the handler — with no environment variable
  to relocate it or add a second one, so the reachable filesystem surface of the one
  path-naming endpoint is fixed at deploy time by `SPLAT_API_DATA_ROOT` alone;
- artifact downloads — the name must match a recorded artifact, and the stored
  relative path is re-joined under the job directory and re-checked, so a mutated
  database row still cannot escape;
- stage log reads.

Identifiers are narrower than the filesystem allows: `^[a-z0-9][a-z0-9_-]{0,62}$`,
with no dots (no traversal, no hidden files), no `=` or `,` (Hydra override
separators), and no whitespace — because a scene id is interpolated into 3DGRUT
Hydra overrides as `experiment_name`. Caller-supplied image names must be bare
filenames matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` with an allowed extension,
so a caller cannot smuggle a path into the selected-views file.

`safe_join` distinguishes "resolves outside the root" (fatal) from "is a symlink"
(fine), because the prep stage legitimately symlinks source images and
`points3D.bin`. For the same reason, reported scene sizes skip symlinks, and
intermediate pruning skips directories that are symlinks.

---

## Subprocess safety

- **argv only, never a shell.** Stages are spawned with
  `asyncio.create_subprocess_exec` and an argv tuple. There is no shell, no
  `shell=True`, and no string interpolation of caller data into a command line;
  the human-readable `stage.command` shown in the manifest is produced with
  `shlex.quote` for display only.
- **Caller data reaches the CLIs as files, not flags.** `selected_image_names` and
  `trajectory` are materialized into `inputs/selected_train_images.txt` and
  `inputs/trajectory.json` and passed by path. The only caller-derived *values* on
  a command line are integers (step counts, inference steps) that pydantic bounded,
  and `metric_scale`, which is passed as `repr(float(...))`.
- **Environment allowlist.** The child gets a constructed environment, never the
  API process's own: 14 forwarded names (see
  [PIPELINE.md](PIPELINE.md#scheduling-concurrency-and-caching)) plus a handful the
  service sets itself. Secrets belonging to other services cannot leak into a
  pipeline that writes user-visible logs.
- **Process-group isolation.** `start_new_session=True` puts each stage in its own
  session, so timeout and cancellation are `SIGTERM` to the process group followed
  by `SIGKILL` after a 20 s grace period. Without this, torchrun and dataloader
  children would survive and keep holding a GPU.
- **Per-stage timeout.** `SPLAT_API_STAGE_TIMEOUT_SECONDS` (default 24 h, minimum
  30) bounds every stage individually. On expiry the group is killed and the stage
  is marked failed with `Timed out after <n>s`.
- **stdin is `/dev/null`** and stdout/stderr are piped to a per-stage log file, so
  a stage cannot block on input or write to the service's own streams.
- **Device pinning.** Each worker exports `CUDA_VISIBLE_DEVICES=<its device>`, and
  `MAX_CONCURRENT_JOBS` may not exceed the visible GPU count, so jobs cannot
  contend for a device.
- **Telemetry and tracking disabled** in stages: `WANDB_MODE=disabled`,
  `HF_HUB_DISABLE_TELEMETRY=1`, and `--no-use_wandb` on the ArtiFixer3D CLI.
- **Parsed stage output is allowlisted.** Only nine known `key=value` keys are
  absorbed into job metrics; everything else a stage prints is log text.

---

## What is deliberately not exposed

All request models set `extra="forbid"`: an unexpected field is a client bug or an
attempt to reach a parameter the API does not intend to expose, and silently
ignoring it is how injection bugs survive code review.

| Not exposed | Why |
| --- | --- |
| Hydra / 3DGRUT config overrides | An override string reaches config composition and can set arbitrary keys, including output paths and `resume=` targets. The service builds every override itself and pins the identifiers that get interpolated. The one `resume=` the pipeline can produce comes from `--base_checkpoint`, which is gated on the operator-set `SPLAT_API_ARTIFIXER3D_WARM_START` and resolves to a service-computed checkpoint path — never to anything a caller names. |
| Arbitrary output or working paths | All paths are derived from `<data_root>` plus service-generated ids. A caller cannot choose where anything is written or read. |
| Model ids | `--model_id` / `--text_encoder_model_id` come from `SPLAT_API_ARTIFIXER_MODEL_ID`, and `--checkpoint_pt` from `SPLAT_API_ARTIFIXER_CHECKPOINT`. A caller-chosen model id is a request to download and execute operator-unvetted weights from the network. |
| Additional `run_inference` / `run_artifixer3d` flags | `--evalset`, `--render_trajectory`, `--neighbor_selection_mode`, `--sink_size`, `--save_frame_outputs_only` and `--phases` are computed from `mode` and whether a trajectory was supplied. Pinning them also keeps the derived run-name reproducible, which is how the service locates prediction frames. |
| The exporter as an endpoint | `splat_api.app.exporter` is a module the scheduler invokes on checkpoints the pipeline just wrote. It is never pointed at caller-supplied bytes. |
| Server-side paths in responses | The stored job request holds the absolute scene root and the reconstruction cache key; `JobInfo.request` is an explicit allowlist projection of nine fields (`mode`, `reconstruction_steps`, `artifixer3d_steps`, `inference_steps`, `selected_image_count`, `validation_holdout_auto`, `trajectory_frames`, `metric_scale`, `export_ply`), so those never leave the process. The resolved `selected_image_names` list is likewise not echoed — only its count and whether the API chose it. The `manifest.json` artifact shares the same projection and additionally redacts deployment paths; see [Error-message hygiene](#error-message-hygiene). |
| Trajectory `file_path` and per-frame `w`/`h` | A frame with `file_path` looks like a source image rather than a render target and is rejected upstream; the renderer asserts one resolution across a path. |
| `POST /v1/scenes/register` for non-admins | It is the one endpoint that accepts a filesystem path. |

---

## Error-message hygiene

- `ApiError.message` is returned to the caller verbatim, so by contract it must
  never embed absolute server paths or credentials; internal detail belongs in the
  log record.
- Unhandled exceptions are never surfaced: the handler logs the traceback and
  returns the literal `"Internal server error"` with a `request_id`, because
  pipeline tracebacks contain absolute server paths.
- Stage failures echo at most 12 lines of the failing stage's log tail into
  `job.error` — actionable, not a transcript.
- **Stage logs require the `write` scope.** They are raw subprocess output and carry
  absolute container paths and pipeline tracebacks, so a read-only monitoring
  credential cannot fetch them: both `GET /v1/jobs/{id}/logs/{stage}` and the
  `logs/*.log` artifact download check for `write`
  (`routes.get_stage_log`, `routes.download_artifact`). Deliverables — the splat, the
  checkpoint, frames, previews, the manifest — stay readable with `read` alone.
- **`manifest.json` is redacted, not exempt.** It is a downloadable `metadata`
  artifact, so it carries the same `public_request` projection the HTTP API uses
  (one shared allowlist in `jobstore.PUBLIC_REQUEST_FIELDS`, so the two cannot
  drift), and every string in its metrics and stage commands passes through
  `Scheduler._redact`, which replaces the data root with `<data_root>` and the repo
  root with `<repo_root>`. The absolute `scene_root`, the `reconstruction_cache_key`
  and the resolved `selected_image_names` never appear. The command lines are kept
  in placeholder form so a run stays auditable and reproducible in shape without
  disclosing where the deployment lives.
- A rejected credential is logged as `key_id: "unknown"` — never the offered value.
- `X-Request-Id` is echoed from the client only when it is alphanumeric and at most
  64 characters, because it lands in log records; otherwise a fresh 16-hex-char id
  is generated. The same id appears in every error envelope, so an operator can
  join a client-visible failure to a server log without the response carrying
  internal detail.
- Logs are one JSON object per line, so they are queryable without a parser and
  message text cannot forge log structure.
- Responses carry `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`,
  `Cache-Control: no-store`, and
  `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'`.
  This is a JSON API, so a strict CSP costs nothing and blocks any injected markup
  from loading resources should a response ever be rendered as HTML.

---

## Rate limiting and resource caps

An in-process token bucket runs before any handler work: default 120 requests per
minute with a burst of 40, keyed by credential digest when one is present and by
client IP otherwise, so one noisy tenant cannot exhaust another's allowance.
`/healthz` and `/readyz` are exempt. Setting
`SPLAT_API_RATE_LIMIT_PER_MINUTE=0` disables it. See
[API.md](API.md#rate-limiting) for the client-visible contract.

Two properties matter for security rather than fairness:

- It bounds API-key brute-force against the cheap endpoints.
- Bucket storage is itself bounded. Keys can be attacker-controlled (IPs), so once
  the map exceeds 20000 entries, everything not touched in the last hour is evicted
  wholesale, rather than growing without limit.

Rate limiting is *not* the admission control for GPU work — the bounded job queue
is (`SPLAT_API_QUEUE_CAPACITY`, default 256, then `503`), with one worker per GPU.
Other exhaustion bounds: upload and archive caps above; `MAX_TRAJECTORY_FRAMES`
(2000); the 400-line in-memory log tail per stage; a 256 KiB ceiling on
`tail_bytes` for log reads; artifacts over 8 GiB are not hashed; and `/readyz`
reports `degraded` once free disk falls below 2%, which is the signal to stop
routing work to the node.

The limiter is per process, which is the correct granularity here: the service runs
a single API process by design, because the scheduler, its queue and GPU assignment
are process-local state.

---

## Container hardening expectations

The service does not ship a container image; these are the expectations the code is
written against.

| Expectation | Why it holds |
| --- | --- |
| **Non-root** | Nothing needs privilege. The service creates its own tree under `<data_root>` with mode `0750` and writes files `0640`. Run as a dedicated uid that owns `<data_root>` and can read the repo and the checkpoint. |
| **Read-only root filesystem** | All writes are confined to `<data_root>`. Mount that read-write, plus writable paths for whatever caches the forwarded env vars point at (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `XDG_CACHE_HOME`) and a temp dir if CUDA/Triton needs one. |
| **Dropped capabilities** | `cap_drop: ALL`, `no-new-privileges`. Nothing binds a privileged port (default 8000) or manipulates devices beyond the GPUs the runtime injects. |
| **GPU access scoped** | Expose only the devices this instance should use, and set `SPLAT_API_CUDA_DEVICES` to match — auto-detection otherwise probes `/dev/nvidia0..15`. |
| **Process reaping** | Stages run in their own sessions and are group-signalled; make sure PID 1 reaps (init/tini or the runtime's own), or killed stage trees can accumulate as zombies. |
| **Egress restricted** | See below — weight downloads are the only outbound need. |
| **`/docs` off in production** if the port is reachable beyond the trusted network: the OpenAPI schema and Swagger UI need no credential (`SPLAT_API_DOCS_ENABLED=false`). `/metrics` requires the `read` scope, so give the scraper a read-only key; it can still be disabled entirely with `SPLAT_API_METRICS_ENABLED=false`. |
| **Repo mounted read-only** | Stages run with cwd at `SPLAT_API_REPO_ROOT` and `PYTHONPATH` pointing there. Nothing needs to write to it. |

---

## Residual risks and operator responsibilities

0. **Post-review hardening.** An adversarial review of this service produced 18
   verified findings, all fixed; the ones that change the security posture are worth
   naming because the controls above depend on them:
   - Stage output is drained in fixed-size chunks, splitting on CR as well as LF. A
     line longer than the 64 KiB stream limit used to kill the reader, wedge the
     child on a full pipe, and make the stage unkillable — a denial of service any
     caller could trigger by submitting work whose CLI prints a progress bar.
   - The rate limiter charges the source address on every request, not only the
     offered credential. Bucketing on the credential alone meant a caller rotating
     credential values got a fresh allowance each time, leaving key guessing
     unthrottled. Bucket count is now bounded by LRU eviction rather than by age.
   - `multipart/form-data` uploads must declare `Content-Length`, checked before
     Starlette's parser spools parts to the system temp directory (which enforces no
     size cap of its own and lies outside `<data_root>`).
   - `trajectory.frames` and `selected_image_names` are bounded at parse time, so a
     hostile body cannot spend seconds of event-loop time and gigabytes of RSS
     before the configured limit rejects it.
   - Corrupt archive data, unsupported ZIP compression methods, and undecodable
     image bytes are 400/422 rather than 500-with-traceback. Only stored and
     deflated members are accepted.
   - Names read out of `images.bin` are charset-validated like caller-supplied ones,
     because they are written into the newline-delimited selected-views file the
     prepare CLI parses.
   - Paths scraped from stage stdout are rejected unless they resolve inside the
     job's own directory.
   - `/metrics` now requires the `read` scope; `/readyz` and `/metrics` do their
     filesystem and database work in a thread rather than on the event loop.
   - Cancellation is a read-modify-write under the store lock, so it cannot drop
     artifacts a worker published concurrently; a failed queue admission deletes its
     job row instead of leaving an unrunnable `queued` phantom; and deduplicated or
     rejected uploads no longer leak their staging directory.

1. **TLS termination is out of scope.** The service speaks plain HTTP. Terminate
   TLS in front of it, set `SPLAT_API_TRUSTED_HOSTS` to the real hostnames (the
   default `*` disables host checking), and set
   `SPLAT_API_FORWARDED_ALLOW_IPS` to your proxy's address — otherwise a client can
   forge `X-Forwarded-For` and, with it, the rate-limit bucket for unauthenticated
   requests. Bearer credentials over plain HTTP are recoverable from the wire.
2. **The exporter's `torch.load` fallback is trusted-internal-only.** 3DGRUT stores
   its OmegaConf config inside the checkpoint, so `weights_only=True` cannot
   deserialize it; the exporter tries safe mode first and falls back to a full
   unpickle. That is acceptable *only* because it is invoked exclusively by the
   scheduler on checkpoints the pipeline just wrote inside the job directory. Do
   not repurpose the module for caller-supplied `.pt` files, and treat
   `<data_root>/jobs` and the reconstruction cache as integrity-sensitive: write
   access there is equivalent to code execution as the service user.
3. **Weight downloads need egress.** MoGe (metric scale), the caption model, and
   the ArtiFixer base model resolve through Hugging Face on first use, and
   `HF_TOKEN` is forwarded into stages when set. Either allow egress to the
   registry or pre-populate `HF_HOME`/`HUGGINGFACE_HUB_CACHE`/`MOGE_MODEL_PATH` and
   run offline. Whatever those caches contain is executed by the pipeline, so treat
   them as trusted inputs and pin/verify what you seed them with. `mode=reconstruct`
   avoids the `scale` and `caption` phases entirely and so needs the least egress.
4. **`write` scope sees full stage logs.** Log access is gated on `write` and the
   manifest is redacted, but a `write`-scoped caller still receives raw repo CLI
   output, including absolute container paths and whatever configuration those CLIs
   choose to print. The service does not filter subprocess output — it cannot, since
   the value of a log is that it is verbatim. Treat `write` as a trusted,
   operator-adjacent scope rather than an ordinary tenant credential.
5. **No per-tenant isolation.** Scopes are global, not per-object: any `read` key
   can list and download any scene or job, any `write` key can cancel any job, and
   any `admin` key can delete any scene. Deploy one instance per trust domain, or
   put an authorizing gateway in front.
6. **`SPLAT_API_REQUIRE_AUTH=false` is a trusted-network-only setting.** It grants
   every reachable client `read`+`write`, which includes queueing GPU work and
   downloading every artifact on the node.
7. **Key rotation is a restart.** Keys are read once at startup; revoking one means
   updating `SPLAT_API_KEYS` and restarting. Queued and running jobs survive the
   restart and resume.
8. **Disk is not reclaimed automatically.** `KEEP_INTERMEDIATE=true` (the default)
   retains every job's working tree, and the reconstruction cache and registered
   scenes are never pruned by the service. Monitor `/readyz`'s
   `disk_free_ratio` and reap old jobs and cache entries out of band.
9. **Registered scenes are operator-supplied inputs.** `POST /v1/scenes/register`
   reads bytes directly from `<data_root>/import` — they never pass through the ZIP
   hardening path, only through COLMAP validation. Only place trusted captures
   there, and note that deleting such a scene removes the record, not the files.
10. **Availability under load is bounded, not guaranteed.** A caller with `write`
    can fill the queue to `QUEUE_CAPACITY` with long jobs; subsequent submissions
    get `503` until workers drain. There is no per-key job quota.
