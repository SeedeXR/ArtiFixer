# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven configuration.

Every knob is read once at startup into an immutable :class:`Settings`. Nothing
in the request path reads ``os.environ`` so that behaviour is reproducible and
testable, and so that a request can never influence process configuration.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "SPLAT_API_"

# Scope model: read, write, admin. Scopes are listed explicitly per key, with one
# exception: granting `admin` implies read and write (see `_expand_scopes`),
# because an admin key that cannot read the resources it manages is never useful.
VALID_SCOPES = frozenset({"read", "write", "admin"})

KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MIN_API_KEY_LENGTH = 24

GIB = 1024**3


class ConfigError(RuntimeError):
    """Raised for malformed configuration so startup fails loudly."""


@dataclass(frozen=True)
class ApiKey:
    """A single credential.

    Only the SHA-256 digest is retained, so a heap dump or log of the settings
    object cannot leak a usable credential.
    """

    key_id: str
    digest: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return scope in self.scopes


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _env(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{ENV_PREFIX}{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{ENV_PREFIX}{name} must be <= {maximum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = _env(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{ENV_PREFIX}{name} must be >= {minimum}, got {value}")
    return value


def _env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = _env(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def hash_api_key(secret: str) -> str:
    """Return the digest stored for ``secret``.

    Plain SHA-256 is the right primitive here: API keys are high-entropy random
    strings, not user-chosen passwords, so a slow KDF buys nothing while adding
    per-request latency. Minimum length is enforced at parse time to keep that
    assumption true.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _expand_scopes(scopes: frozenset[str]) -> frozenset[str]:
    return frozenset(VALID_SCOPES) if "admin" in scopes else scopes


def parse_api_keys(raw: str) -> tuple[ApiKey, ...]:
    """Parse ``SPLAT_API_KEYS``.

    Format is ``key_id:secret:scope[+scope...]`` entries separated by ``|``.
    The secret may be given as ``sha256:<hex>`` so deployments never have to put
    a live credential in the environment::

        SPLAT_API_KEYS='ci:sk_live_abc...:write|ops:sha256:9f86d0...:admin'
    """
    keys: list[ApiKey] = []
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for entry in raw.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) == 4 and parts[1] == "sha256":
            key_id, digest, scope_spec = parts[0], parts[2].lower(), parts[3]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ConfigError(f"API key {key_id!r} has a malformed sha256 digest")
        elif len(parts) == 3:
            key_id, secret, scope_spec = parts
            if len(secret) < MIN_API_KEY_LENGTH:
                raise ConfigError(
                    f"API key {key_id!r} secret is {len(secret)} chars; "
                    f"at least {MIN_API_KEY_LENGTH} are required"
                )
            digest = hash_api_key(secret)
        else:
            raise ConfigError(
                "API key entries must be 'key_id:secret:scopes' or 'key_id:sha256:<hex>:scopes', "
                f"got {entry!r}"
            )

        if not KEY_ID_PATTERN.match(key_id):
            raise ConfigError(f"API key id {key_id!r} must match {KEY_ID_PATTERN.pattern}")
        scopes = frozenset(scope.strip().lower() for scope in scope_spec.split("+") if scope.strip())
        unknown = scopes - VALID_SCOPES
        if unknown:
            raise ConfigError(f"API key {key_id!r} has unknown scopes: {sorted(unknown)}")
        if not scopes:
            raise ConfigError(f"API key {key_id!r} has no scopes")
        if key_id in seen_ids:
            raise ConfigError(f"Duplicate API key id: {key_id!r}")
        if digest in seen_digests:
            raise ConfigError(f"Duplicate API key secret for id {key_id!r}")
        seen_ids.add(key_id)
        seen_digests.add(digest)
        keys.append(ApiKey(key_id=key_id, digest=digest, scopes=_expand_scopes(scopes)))
    return tuple(keys)


def _default_repo_root() -> Path:
    # splat_api/app/config.py -> splat_api/app -> splat_api -> <repo root>
    return Path(__file__).resolve().parents[2]


def _detect_cuda_devices() -> tuple[int, ...]:
    """Enumerate visible GPUs without importing torch.

    Importing torch at settings time would cost seconds and megabytes in the API
    process, which never runs a model itself. ``CUDA_VISIBLE_DEVICES`` is the
    authority when set; otherwise we count device nodes.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        stripped = visible.strip()
        if not stripped:
            return ()
        devices: list[int] = []
        for item in stripped.split(","):
            item = item.strip()
            if item.isdigit():
                devices.append(int(item))
        return tuple(devices)
    return tuple(index for index in range(16) if Path(f"/dev/nvidia{index}").exists())


@dataclass(frozen=True)
class Settings:
    """Immutable service configuration."""

    # --- filesystem -----------------------------------------------------
    data_root: Path
    repo_root: Path
    python_executable: str

    # --- auth / transport ----------------------------------------------
    api_keys: tuple[ApiKey, ...]
    require_auth: bool
    trusted_hosts: tuple[str, ...]
    cors_origins: tuple[str, ...]
    max_request_bytes: int
    max_json_bytes: int
    rate_limit_per_minute: int
    rate_limit_burst: int

    # --- upload limits --------------------------------------------------
    max_upload_bytes: int
    max_archive_members: int
    max_uncompressed_bytes: int
    max_compression_ratio: float
    max_images: int
    max_trajectory_frames: int

    # --- pipeline -------------------------------------------------------
    artifixer_checkpoint: Path | None
    artifixer_model_id: str
    cuda_devices: tuple[int, ...]
    max_concurrent_jobs: int
    queue_capacity: int
    stage_timeout_seconds: int
    reconstruction_steps_default: int
    reconstruction_steps_max: int
    artifixer3d_steps_default: int
    artifixer3d_steps_max: int
    artifixer3d_warm_start: bool
    keep_intermediate: bool

    # --- observability --------------------------------------------------
    log_level: str
    metrics_enabled: bool
    docs_enabled: bool

    extra_env: dict[str, str] = field(default_factory=dict)

    # -- derived paths ---------------------------------------------------
    @property
    def scenes_dir(self) -> Path:
        return self.data_root / "scenes"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / "uploads"

    @property
    def database_path(self) -> Path:
        return self.data_root / "splat_api.sqlite3"

    @property
    def artifixer_available(self) -> bool:
        return self.artifixer_checkpoint is not None and self.artifixer_checkpoint.is_file()

    def ensure_directories(self) -> None:
        for path in (self.data_root, self.scenes_dir, self.jobs_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o750)

    def find_key(self, digest: str) -> ApiKey | None:
        for key in self.api_keys:
            if key.digest == digest:
                return key
        return None


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment, validating as we go."""
    data_root = Path(_env("DATA_ROOT", "/data/splat-api")).expanduser().resolve()
    repo_root = Path(_env("REPO_ROOT") or _default_repo_root()).expanduser().resolve()
    if not (repo_root / "data_processing").is_dir():
        raise ConfigError(
            f"{ENV_PREFIX}REPO_ROOT={repo_root} does not look like the ArtiFixer repo "
            "(no data_processing/ directory)"
        )

    raw_keys = _env("KEYS", "")
    api_keys = parse_api_keys(raw_keys or "")
    require_auth = _env_bool("REQUIRE_AUTH", True)
    if require_auth and not api_keys:
        raise ConfigError(
            f"No API keys configured. Set {ENV_PREFIX}KEYS='id:secret:scopes' or set "
            f"{ENV_PREFIX}REQUIRE_AUTH=false for a trusted-network deployment."
        )

    checkpoint_raw = _env("ARTIFIXER_CHECKPOINT")
    checkpoint = Path(checkpoint_raw).expanduser().resolve() if checkpoint_raw else None
    if checkpoint is not None and not checkpoint.is_file():
        raise ConfigError(f"{ENV_PREFIX}ARTIFIXER_CHECKPOINT={checkpoint} is not a file")

    # Read this one from the raw environment: an explicitly empty value means
    # "no GPUs" (CPU-only tests, staging boxes) and must not fall through to
    # autodetection the way an unset variable does.
    devices_raw = os.environ.get(ENV_PREFIX + "CUDA_DEVICES")
    if devices_raw is None:
        cuda_devices = _detect_cuda_devices()
    elif not devices_raw.strip():
        cuda_devices = ()
    else:
        try:
            cuda_devices = tuple(int(item) for item in devices_raw.split(","))
        except ValueError as exc:
            raise ConfigError(f"{ENV_PREFIX}CUDA_DEVICES must be a comma-separated integer list") from exc
        if len(set(cuda_devices)) != len(cuda_devices):
            raise ConfigError(f"{ENV_PREFIX}CUDA_DEVICES contains duplicate device indices")
        if any(index < 0 for index in cuda_devices):
            raise ConfigError(f"{ENV_PREFIX}CUDA_DEVICES indices must be non-negative")

    default_concurrency = max(1, len(cuda_devices))
    max_concurrent_jobs = _env_int("MAX_CONCURRENT_JOBS", default_concurrency, minimum=1, maximum=64)
    if cuda_devices and max_concurrent_jobs > len(cuda_devices):
        raise ConfigError(
            f"{ENV_PREFIX}MAX_CONCURRENT_JOBS={max_concurrent_jobs} exceeds the "
            f"{len(cuda_devices)} visible GPU(s); a stage needs a dedicated device"
        )

    extra_env: dict[str, str] = {}
    for name in ("HF_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_CACHE", "MOGE_MODEL_PATH", "TORCH_HOME", "TRITON_CACHE_DIR"):
        value = os.environ.get(name)
        if value:
            extra_env[name] = value

    settings = Settings(
        data_root=data_root,
        repo_root=repo_root,
        python_executable=_env("PYTHON", "") or os.environ.get("PYTHON", "") or "python",
        api_keys=api_keys,
        require_auth=require_auth,
        trusted_hosts=_env_list("TRUSTED_HOSTS", ("*",)),
        cors_origins=_env_list("CORS_ORIGINS", ()),
        max_request_bytes=_env_int("MAX_REQUEST_BYTES", 32 * GIB, minimum=1024),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 120, minimum=0),
        rate_limit_burst=_env_int("RATE_LIMIT_BURST", 40, minimum=1),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 16 * GIB, minimum=1024),
        max_archive_members=_env_int("MAX_ARCHIVE_MEMBERS", 20_000, minimum=4),
        max_uncompressed_bytes=_env_int("MAX_UNCOMPRESSED_BYTES", 64 * GIB, minimum=1024),
        max_compression_ratio=_env_float("MAX_COMPRESSION_RATIO", 200.0, minimum=1.0),
        max_images=_env_int("MAX_IMAGES", 4_000, minimum=2),
        max_trajectory_frames=_env_int("MAX_TRAJECTORY_FRAMES", 2_000, minimum=1),
        max_json_bytes=_env_int("MAX_JSON_BYTES", 32 * 1024 * 1024, minimum=1024),
        artifixer_checkpoint=checkpoint,
        artifixer_model_id=_env("ARTIFIXER_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers") or "",
        cuda_devices=cuda_devices,
        max_concurrent_jobs=max_concurrent_jobs,
        queue_capacity=_env_int("QUEUE_CAPACITY", 256, minimum=1),
        stage_timeout_seconds=_env_int("STAGE_TIMEOUT_SECONDS", 24 * 3600, minimum=30),
        reconstruction_steps_default=_env_int("RECONSTRUCTION_STEPS_DEFAULT", 10_000, minimum=100),
        reconstruction_steps_max=_env_int("RECONSTRUCTION_STEPS_MAX", 100_000, minimum=100),
        artifixer3d_steps_default=_env_int("ARTIFIXER3D_STEPS_DEFAULT", 30_000, minimum=100),
        artifixer3d_steps_max=_env_int("ARTIFIXER3D_STEPS_MAX", 200_000, minimum=100),
        # Upstream trains ArtiFixer3D from scratch; warm-starting from the base
        # reconstruction converges faster but is a deviation, so it is opt-in.
        artifixer3d_warm_start=_env_bool("ARTIFIXER3D_WARM_START", False),
        keep_intermediate=_env_bool("KEEP_INTERMEDIATE", True),
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        metrics_enabled=_env_bool("METRICS_ENABLED", True),
        docs_enabled=_env_bool("DOCS_ENABLED", True),
        extra_env=extra_env,
    )
    if settings.reconstruction_steps_default > settings.reconstruction_steps_max:
        raise ConfigError("RECONSTRUCTION_STEPS_DEFAULT exceeds RECONSTRUCTION_STEPS_MAX")
    if settings.artifixer3d_steps_default > settings.artifixer3d_steps_max:
        raise ConfigError("ARTIFIXER3D_STEPS_DEFAULT exceeds ARTIFIXER3D_STEPS_MAX")
    return settings
