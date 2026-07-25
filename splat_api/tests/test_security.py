# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configuration parsing, authentication, and rate limiting."""

from __future__ import annotations

from dataclasses import replace

import pytest

from splat_api.app.config import (
    ConfigError,
    Settings,
    hash_api_key,
    load_settings,
    parse_api_keys,
)
from splat_api.app.errors import Forbidden, Unauthorized
from splat_api.app.ratelimit import TokenBucketLimiter
from splat_api.app.security import authenticate

GOOD_SECRET = "sk-abcdefghijklmnopqrstuvwxyz"


class _FakeRequest:
    """Minimal stand-in for the header access `authenticate` performs."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = {key.lower(): value for key, value in headers.items()}


class TestParseApiKeys:
    def test_parses_plaintext_secrets_and_expands_admin(self) -> None:
        keys = parse_api_keys(f"ci:{GOOD_SECRET}:write|ops:{GOOD_SECRET}x:admin")
        assert [key.key_id for key in keys] == ["ci", "ops"]
        assert keys[0].scopes == frozenset({"write"})
        assert keys[1].scopes == frozenset({"read", "write", "admin"})

    def test_stores_only_the_digest(self) -> None:
        keys = parse_api_keys(f"ci:{GOOD_SECRET}:write")
        assert keys[0].digest == hash_api_key(GOOD_SECRET)
        assert GOOD_SECRET not in repr(keys[0])

    def test_accepts_a_prehashed_secret(self) -> None:
        digest = hash_api_key(GOOD_SECRET)
        keys = parse_api_keys(f"ops:sha256:{digest}:read+write")
        assert keys[0].digest == digest
        assert keys[0].scopes == frozenset({"read", "write"})

    def test_empty_specification_yields_no_keys(self) -> None:
        assert parse_api_keys("") == ()

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            (f"ci:{GOOD_SECRET}", "must be"),
            ("ci:short:write", "at least"),
            (f"ci:{GOOD_SECRET}:superuser", "unknown scopes"),
            (f"ci:{GOOD_SECRET}:", "no scopes"),
            (f"bad id:{GOOD_SECRET}:read", "must match"),
            (f"ci:{GOOD_SECRET}:read|ci:{GOOD_SECRET}y:read", "Duplicate API key id"),
            (f"ci:{GOOD_SECRET}:read|other:{GOOD_SECRET}:read", "Duplicate API key secret"),
            ("ops:sha256:nothex:read", "malformed sha256"),
        ],
    )
    def test_rejects_malformed_specifications(self, spec: str, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            parse_api_keys(spec)


class TestSettingsValidation:
    def test_rejects_a_repo_root_that_is_not_the_artifixer_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(tmp_path))
        monkeypatch.setenv("SPLAT_API_KEYS", "")
        with pytest.raises(ConfigError, match="does not look like the ArtiFixer repo"):
            load_settings()

    def test_requires_keys_unless_auth_is_explicitly_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from splat_api.tests.conftest import REPO_ROOT

        monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(REPO_ROOT))
        monkeypatch.setenv("SPLAT_API_KEYS", "")
        with pytest.raises(ConfigError, match="No API keys configured"):
            load_settings()

        monkeypatch.setenv("SPLAT_API_REQUIRE_AUTH", "false")
        assert load_settings().require_auth is False

    def test_rejects_more_workers_than_gpus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from splat_api.tests.conftest import REPO_ROOT, WRITE_KEY

        monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(REPO_ROOT))
        monkeypatch.setenv("SPLAT_API_KEYS", f"w:{WRITE_KEY}:write")
        monkeypatch.setenv("SPLAT_API_CUDA_DEVICES", "0")
        monkeypatch.setenv("SPLAT_API_MAX_CONCURRENT_JOBS", "4")
        with pytest.raises(ConfigError, match="exceeds the 1 visible GPU"):
            load_settings()

    def test_rejects_a_nonexistent_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from splat_api.tests.conftest import REPO_ROOT, WRITE_KEY

        monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(REPO_ROOT))
        monkeypatch.setenv("SPLAT_API_KEYS", f"w:{WRITE_KEY}:write")
        monkeypatch.setenv("SPLAT_API_ARTIFIXER_CHECKPOINT", str(tmp_path / "absent.pt"))
        with pytest.raises(ConfigError, match="is not a file"):
            load_settings()

    def test_rejects_a_malformed_boolean(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from splat_api.tests.conftest import REPO_ROOT, WRITE_KEY

        monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(REPO_ROOT))
        monkeypatch.setenv("SPLAT_API_KEYS", f"w:{WRITE_KEY}:write")
        monkeypatch.setenv("SPLAT_API_METRICS_ENABLED", "sometimes")
        with pytest.raises(ConfigError, match="must be a boolean"):
            load_settings()


class TestAuthenticate:
    def test_accepts_a_bearer_token(self, settings: Settings) -> None:
        from splat_api.tests.conftest import WRITE_KEY

        principal = authenticate(_FakeRequest({"Authorization": f"Bearer {WRITE_KEY}"}), settings)
        assert principal.key_id == "writer"
        assert principal.authenticated is True

    def test_accepts_the_api_key_header(self, settings: Settings) -> None:
        from splat_api.tests.conftest import ADMIN_KEY

        principal = authenticate(_FakeRequest({"X-API-Key": ADMIN_KEY}), settings)
        assert principal.key_id == "ops"
        assert "admin" in principal.scopes

    def test_rejects_a_missing_credential(self, settings: Settings) -> None:
        with pytest.raises(Unauthorized, match="Missing API key"):
            authenticate(_FakeRequest({}), settings)

    def test_rejects_a_wrong_credential(self, settings: Settings) -> None:
        with pytest.raises(Unauthorized, match="Invalid API key"):
            authenticate(_FakeRequest({"X-API-Key": "wrong-key-000000000000000"}), settings)

    def test_rejects_a_non_bearer_scheme(self, settings: Settings) -> None:
        with pytest.raises(Unauthorized, match="Bearer scheme"):
            authenticate(_FakeRequest({"Authorization": "Basic dXNlcjpwYXNz"}), settings)

    def test_scope_enforcement(self, settings: Settings) -> None:
        from splat_api.tests.conftest import READ_KEY

        principal = authenticate(_FakeRequest({"X-API-Key": READ_KEY}), settings)
        principal.require("read")
        with pytest.raises(Forbidden, match="'write' scope"):
            principal.require("write")

    def test_anonymous_principal_when_auth_disabled(self, settings: Settings) -> None:
        open_settings = replace(settings, require_auth=False)
        principal = authenticate(_FakeRequest({}), open_settings)
        assert principal.authenticated is False
        assert principal.scopes == frozenset({"read", "write"})
        # Admin remains gated even with auth disabled.
        with pytest.raises(Forbidden):
            principal.require("admin")

    def test_a_bad_credential_still_fails_when_auth_is_optional(self, settings: Settings) -> None:
        open_settings = replace(settings, require_auth=False)
        with pytest.raises(Unauthorized):
            authenticate(_FakeRequest({"X-API-Key": "not-a-key-00000000000000000"}), open_settings)


class TestTokenBucket:
    def test_allows_up_to_the_burst_then_refuses(self) -> None:
        limiter = TokenBucketLimiter(per_minute=60, burst=3)
        assert [limiter.acquire("k")[0] for _ in range(3)] == [True, True, True]
        allowed, retry_after = limiter.acquire("k")
        assert allowed is False
        assert retry_after >= 1

    def test_buckets_are_per_key(self) -> None:
        limiter = TokenBucketLimiter(per_minute=60, burst=1)
        assert limiter.acquire("a")[0] is True
        assert limiter.acquire("b")[0] is True
        assert limiter.acquire("a")[0] is False

    def test_refills_over_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = {"now": 1000.0}
        monkeypatch.setattr("splat_api.app.ratelimit.time.monotonic", lambda: clock["now"])
        limiter = TokenBucketLimiter(per_minute=60, burst=1)
        assert limiter.acquire("k")[0] is True
        assert limiter.acquire("k")[0] is False
        clock["now"] += 2.0
        assert limiter.acquire("k")[0] is True

    def test_disabled_when_rate_is_zero(self) -> None:
        limiter = TokenBucketLimiter(per_minute=0, burst=1)
        assert limiter.enabled is False
        assert all(limiter.acquire("k")[0] for _ in range(100))

    def test_bucket_count_is_hard_bounded(self) -> None:
        """A flood of fresh keys must not grow memory without limit.

        Age-based pruning alone does not hold here: under a flood every bucket is
        recent, so nothing would ever be evicted. Least-recently-used does.
        """
        limiter = TokenBucketLimiter(per_minute=60, burst=5, max_buckets=4)
        for index in range(500):
            limiter.acquire(f"key-{index}")
        assert len(limiter._buckets) <= 4
        assert "key-499" in limiter._buckets
        assert "key-0" not in limiter._buckets

    def test_recently_used_buckets_survive_eviction(self) -> None:
        limiter = TokenBucketLimiter(per_minute=60, burst=5, max_buckets=3)
        for index in range(3):
            limiter.acquire(f"key-{index}")
        limiter.acquire("key-0")  # refresh the oldest
        limiter.acquire("key-new")
        assert "key-0" in limiter._buckets
        assert "key-1" not in limiter._buckets
