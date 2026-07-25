# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process token-bucket rate limiting.

Scope: one process. That is the correct granularity here because the service is
GPU-bound and single-node; the real admission control is the job queue, and this
limiter exists to keep cheap endpoints from being used to hammer the store or to
brute-force API keys.

Buckets are keyed by principal (API key id) when authenticated and by client IP
otherwise, so one noisy tenant cannot exhaust another's allowance.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    """Classic token bucket: ``rate`` tokens/second up to ``burst``."""

    def __init__(self, *, per_minute: int, burst: int, max_buckets: int = 20_000) -> None:
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._enabled = per_minute > 0
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()
        self._max_buckets = max_buckets

    @property
    def enabled(self) -> bool:
        return self._enabled

    def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, int]:
        """Try to spend ``cost`` tokens.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is rounded up
        to at least one second so a client that honours it always makes progress.
        """
        if not self._enabled:
            return True, 0
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                # Hard LRU bound. Age-based pruning alone does not hold under a
                # flood of fresh keys, which is exactly when the bound matters.
                while len(self._buckets) >= self._max_buckets:
                    self._buckets.popitem(last=False)
                bucket = _Bucket(tokens=self._burst, updated_at=now)
                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
            bucket.updated_at = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0
            deficit = cost - bucket.tokens
            retry_after = max(1, int(deficit / self._rate) + 1) if self._rate > 0 else 60
            return False, retry_after

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
