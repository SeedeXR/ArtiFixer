# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication, authorization, and the request-hardening middlewares."""

from __future__ import annotations

import hmac
import logging
import secrets
import time
from dataclasses import dataclass

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from splat_api.app.config import Settings, hash_api_key
from splat_api.app.errors import ApiError, BadRequest, Forbidden, PayloadTooLarge, RateLimited, Unauthorized
from splat_api.app.ratelimit import TokenBucketLimiter

logger = logging.getLogger("splat_api.security")

ANONYMOUS_KEY_ID = "anonymous"
# When auth is disabled the caller is trusted for ordinary work but never for
# admin operations, which can name server-side filesystem paths.
ANONYMOUS_SCOPES = frozenset({"read", "write"})

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    # This is a JSON API; a strict CSP costs nothing and blocks any injected markup
    # from loading resources should a response ever be rendered as HTML.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cache-Control": "no-store",
}


@dataclass(frozen=True)
class Principal:
    key_id: str
    scopes: frozenset[str]
    authenticated: bool

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise Forbidden(f"This credential lacks the {scope!r} scope")


def _extract_credential(request: Request) -> str | None:
    """Read the credential from ``Authorization: Bearer`` or ``X-API-Key``."""
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            raise Unauthorized("Authorization header must use the Bearer scheme")
        return value.strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


def authenticate(request: Request, settings: Settings) -> Principal:
    """Resolve the caller.

    The digest comparison uses :func:`hmac.compare_digest` against every
    configured key, and an unknown credential still walks the full list, so
    neither the value nor the number of configured keys is observable through
    response timing.
    """
    credential = _extract_credential(request)
    if not settings.require_auth:
        if credential is None:
            return Principal(key_id=ANONYMOUS_KEY_ID, scopes=frozenset(ANONYMOUS_SCOPES), authenticated=False)
        # Auth is optional but a credential was offered: honour it so scoped keys
        # still work, and reject a wrong one rather than silently downgrading.
    if credential is None:
        raise Unauthorized("Missing API key. Send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.")

    digest = hash_api_key(credential)
    matched = None
    for key in settings.api_keys:
        if hmac.compare_digest(key.digest, digest):
            matched = key
    if matched is None:
        logger.warning("rejected credential", extra={"key_id": "unknown"})
        raise Unauthorized("Invalid API key")
    return Principal(key_id=matched.key_id, scopes=matched.scopes, authenticated=True)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, and emit one access log line.

    The request id is echoed in ``X-Request-Id`` and embedded in every error
    payload so an operator can join a client-visible failure to a server log
    without the response having to carry internal detail.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        incoming = request.headers.get("x-request-id", "")
        # Never trust a client-supplied id verbatim: it lands in log records.
        request_id = incoming if incoming.isalnum() and len(incoming) <= 64 else secrets.token_hex(8)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error",
                extra={"request_id": request_id, "path": request.url.path, "method": request.method},
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-Id"] = request_id
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        logger.info(
            "%s %s -> %d in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"request_id": request_id},
        )
        return response


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests from the declared ``Content-Length``.

    Two limits apply: a large one for binary uploads and a much smaller one for
    JSON, since a job request is a few kilobytes and there is no reason to let a
    caller stream gigabytes into the pydantic parser.

    This is a pre-filter only. Streaming upload handlers enforce their own byte
    cap while reading, because ``Content-Length`` may be absent under chunked
    transfer encoding.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, max_json_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes
        self._max_json_bytes = max_json_bytes

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        raw_length = request.headers.get("content-length")
        if raw_length is not None:
            request_id = getattr(request.state, "request_id", "-")
            try:
                length = int(raw_length)
            except ValueError:
                error: ApiError = BadRequest("Malformed Content-Length header")
                return JSONResponse(
                    status_code=error.status_code, content=error.to_payload(request_id)
                )
            content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
            limit = self._max_json_bytes if content_type == "application/json" else self._max_bytes
            if length > limit:
                error = PayloadTooLarge(f"Request body is {length} bytes; the limit is {limit}")
                return JSONResponse(
                    status_code=error.status_code, content=error.to_payload(request_id)
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the token bucket before any handler work happens."""

    def __init__(self, app: ASGIApp, limiter: TokenBucketLimiter, settings: Settings) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._settings = settings

    def _bucket_keys(self, request: Request) -> list[str]:
        """Every bucket a request must pay into.

        The source address is always charged. Bucketing only on the offered
        credential would let a caller rotating credential values mint a fresh
        allowance per request, which is precisely the key-guessing case this
        limiter exists to bound — the limiter runs before authentication, so it
        cannot tell a wrong key from a right one.

        An authenticated caller additionally pays into a per-credential bucket, so
        one tenant cannot spend another's allowance from a shared address.
        """
        client = request.client
        keys = [f"ip:{client.host}" if client else "ip:unknown"]
        credential = request.headers.get("authorization") or request.headers.get("x-api-key")
        if credential:
            # Bucket on the digest, not the key: bucket keys end up in memory dumps
            # and must not be replayable credentials.
            keys.append("key:" + hash_api_key(credential)[:32])
        return keys

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not self._limiter.enabled or request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)
        allowed, retry_after = True, 0
        for key in self._bucket_keys(request):
            key_allowed, key_retry = self._limiter.acquire(key)
            if not key_allowed:
                allowed, retry_after = False, max(retry_after, key_retry)
        if not allowed:
            error = RateLimited("Rate limit exceeded", retry_after=retry_after)
            response: Response = JSONResponse(
                status_code=error.status_code,
                content=error.to_payload(getattr(request.state, "request_id", "-")),
            )
            response.headers["Retry-After"] = str(retry_after)
            return response
        return await call_next(request)
