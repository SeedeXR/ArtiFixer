# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed API errors with stable machine-readable codes."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for errors that map to a structured HTTP response.

    Messages on these exceptions are returned to the caller verbatim, so they
    must never embed absolute server paths or credentials. Internal detail
    belongs in the log record, not in ``message``.
    """

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self, request_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {"code": self.code, "message": self.message, "request_id": request_id}
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class BadRequest(ApiError):
    status_code = 400
    code = "bad_request"


class Unauthorized(ApiError):
    status_code = 401
    code = "unauthorized"


class Forbidden(ApiError):
    status_code = 403
    code = "forbidden"


class NotFound(ApiError):
    status_code = 404
    code = "not_found"


class Conflict(ApiError):
    status_code = 409
    code = "conflict"


class PayloadTooLarge(ApiError):
    status_code = 413
    code = "payload_too_large"


class UnprocessableInput(ApiError):
    """The request parsed but the referenced data is unusable (e.g. bad COLMAP)."""

    status_code = 422
    code = "unprocessable_input"


class RateLimited(ApiError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, retry_after: int, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServiceUnavailable(ApiError):
    status_code = 503
    code = "service_unavailable"
