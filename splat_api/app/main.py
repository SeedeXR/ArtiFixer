# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASGI application factory.

Run with::

    uvicorn splat_api.app.main:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from splat_api import __version__
from splat_api.app import routes
from splat_api.app.config import Settings, load_settings
from splat_api.app.errors import ApiError, RateLimited
from splat_api.app.jobstore import AsyncJobStore, JobStore
from splat_api.app.ratelimit import TokenBucketLimiter
from splat_api.app.scheduler import Scheduler
from splat_api.app.schemas import HealthStatus
from splat_api.app.security import BodyLimitMiddleware, RateLimitMiddleware, RequestContextMiddleware

logger = logging.getLogger("splat_api")


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line, so logs are queryable without a parser.

    The record's ``request_id`` (attached by :class:`RequestContextMiddleware`) is
    promoted to a top-level field to make joining a client error to server logs a
    single query.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False


def _readiness_checks(settings: Settings, store: JobStore, scheduler: Scheduler) -> dict[str, Any]:
    usage = shutil.disk_usage(settings.data_root)
    checks: dict[str, Any] = {
        "database": "ok",
        "data_root_writable": os.access(settings.data_root, os.W_OK),
        "disk_free_bytes": usage.free,
        "disk_free_ratio": round(usage.free / usage.total, 4) if usage.total else 0.0,
        "gpu_count": len(settings.cuda_devices),
        "queue_depth": scheduler.queue_depth(),
        "artifixer_checkpoint_configured": settings.artifixer_available,
    }
    try:
        checks["jobs_by_state"] = store.state_counts()
    except Exception as exc:  # pragma: no cover - surfaced as degraded
        checks["database"] = f"error: {exc}"
    return checks


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        store = JobStore(settings.database_path)
        async_store = AsyncJobStore(store)
        scheduler = Scheduler(settings, async_store)
        application.state.settings = settings
        application.state.store = async_store
        application.state.sync_store = store
        application.state.scheduler = scheduler
        await scheduler.start()
        logger.info(
            "splat_api %s ready: data_root=%s gpus=%s artifixer=%s",
            __version__,
            settings.data_root,
            list(settings.cuda_devices) or "none",
            "configured" if settings.artifixer_available else "absent",
        )
        try:
            yield
        finally:
            await scheduler.stop()
            store.close()

    app = FastAPI(
        title="COLMAP to Gaussian Splat API",
        description=(
            "Turns a COLMAP sparse reconstruction into a Gaussian splat using 3DGUT, "
            "optionally corrected by the ArtiFixer video diffusion model (ArtiFixer3D / ArtiFixer3D+)."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    # Middleware order is bottom-up in Starlette: the last added runs first. We
    # want request context (ids, logging) outermost, then host/rate/size checks,
    # so that every rejection is still logged with a request id.
    # No GZipMiddleware. It would compress artifact downloads too: Starlette only
    # exempts text/event-stream, so every FileResponse would be deflated at level 9
    # on the event loop, defeating the sendfile path and burning CPU on already
    # incompressible data (PLY, .pt, .zip, .mp4). JSON responses here are small;
    # compress them at the reverse proxy if it matters.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["authorization", "x-api-key", "content-type"],
            max_age=600,
        )
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
        max_json_bytes=settings.max_json_bytes,
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=TokenBucketLimiter(
            per_minute=settings.rate_limit_per_minute, burst=settings.rate_limit_burst
        ),
        settings=settings,
    )
    if settings.trusted_hosts and settings.trusted_hosts != ("*",):
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        if exc.status_code >= 500:
            logger.error("api error %s: %s", exc.code, exc.message, extra={"request_id": request_id})
        response = JSONResponse(status_code=exc.status_code, content=exc.to_payload(request_id))
        if isinstance(exc, RateLimited):
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request body failed validation",
                    "request_id": request_id,
                    "details": {"errors": json.loads(json.dumps(exc.errors(), default=str))},
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled exception", extra={"request_id": request_id})
        # Never leak exception text: tracebacks from the pipeline contain absolute
        # server paths. The request id is the handle for correlating with logs.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": request_id,
                }
            },
        )

    @app.get("/healthz", response_model=HealthStatus, tags=["meta"])
    async def healthz() -> HealthStatus:
        """Liveness: the process is up and serving. Never touches the GPU."""
        return HealthStatus(status="ok", version=__version__, checks={"process": "ok"})

    @app.get("/readyz", response_model=HealthStatus, tags=["meta"])
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: storage writable, database reachable, disk not exhausted.

        The checks stat the filesystem and query SQLite, so they run in a thread.
        This endpoint is unauthenticated and exempt from rate limiting, which makes
        it exactly the wrong place to block the event loop.
        """
        checks = await asyncio.to_thread(
            _readiness_checks,
            request.app.state.settings,
            request.app.state.sync_store,
            request.app.state.scheduler,
        )
        healthy = (
            checks["database"] == "ok"
            and bool(checks["data_root_writable"])
            and float(checks["disk_free_ratio"]) > 0.02
        )
        status = HealthStatus(status="ok" if healthy else "degraded", version=__version__, checks=checks)
        return JSONResponse(status_code=200 if healthy else 503, content=status.model_dump())

    if settings.metrics_enabled:

        @app.get("/metrics", response_class=PlainTextResponse, tags=["meta"])
        async def metrics(request: Request, _: routes.RequireRead) -> PlainTextResponse:
            """Prometheus text exposition of queue and job counters.

            Requires the `read` scope: job counts are operational information about
            what this node is doing. Give the scraper a read-only key.
            """
            store: JobStore = request.app.state.sync_store
            scheduler: Scheduler = request.app.state.scheduler
            counts = await asyncio.to_thread(store.state_counts)
            lines = [
                "# HELP splat_api_jobs_total Jobs by state.",
                "# TYPE splat_api_jobs_total gauge",
            ]
            for state in ("queued", "running", "succeeded", "failed", "cancelled"):
                lines.append(f'splat_api_jobs_total{{state="{state}"}} {counts.get(state, 0)}')
            lines += [
                "# HELP splat_api_queue_depth Jobs waiting for a GPU worker.",
                "# TYPE splat_api_queue_depth gauge",
                f"splat_api_queue_depth {scheduler.queue_depth()}",
                "# HELP splat_api_workers Configured concurrent GPU workers.",
                "# TYPE splat_api_workers gauge",
                f"splat_api_workers {request.app.state.settings.max_concurrent_jobs}",
            ]
            return PlainTextResponse("\n".join(lines) + "\n")

    app.include_router(routes.router)
    return app


def main() -> int:
    """Serve the app with uvicorn, using uvloop/httptools when available."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "splat_api.app.main:create_app",
        factory=True,
        host=os.environ.get("SPLAT_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("SPLAT_API_PORT", "8000")),
        # One process: the scheduler, its queue and the GPU assignment are
        # process-local state. Concurrency comes from the worker tasks, not from
        # forked API workers, and the real bottleneck is the GPU.
        workers=1,
        loop="uvloop",
        http="httptools",
        log_config=None,
        access_log=False,
        timeout_keep_alive=75,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("SPLAT_API_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
