"""
main.py — FastAPI application entry point.

Application startup order:
  1. configure_logging() — set log level from settings before any logger fires.
  2. App creation — registers routes and exception handlers.
  3. Request ID middleware — injects a UUID into every request's state.
  4. Uvicorn starts serving (when run via `uvicorn app.main:app`).

Middleware — Request ID:
  Every incoming request is assigned a UUID (X-Request-ID). If the client
  sends one, we reuse it (useful for tracing across services). Otherwise we
  generate one. The ID propagates through request.state so route handlers
  and services can include it in log lines.

Global exception handler:
  Catches any unhandled exception (programming errors, library bugs) and
  returns a consistent JSON error shape instead of a raw stack trace.
  The stack trace is logged at ERROR level for debugging.
  Critically, we do NOT return the exception message to the client — it may
  contain internal paths, library names, or other sensitive detail.
"""

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes import router
from app.utils.logging_config import configure_logging

# Configure logging as the very first thing — before any module-level loggers fire.
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Document Intelligence",
    description=(
        "RAG pipeline for enterprise security and compliance documents. "
        "Upload PDF/DOCX/TXT files and query them with natural language."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware — attach a request ID to every request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """
    Inject a request ID into request.state for use in log lines.

    If the client sends X-Request-ID, reuse it. Otherwise generate a UUID.
    The ID is echoed back in the X-Request-ID response header so clients
    can correlate their requests to our logs.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler — catch-all for unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Return a safe, consistent error body for any unhandled exception.

    We log the full traceback (exc_info=True) at ERROR level but return
    only a generic message to the client. This prevents leaking internal
    details while still making the error debuggable.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "[%s] Unhandled exception on %s %s",
        request_id,
        request.method,
        request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected internal error occurred.",
            "error_type": "InternalServerError",
        },
    )


# ---------------------------------------------------------------------------
# Include routes
# ---------------------------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    logger.info(
        "Enterprise Document Intelligence API starting — "
        "model=%s pinecone_index=%s",
        settings.llm_model,
        settings.pinecone_index_name,
    )

