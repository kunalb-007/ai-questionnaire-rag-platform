"""
routes.py — FastAPI route handlers.

These handlers are intentionally thin:
  - Validate HTTP-layer concerns (file presence, content-type).
  - Delegate all business logic to the service layer.
  - Map service-layer exceptions to appropriate HTTP status codes.
  - Return Pydantic response models.

HTTP status code rationale:
  400 Bad Request     — client sent invalid input (bad file type, empty doc).
  422 Unprocessable   — Pydantic validation failed (auto-handled by FastAPI).
  500 Internal Error  — unexpected server-side failure.
  502 Bad Gateway     — LLM API returned an error.
  503 Service Unavail — Pinecone unreachable.
  504 Gateway Timeout — LLM did not respond in time.

Request ID:
  Each request gets a UUID injected by the middleware in main.py.
  We pull it from request.state and include it in log lines so all log
  entries from a single request can be correlated.
"""

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    UploadResponse,
)
from app.config import settings
from app.services.document_service import (
    EmptyDocumentError,
    IngestionError,
    UnsupportedFileTypeError,
    ingest_upload,
)
from app.services.query_service import (
    LLMAPIError,
    LLMTimeoutError,
    RetrievalError,
    run_query,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Upload a PDF, DOCX, or TXT file. The file is parsed, chunked, "
                "embedded, and stored in Pinecone ready for querying.",
)
async def upload_document(
        request: Request,
        file: UploadFile = File(..., description="PDF, DOCX, or TXT file to ingest."),
) -> UploadResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    filename = file.filename or "unknown"

    logger.info("[%s] Upload request — file=%s", request_id, filename)

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = ingest_upload(filename=filename, file_bytes=file_bytes)
    except UnsupportedFileTypeError as exc:
        logger.warning("[%s] Unsupported file type: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except EmptyDocumentError as exc:
        logger.warning("[%s] Empty document: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    except IngestionError as exc:
        logger.error("[%s] Ingestion error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {exc}",
        )

    logger.info(
        "[%s] Upload complete — file=%s chunks=%d",
        request_id,
        result["filename"],
        result["chunks_created"],
    )

    return UploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunks_created=result["chunks_created"],
        status=result["status"],
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against indexed documents",
    description="Submit a question. The system retrieves the most relevant document "
                "chunks and generates a cited answer using the configured LLM.",
)
async def query_documents(
        request: Request,
        body: QueryRequest,
) -> QueryResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info(
        "[%s] Query request — question_len=%d top_k=%d",
        request_id,
        len(body.question),
        body.top_k,
    )

    try:
        result = run_query(question=body.question, top_k=body.top_k)
    except RetrievalError as exc:
        logger.error("[%s] Retrieval failure: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {exc}",
        )
    except LLMTimeoutError as exc:
        logger.error("[%s] LLM timeout: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        )
    except LLMAPIError as exc:
        logger.error("[%s] LLM API error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {exc}",
        )

    logger.info(
        "[%s] Query complete — retrieval=%.0fms generation=%.0fms",
        request_id,
        result["retrieval_latency_ms"],
        result["generation_latency_ms"],
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[
            SourceReference(document=s["document"], page=s["page"])
            for s in result["sources"]
        ],
        retrieval_latency_ms=result["retrieval_latency_ms"],
        generation_latency_ms=result["generation_latency_ms"],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns 200 if the API is running and Pinecone is reachable.",
)
async def health_check() -> HealthResponse:
    """
    Lightweight health check.

    Checks Pinecone connectivity without performing an embedding or LLM call.
    """
    pinecone_status = "unreachable"

    try:
        from app.vectorstore.pinecone_service import get_index

        index = get_index()
        index.describe_index_stats()

        pinecone_status = "connected"

    except Exception as exc:
        logger.warning("Health check: Pinecone unreachable — %s", exc)

    return HealthResponse(
        status="ok",
        pinecone=pinecone_status,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )