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

import csv
import io
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QuestionnaireResponse,
    QuestionResult,
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
from app.services.questionnaire_service import (
    QuestionnaireParseError,
    parse_questionnaire_csv,
    process_questionnaire,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Maximum permitted upload size enforced before the file is fully read into
# memory. Prevents memory exhaustion from oversized or malicious uploads.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

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

    # Read up to the limit + 1 so we can distinguish "at limit" from "over
    # limit" without loading the entire file into memory first.
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the maximum allowed size of "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
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
        status=result["status"],
        top_retrieval_score=result["top_retrieval_score"],
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

# ---------------------------------------------------------------------------
# POST /questionnaires/process
# ---------------------------------------------------------------------------

@router.post(
    "/questionnaires/process",
    response_model=QuestionnaireResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a security questionnaire CSV",
    description=(
            "Upload a CSV file containing security questions. Each question is "
            "answered using RAG against the indexed documents. Returns structured "
            "results with answer text, source citations, and a status per question: "
            "'answered', 'insufficient_evidence', or 'no_documents'."
    ),
)
async def process_questionnaire_endpoint(
        request: Request,
        file: UploadFile = File(..., description="CSV file with a question column."),
        question_column: str = Query(
            default="question",
            description=(
                    "Name of the CSV column that contains the questions. "
                    "Case-insensitive. Defaults to 'question'."
            ),
        ),
) -> QuestionnaireResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    filename = file.filename or "unknown"

    logger.info(
        "[%s] Questionnaire process request — file=%s column=%s",
        request_id, filename, question_column,
    )

    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the maximum allowed size of "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    try:
        rows = parse_questionnaire_csv(file_bytes, question_column)
    except QuestionnaireParseError as exc:
        logger.warning("[%s] CSV parse error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )

    logger.info(
        "[%s] Processing %d question(s) from '%s'.",
        request_id, len(rows), filename,
    )

    raw_results = process_questionnaire(rows)

    # Build typed QuestionResult objects for the response model
    results = [
        QuestionResult(
            row=r["row"],
            question=r["question"],
            answer=r["answer"],
            status=r["status"],
            sources=[
                SourceReference(document=s["document"], page=s["page"])
                for s in r["sources"]
            ],
            top_retrieval_score=r["top_retrieval_score"],
        )
        for r in raw_results
    ]

    # Aggregate counts for the summary fields
    answered = sum(1 for r in raw_results if r["status"] == "answered")
    insufficient = sum(1 for r in raw_results if r["status"] == "insufficient_evidence")
    no_docs = sum(1 for r in raw_results if r["status"] == "no_documents")

    logger.info(
        "[%s] Questionnaire complete — total=%d answered=%d insufficient=%d no_docs=%d",
        request_id, len(results), answered, insufficient, no_docs,
    )

    return QuestionnaireResponse(
        total_questions=len(results),
        answered=answered,
        insufficient_evidence=insufficient,
        no_documents=no_docs,
        results=results,
    )

# ---------------------------------------------------------------------------
# POST /questionnaires/export
# ---------------------------------------------------------------------------

@router.post(
    "/questionnaires/export",
    summary="Process a questionnaire CSV and return a completed CSV",
    description=(
            "Upload a CSV file containing security questions. Processes each "
            "question using RAG and returns a completed CSV file for download. "
            "The output CSV contains the original question plus answer, status, "
            "top retrieval score, and source citations columns."
    ),
    response_class=StreamingResponse,
)
async def export_questionnaire_endpoint(
        request: Request,
        file: UploadFile = File(..., description="CSV file with a question column."),
        question_column: str = Query(
            default="question",
            description="Name of the CSV column that contains the questions.",
        ),
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    filename = file.filename or "questionnaire"

    logger.info(
        "[%s] Questionnaire export request — file=%s column=%s",
        request_id, filename, question_column,
    )

    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the maximum allowed size of "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    try:
        rows = parse_questionnaire_csv(file_bytes, question_column)
    except QuestionnaireParseError as exc:
        logger.warning("[%s] CSV parse error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )

    raw_results = process_questionnaire(rows)

    # Build the output CSV in memory using stdlib csv — no extra dependencies
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["row", "question", "answer", "status", "top_retrieval_score", "sources"],
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for r in raw_results:
        # Format sources as a readable string for Excel / Sheets compatibility.
        # Semicolon-separated so it stays in one cell without breaking CSV structure.
        sources_str = "; ".join(
            f"{s['document']} (p.{s['page']})" for s in r["sources"]
        ) or "—"

        writer.writerow({
            "row": r["row"],
            "question": r["question"],
            "answer": r["answer"],
            "status": r["status"],
            "top_retrieval_score": r["top_retrieval_score"] if r["top_retrieval_score"] is not None else "—",
            "sources": sources_str,
        })

    output.seek(0)
    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility

    # Derive output filename from the uploaded file's stem
    import pathlib
    stem = pathlib.Path(filename).stem
    output_filename = f"{stem}_completed.csv"

    logger.info(
        "[%s] Export complete — %d rows written to '%s'.",
        request_id, len(raw_results), output_filename,
    )

    return StreamingResponse(
        content=io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{output_filename}"'
        },
    )