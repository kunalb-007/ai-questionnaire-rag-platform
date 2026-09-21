"""
document_service.py — Orchestrates the full document ingestion pipeline.

This service is called by the /documents/upload route. It:
  1. Saves the uploaded bytes to a temp file.
  2. Calls parser → cleaner → chunker → embedder → Qdrant.
  3. Measures latency at each stage.
  4. Raises typed exceptions that the route handler converts to HTTP errors.

Why a service layer?
  The route handler deals with HTTP concerns (file upload, response codes).
  The service layer deals with business logic (what steps to run, in what
  order, what constitutes success or failure). Keeping them separate makes
  each easier to test in isolation.

Phase 7 — Latency measurement:
  We time each pipeline stage independently using time.perf_counter().
  perf_counter() is the highest-resolution timer available in Python and
  is not affected by system clock adjustments (unlike time.time()).

  Why LLM latency dominates in the query pipeline:
    Embedding a query takes ~5-20 ms (local model, in-memory).
    Qdrant search takes ~5-50 ms (local, small collection).
    LLM generation takes 500-5000 ms (network round-trip + token generation).
    At scale, embedding and retrieval become negligible; the LLM is the
    bottleneck. Streaming responses mitigate *perceived* latency.
"""

import logging
import tempfile
import time
from pathlib import Path

from app.config import settings
from app.ingestion.parser import parse_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedding_service import embed_texts, EMBEDDING_DIMENSION
from app.vectorstore.qdrant_service import get_client, ensure_collection, insert_chunks

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class IngestionError(RuntimeError):
    pass


def ingest_upload(
        filename: str,
        file_bytes: bytes,
) -> dict:
    """
    Run the full ingestion pipeline for an uploaded file.

    Args:
        filename: original filename from the upload (used for metadata and doc_id).
        file_bytes: raw bytes of the uploaded file.

    Returns:
        dict with document_id, filename, chunks_created, status, and per-stage
        latencies (for internal logging; not all fields are returned to the client).

    Raises:
        UnsupportedFileTypeError: extension not in ALLOWED_EXTENSIONS.
        EmptyDocumentError: file produces no extractable text.
        IngestionError: any other processing failure.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{suffix}' is not supported. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    timings: dict[str, float] = {}

    # Write bytes to a temp file so the existing parser (which reads from disk)
    # can be reused without modification.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # --- Parse ---
        t0 = time.perf_counter()
        try:
            pages = parse_document(tmp_path)
            # Restore original filename in metadata (temp path has random name)
            for page in pages:
                page["filename"] = filename
        except FileNotFoundError as exc:
            raise IngestionError(f"Temporary file lost during processing: {exc}") from exc
        except ValueError as exc:
            # parse_document raises ValueError for empty / unreadable docs
            raise EmptyDocumentError(str(exc)) from exc
        timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- Clean ---
        t0 = time.perf_counter()
        pages = clean_pages(pages)
        timings["clean_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if not pages:
            raise EmptyDocumentError(
                f"'{filename}' produced no text after cleaning."
            )

        # --- Chunk ---
        t0 = time.perf_counter()
        chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        timings["chunk_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if not chunks:
            raise EmptyDocumentError(
                f"'{filename}' produced no chunks after splitting."
            )

        # --- Embed ---
        t0 = time.perf_counter()
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = embed_texts(texts, settings.embedding_model)
        except Exception as exc:
            raise IngestionError(f"Embedding generation failed: {exc}") from exc
        timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- Store in Qdrant ---
        t0 = time.perf_counter()
        try:
            client = get_client(settings.qdrant_url)
            ensure_collection(client, settings.collection_name, EMBEDDING_DIMENSION)
            insert_chunks(client, settings.collection_name, chunks, embeddings)
        except Exception as exc:
            raise IngestionError(f"Qdrant storage failed: {exc}") from exc
        timings["store_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    finally:
        # Always clean up the temp file regardless of success or failure
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(
        "Ingestion complete — file=%s chunks=%d timings=%s",
        filename,
        len(chunks),
        timings,
    )

    return {
        "document_id": _make_doc_id(filename),
        "filename": filename,
        "chunks_created": len(chunks),
        "status": "indexed",
        "timings_ms": timings,
    }


def _make_doc_id(filename: str) -> str:
    """
    Derive a stable document identifier from the filename.

    Simple approach: strip extension, replace spaces and dots with underscores.
    In production you might use a SHA-256 content hash to detect duplicate uploads.
    """
    stem = Path(filename).stem
    return stem.replace(" ", "_").replace(".", "_").lower()