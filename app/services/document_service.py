"""
document_service.py — Orchestrates the full document ingestion pipeline.

Demo-friendly change:
  Previously: read file from disk path
  Now: accepts raw bytes from HTTP upload (UploadFile.read())

The service still writes a temp file internally because the parser
reads from disk (PyMuPDF and Docling both need a file path).
The temp file is hidden from the caller — they just pass bytes.

Pipeline:
  bytes → temp file → parse → clean → chunk → embed → Pinecone
"""

import logging
import tempfile
import time
from pathlib import Path

from app.config import settings
from app.ingestion.parser import parse_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedding_service import embed_texts
from app.vectorstore.pinecone_service import ensure_index, insert_chunks

logger = logging.getLogger(__name__)

# ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
EMBEDDING_DIMENSION = settings.embedding_dimension  # 1536


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class IngestionError(RuntimeError):
    pass


def ingest_upload(filename: str, file_bytes: bytes) -> dict:
    """
    Ingest an uploaded file provided as raw bytes.

    Args:
        filename   : original filename (used for metadata and doc_id)
        file_bytes : raw bytes from the HTTP upload — NOT a disk path

    Why bytes instead of file path?
      The FastAPI route does:  file_bytes = await file.read()
      No temp file is visible to the caller.
      The service writes its own temp file internally for parsers that
      need a disk path (PyMuPDF, Docling). Caller stays clean.

    Returns:
        dict with document_id, filename, chunks_created, status, timings
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{suffix}' is not supported. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    timings: dict[str, float] = {}

    # Write bytes to temp file so parser (disk-based) can be reused
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # --- Parse ---
        t0 = time.perf_counter()
        try:
            pages = parse_document(tmp_path)
            for page in pages:
                page["filename"] = filename  # restore original name (not temp path)
        except FileNotFoundError as exc:
            raise IngestionError(f"Temp file lost during processing: {exc}") from exc
        except ValueError as exc:
            raise EmptyDocumentError(str(exc)) from exc
        timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- Clean ---
        t0 = time.perf_counter()
        pages = clean_pages(pages)
        timings["clean_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if not pages:
            raise EmptyDocumentError(f"'{filename}' produced no text after cleaning.")

        # --- Chunk ---
        t0 = time.perf_counter()
        chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        timings["chunk_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if not chunks:
            raise EmptyDocumentError(f"'{filename}' produced no chunks after splitting.")

        # --- Embed (OpenAI text-embedding-3-small) ---
        t0 = time.perf_counter()
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = embed_texts(texts)
        except Exception as exc:
            raise IngestionError(f"Embedding generation failed: {exc}") from exc
        timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- Store in Pinecone ---
        t0 = time.perf_counter()
        try:
            ensure_index()
            insert_chunks(chunks, embeddings)
        except Exception as exc:
            raise IngestionError(f"Pinecone storage failed: {exc}") from exc
        timings["store_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    finally:
        # Always remove temp file — regardless of success or failure
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(
        "Ingestion complete — file=%s chunks=%d timings=%s",
        filename, len(chunks), timings,
    )

    return {
        "document_id": _make_doc_id(filename),
        "filename": filename,
        "chunks_created": len(chunks),
        "status": "indexed",
        "timings_ms": timings,
    }


def _make_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace(" ", "_").replace(".", "_").lower()