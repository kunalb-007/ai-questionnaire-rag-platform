"""
schemas.py — Pydantic request and response models for all API endpoints.

Why Pydantic?
  FastAPI uses Pydantic models to:
    1. Validate and parse incoming JSON bodies automatically.
    2. Serialise outgoing responses to JSON.
    3. Generate an OpenAPI schema (visible at /docs) for free.

  If a request body fails validation, FastAPI returns a 422 Unprocessable
  Entity response with field-level error detail — no manual validation code
  required.

Interview note on validation choices:
  - question: min_length=3 rejects nonsense like "?" but allows short
    questions ("What is MFA?"). max_length=500 prevents prompt-injection
    via extremely long inputs overwhelming the LLM context.
  - top_k: bounded 1–20. Values above ~20 rarely improve results and
    increase prompt size and LLM cost.
  - We deliberately do NOT validate document content here — that belongs
    in the service layer where we have the actual parsed bytes.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /documents/upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    document_id: str = Field(description="Stable identifier derived from the filename.")
    filename: str
    chunks_created: int
    status: str = Field(description="'indexed' on success.")


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="Natural-language question to answer from the document store.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Higher values provide more context "
                    "but increase LLM prompt size and cost.",
    )


class SourceReference(BaseModel):
    document: str = Field(description="Filename of the source document.")
    page: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    retrieval_latency_ms: float = Field(
        description="Time from query embedding to Qdrant result, in milliseconds."
    )
    generation_latency_ms: float = Field(
        description="Time from prompt construction to LLM response, in milliseconds."
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when all dependencies are reachable.")
    pinecone: str = Field(description="'connected' or 'unreachable'.")
    embedding_model: str
    llm_model: str


# ---------------------------------------------------------------------------
# Error response (used by exception handlers — not a route model)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    error_type: str