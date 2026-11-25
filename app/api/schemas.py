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
    status: str = Field(
        description=(
            "'answered' — LLM generated a grounded answer. "
            "'insufficient_evidence' — top retrieval score was below the "
            "configured threshold; LLM was not called. "
            "'no_documents' — zero chunks returned from the vector store."
        )
    )
    top_retrieval_score: float | None = Field(
        description=(
            "Highest similarity score among retrieved chunks. "
            "None when no chunks were returned. "
            "Do not treat this as an answer confidence score — "
            "it measures vector similarity, not factual correctness."
        )
    )
    retrieval_latency_ms: float = Field(
        description="Time from query embedding to Pinecone result, in milliseconds."
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

# ---------------------------------------------------------------------------
# /questionnaires/process and /questionnaires/export
# ---------------------------------------------------------------------------

class QuestionResult(BaseModel):
    row: int = Field(description="1-based row number from the uploaded CSV.")
    question: str
    answer: str
    status: str = Field(
        description=(
            "'answered' — grounded answer generated. "
            "'insufficient_evidence' — top retrieval score below threshold. "
            "'no_documents' — vector store returned zero chunks."
        )
    )
    sources: list[SourceReference]
    top_retrieval_score: float | None = Field(
        description="Highest similarity score among retrieved chunks. None if no chunks found."
    )


class QuestionnaireResponse(BaseModel):
    total_questions: int = Field(description="Total questions processed from the CSV.")
    answered: int = Field(description="Questions with status 'answered'.")
    insufficient_evidence: int = Field(
        description="Questions where retrieval score was below the configured threshold."
    )
    no_documents: int = Field(description="Questions where the vector store returned nothing.")
    results: list[QuestionResult]