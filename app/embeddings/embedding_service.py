"""
embedding_service.py — OpenAI text-embedding-3-small wrapper.

Why text-embedding-3-small over local FastEmbed?
  - Higher quality embeddings (trained on much larger corpus)
  - No local model download / GPU / ONNX runtime needed
  - Easy to swap to text-embedding-3-large for better quality
  - Dimension: 1536

Trade-off: requires API call per batch (latency + cost).
For demo scale: negligible. For production: batch aggressively.

The same model MUST be used for ingestion AND query embedding.
Changing models requires full re-ingestion.
"""

import logging
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 1536-dim vectors."""
    if not texts:
        return []

    client = _get_client()
    response = client.embeddings.create(
        model=settings.embedding_model,  # "text-embedding-3-small"
        input=texts,
    )
    embeddings = [item.embedding for item in response.data]

    logger.debug("Generated %d embeddings (dim=%d).", len(embeddings),
                 len(embeddings[0]) if embeddings else 0)
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    if not query.strip():
        raise ValueError("Query text cannot be empty.")
    return embed_texts([query])[0]