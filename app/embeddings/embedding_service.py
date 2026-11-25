"""
embedding_service.py — OpenRouter embedding service.

Model:
    nvidia/nemotron-3.5-embed-1b:free

The same embedding model must be used for ingestion and query embedding.
Changing models requires re-ingestion.
"""

import logging
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        _client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts and return their vectors."""

    if not texts:
        return []

    client = _get_client()

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    # Preserve input ordering even if response ordering differs.
    embeddings = [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]

    if embeddings and len(embeddings[0]) != settings.embedding_dimension:
        raise ValueError(
            f"Expected embedding dimension {settings.embedding_dimension}, "
            f"received {len(embeddings[0])}."
        )

    logger.debug(
        "Generated %d embeddings (dim=%d).",
        len(embeddings),
        len(embeddings[0]) if embeddings else 0,
    )

    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""

    if not query.strip():
        raise ValueError("Query text cannot be empty.")

    return embed_texts([query])[0]