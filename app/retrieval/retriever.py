"""
retriever.py — Accept a query, run hybrid search, optional reranking,
return structured RetrievedChunk list.

Changes from original:
  - Qdrant similarity_search → Pinecone hybrid_search
  - embed_query no longer called here (pinecone_service handles it internally)
  - Hybrid search + cross-encoder reranking controlled by config flags
  - RetrievedChunk dataclass unchanged — generator is unaffected

Retrieval flow:
  1. query string
  2. → pinecone_service.hybrid_search()
       a. OpenAI dense embedding
       b. BM25 sparse encoding
       c. Pinecone RRF combination
       d. (optional) cross-encoder reranking
  3. → list[RetrievedChunk]
"""

import logging
from dataclasses import dataclass

from app.config import settings
from app.vectorstore.pinecone_service import hybrid_search

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int
    chunk_index: int
    score: float


def retrieve(
        query: str,
        top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a query using hybrid search.

    Args:
        query  : user's natural-language question
        top_k  : number of final results to return

    Returns:
        List of RetrievedChunk ordered by relevance (best first).
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    logger.info(
        "Retrieving top-%d chunks (hybrid=%s, rerank=%s).",
        top_k, settings.use_hybrid_search, settings.use_reranking,
    )

    raw_results = hybrid_search(
        query=query,
        top_k=top_k,
        rerank=settings.use_reranking,
        fetch_k=settings.rerank_candidates,
    )

    chunks = [
        RetrievedChunk(
            text=r["text"],
            filename=r["filename"],
            page=r["page"],
            chunk_index=r["chunk_index"],
            score=r["score"],
        )
        for r in raw_results
    ]

    if not chunks:
        logger.warning("No results returned for query: '%s'", query[:80])

    return chunks