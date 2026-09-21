"""
retriever.py — Accept a natural-language query, embed it, search Qdrant,
and return structured retrieval results.

This module is the bridge between the user's question and the vector store.
It deliberately does not call the LLM — retrieval and generation are kept
separate so each can be tested, measured, and improved independently.

Retrieval flow:
  1. User query string → embed_query() → query vector (384 floats)
  2. query vector → Qdrant similarity_search() → top-K ScoredPoints
  3. ScoredPoints → RetrievedChunk list (normalised structure for generator)

Why return a structured dataclass instead of raw Qdrant objects?
  Decouples the generator from Qdrant's SDK types. If we switch to a
  different vector DB, only this file changes — the generator is unaffected.

Interview note on retrieval relevance:
  Cosine similarity gives a score but not a guarantee of relevance.
  A score of 0.85 means the query and chunk vectors are geometrically
  close, but "geometric closeness" in embedding space reflects statistical
  co-occurrence in training data, not logical relevance to the question.
  Techniques to improve relevance:
    - Reranking: pass top-K through a cross-encoder that scores relevance
      directly (more expensive but more accurate).
    - Hybrid search: combine dense vector search with BM25 keyword search.
    - Metadata filtering: restrict search to specific documents.
  We don't implement these in the MVP but they are the natural next step.
"""

import logging
from dataclasses import dataclass

from app.embeddings.embedding_service import embed_query
from app.vectorstore.qdrant_service import similarity_search, get_client

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
        qdrant_url: str,
        collection_name: str,
        embedding_model: str,
        top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Embed a query and retrieve the most relevant chunks from Qdrant.

    Args:
        query: the user's natural-language question.
        qdrant_url: URL of the Qdrant instance.
        collection_name: target Qdrant collection.
        embedding_model: must match the model used during ingestion.
        top_k: number of chunks to return.

    Returns:
        List of RetrievedChunk ordered by descending similarity score.

    Raises:
        ValueError: if query is empty.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Step 1: embed the query using the same model used for document chunks
    query_vector = embed_query(query, embedding_model)
    logger.info(
        "Query embedded (dim=%d). Searching collection '%s' for top-%d.",
        len(query_vector),
        collection_name,
        top_k,
    )

    # Step 2: vector similarity search in Qdrant
    client = get_client(qdrant_url)
    scored_points = similarity_search(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
    )

    # Step 3: map Qdrant ScoredPoint objects to our clean dataclass
    results = []
    for point in scored_points:
        payload = point.payload or {}
        results.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                filename=payload.get("filename", "unknown"),
                page=payload.get("page", 0),
                chunk_index=payload.get("chunk_index", -1),
                score=round(point.score, 4),
            )
        )

    if not results:
        logger.warning("No results returned for query: '%s'", query)

    return results