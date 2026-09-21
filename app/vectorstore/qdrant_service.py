"""
qdrant_service.py — All Qdrant interactions: create collection, insert
vectors, and perform similarity search.

Why Qdrant over a relational database:
  A relational DB stores rows and supports exact lookups and range queries.
  It cannot efficiently answer "find the 5 rows whose text meaning is most
  similar to this query". Similarity search requires computing distances
  between high-dimensional vectors, which Qdrant optimises using HNSW
  (Hierarchical Navigable Small World) graphs — an approximate nearest-
  neighbour (ANN) algorithm that is both fast and memory-efficient.

Qdrant concepts:
  Collection — analogous to a table; holds vectors of a fixed dimension.
  Point     — one record: a vector + a payload (arbitrary JSON metadata).
  Payload   — the metadata we store alongside each vector (text, filename,
               page number, etc.). Returned with search results so we can
               cite the source.

Why persistent local storage:
  Running Qdrant with `docker run -v ./qdrant_storage:/qdrant/storage ...`
  means data survives container restarts. In production you'd point at a
  managed Qdrant cluster or Cloud.

Distance metric — Cosine:
  Chosen to match our L2-normalised embeddings. For normalised vectors,
  cosine similarity and dot-product produce identical rankings.

Interview note on metadata filtering:
  Qdrant supports payload filters (e.g., return only chunks from a specific
  filename). This allows per-document Q&A without re-ingesting. We don't
  expose this in the MVP but the payload fields are stored ready for it.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
)

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def get_client(qdrant_url: str) -> QdrantClient:
    """Create a Qdrant client pointed at the given URL."""
    logger.info("Connecting to Qdrant at %s", qdrant_url)
    return QdrantClient(url=qdrant_url)


def ensure_collection(
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Idempotent: safe to call on every startup.

    Args:
        vector_size: must match the embedding model output dimension.
                     Mismatches cause insertion errors.
    """
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        logger.info("Collection '%s' already exists — skipping creation.", collection_name)
        return

    logger.info(
        "Creating collection '%s' (dim=%d, metric=Cosine).",
        collection_name,
        vector_size,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def insert_chunks(
        client: QdrantClient,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
) -> None:
    """
    Insert chunk embeddings and their metadata into Qdrant.

    Each point stores:
      - vector: the embedding of the chunk text.
      - payload: chunk text, filename, page number, chunk index, doc_id.

    Payload is what gets returned at query time and used for citations.

    Args:
        chunks: list of Chunk objects from the chunker.
        embeddings: parallel list of embedding vectors (same order as chunks).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) and embedding count "
            f"({len(embeddings)}) must match."
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # unique ID per chunk
            vector=embedding,
            payload={
                "text": chunk.text,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "doc_id": chunk.doc_id,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # batch upload — Qdrant client handles chunking internally
    client.upload_points(collection_name=collection_name, points=points)

    logger.info(
        "Inserted %d points into collection '%s'.", len(points), collection_name
    )


def similarity_search(
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
) -> list[ScoredPoint]:
    """
    Retrieve the top-K most similar chunks to the query vector.

    Returns ScoredPoint objects which each contain:
      - score: cosine similarity (0–1 for normalised vectors)
      - payload: the metadata dict we stored at insertion time

    Interview note on top-K trade-offs:
      - High K (e.g., 20): more context for the LLM but the prompt grows
        and lower-ranked results may be irrelevant (diluting the answer).
      - Low K (e.g., 3): fast, focused, but risks missing the right chunk
        if the best match is ranked 4th.
      - Why similarity search can return irrelevant results: the model
        embeds *statistical co-occurrence patterns* from training data.
        "What is our MFA policy?" may surface a chunk about "policy
        violation penalties" because "policy" appears in both. The LLM
        must then ignore irrelevant chunks — which it sometimes fails to do.
    """
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    logger.info(
        "Similarity search returned %d result(s) (top_k=%d).", len(results), top_k
    )
    return results