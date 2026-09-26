"""
pinecone_service.py — Pinecone vector store operations.

Hybrid search implementation:
  Dense  : text-embedding-3-small (OpenAI) — semantic meaning
  Sparse : BM25 (keyword matching)         — exact term overlap
  Combined with Reciprocal Rank Fusion (RRF) — no score scaling needed

Cross-encoder reranking (medium priority):
  After hybrid retrieval we rerank the candidates with a cross-encoder.
  A cross-encoder reads (query, chunk) together — more accurate than
  bi-encoder cosine similarity but too slow to run over entire corpus.
  So: retrieve top-20 with hybrid search → rerank → return top-K.

Why Pinecone over Qdrant?
  Pinecone is fully managed — no Docker, no infra, free tier available.
  Better for demos: no local service to start.
"""

import logging
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
# from sentence_transformers import CrossEncoder
from openai import OpenAI

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients (lazy singletons)
# ---------------------------------------------------------------------------

_pinecone_client: Pinecone | None = None
_openai_client: OpenAI | None = None
_bm25_encoder: BM25Encoder | None = None
# _cross_encoder: CrossEncoder | None = None
_cross_encoder = None

def _get_pinecone() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=settings.llm_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openai_client


def _get_bm25() -> BM25Encoder:
    """
    BM25Encoder from pinecone-text.
    For demo: use default pre-fitted encoder.
    In production: fit on your corpus → encoder.fit(corpus_texts) → save.
    """
    global _bm25_encoder
    if _bm25_encoder is None:
        _bm25_encoder = BM25Encoder().default()  # pre-fitted on MS MARCO
    return _bm25_encoder


def _get_cross_encoder() -> CrossEncoder:
    """
    Cross-encoder for reranking.
    ms-marco-MiniLM-L-6-v2 is small, fast, and good enough for demo.
    """
    global _cross_encoder
#     if _cross_encoder is None:
#         _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
     if _cross_encoder is None:
         from sentence_transformers import CrossEncoder

         _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
         )

    return _cross_encoder


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate dense embeddings using OpenAI text-embedding-3-small.
    Dimension: 1536
    """
    if not texts:
        return []
    client = _get_openai()
    response = client.embeddings.create(
        model=settings.embedding_model,   # "text-embedding-3-small"
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Index Management
# ---------------------------------------------------------------------------

def ensure_index() -> None:
    """
    Create Pinecone index if it doesn't exist.
    Uses serverless spec (free tier friendly).

    Pinecone serverless: no pods to manage, scales automatically,
    pay-per-query model. Good for demos.
    """
    pc = _get_pinecone()
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name in existing:
        logger.info("Pinecone index '%s' already exists.", settings.pinecone_index_name)
        return

    logger.info("Creating Pinecone index '%s'.", settings.pinecone_index_name)
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,   # 1536 for text-embedding-3-small
        metric="dotproduct",                       # required for hybrid search
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,         # "aws"
            region=settings.pinecone_region,       # "us-east-1"
        ),
    )


def get_index():
    return _get_pinecone().Index(settings.pinecone_index_name)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def insert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """
    Insert chunks with both dense and sparse vectors into Pinecone.

    Each vector record:
      id     : unique UUID
      values : dense embedding (1536-dim)
      sparse_values : BM25 sparse vector (for hybrid search)
      metadata : text, filename, page, chunk_index
    """
    import uuid

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count must match.")

    bm25 = _get_bm25()
    texts = [chunk.text for chunk in chunks]
    sparse_vectors = bm25.encode_documents(texts)

    vectors = []
    for chunk, dense_vec, sparse_vec in zip(chunks, embeddings, sparse_vectors):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": dense_vec,
            "sparse_values": sparse_vec,
            "metadata": {
                "text": chunk.text,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "doc_id": chunk.doc_id,
            },
        })

    index = get_index()
    # Pinecone recommends batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    logger.info("Inserted %d vectors into Pinecone index '%s'.",
                len(vectors), settings.pinecone_index_name)


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    fetch_k: int = 20,
) -> list[dict]:
    """
    Hybrid search = dense (semantic) + sparse (BM25 keyword) search.

    How it works:
      1. Encode query → dense vector (OpenAI) + sparse vector (BM25)
      2. Send both to Pinecone in a single query
      3. Pinecone combines scores using Reciprocal Rank Fusion (RRF)
      4. Optionally rerank with cross-encoder

    Why hybrid over dense-only?
      Dense search finds semantically similar text but can miss exact
      keywords ("MFA", "SOC2", version numbers). BM25 catches exact terms.
      Combining both covers semantic AND keyword relevance.

    Why RRF for combining scores?
      Dense and sparse scores are on different scales.
      RRF combines rankings (not raw scores) — no scaling/tuning needed.

    Args:
        query    : user's natural language question
        top_k    : final number of results to return
        rerank   : whether to apply cross-encoder reranking
        fetch_k  : how many candidates to fetch before reranking
    """
    # We fetch more candidates when reranking (rerank needs a larger pool)
    candidates_to_fetch = fetch_k if rerank else top_k

    # Step 1: encode query (dense + sparse)
    dense_vector = embed_query(query)
    bm25 = _get_bm25()
    sparse_vector = bm25.encode_queries(query)

    # Step 2: hybrid query — Pinecone handles RRF internally
    index = get_index()
    response = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=candidates_to_fetch,
        include_metadata=True,
    )

    matches = response.get("matches", [])

    if not matches:
        logger.warning("Hybrid search returned 0 results for query: '%s'", query[:80])
        return []

    # Step 3: cross-encoder reranking (medium priority feature)
    if rerank and len(matches) > 1:
        matches = _rerank(query, matches, top_k)
    else:
        matches = matches[:top_k]

    # Step 4: normalize to our standard result format
    results = []
    for match in matches:
        meta = match.get("metadata", {})
        results.append({
            "text": meta.get("text", ""),
            "filename": meta.get("filename", "unknown"),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", -1),
            "score": round(match.get("score", 0.0), 4),
        })

    logger.info(
        "Hybrid search complete — returned %d results (rerank=%s).",
        len(results), rerank,
    )
    return results


# ---------------------------------------------------------------------------
# Cross-encoder Reranking
# ---------------------------------------------------------------------------

def _rerank(query: str, matches: list[dict], top_k: int) -> list[dict]:
    """
    Rerank Pinecone candidates using a cross-encoder.

    Retrieve top-N candidates from Pinecone, score each
    (query, document) pair with the cross-encoder, then return
    the top-K candidates by rerank score.
    """
    cross_encoder = _get_cross_encoder()

    texts = [
        m.get("metadata", {}).get("text", "")
        for m in matches
    ]

    pairs = [(query, text) for text in texts]

    scores = cross_encoder.predict(pairs)

    logger.info(
        "Cross-encoder reranking: candidates=%d scores=%d",
        len(matches),
        len(scores),
    )

    # Create NEW normal dictionaries instead of modifying
    # Pinecone Match objects in-place.
    reranked_candidates = []

    for match, score in zip(matches, scores):
        metadata = match.get("metadata", {})

        reranked_candidates.append({
            "id": match.get("id"),
            "score": match.get("score", 0.0),
            "rerank_score": float(score),
            "metadata": metadata,
        })

    if len(reranked_candidates) != len(matches):
        raise RuntimeError(
            f"Cross-encoder returned {len(scores)} scores "
            f"for {len(matches)} candidates."
        )

    # Sort using our own normal dictionaries
    reranked_candidates.sort(
        key=lambda m: m["rerank_score"],
        reverse=True,
    )

    logger.debug(
        "Reranking complete — top score: %.4f, bottom score: %.4f",
        reranked_candidates[0]["rerank_score"]
        if reranked_candidates else 0,
        reranked_candidates[-1]["rerank_score"]
        if reranked_candidates else 0,
    )

    return reranked_candidates[:top_k]