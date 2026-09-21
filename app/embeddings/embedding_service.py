"""
embedding_service.py — Generate dense vector embeddings using FastEmbed.

Embedding model:
    BAAI/bge-small-en-v1.5

Output dimension:
    384

Backend:
    FastEmbed (ONNX Runtime)

Why FastEmbed:
    - Runs locally
    - Does not require PyTorch
    - Suitable for semantic search
    - Efficient CPU inference
    - Compatible with Qdrant

Important:
    The same embedding model must be used for both:
    1. Document ingestion
    2. User query embedding

The Qdrant collection dimension must match the embedding dimension.
"""

import logging
from functools import lru_cache

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors.
EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model(model_name: str) -> TextEmbedding:
    """
    Load and cache the embedding model.

    lru_cache ensures that the model is loaded only once
    per application process.

    The model is downloaded from Hugging Face during
    the first execution and cached locally.
    """

    logger.info("Loading embedding model: %s", model_name)

    model = TextEmbedding(
        model_name=model_name
    )

    logger.info(
        "Embedding model loaded successfully: %s",
        model_name
    )

    return model


# ---------------------------------------------------------
# Text Embeddings
# ---------------------------------------------------------

def embed_texts(
        texts: list[str],
        model_name: str = DEFAULT_EMBEDDING_MODEL
) -> list[list[float]]:
    """
    Generate embeddings for multiple text strings.

    Args:
        texts:
            List of input text strings.

        model_name:
            FastEmbed model identifier.

    Returns:
        A list of embedding vectors.

        Example:
            [
                [0.12, -0.04, ...],
                [0.08, 0.11, ...]
            ]

    Notes:
        - One vector is generated for each input text.
        - The selected model produces 384-dimensional vectors.
        - The model is cached after the first load.
    """

    if not texts:
        return []

    # Validate input
    cleaned_texts = []

    for text in texts:
        if not isinstance(text, str):
            raise TypeError(
                "Every item in texts must be a string."
            )

        if not text.strip():
            raise ValueError(
                "Input text cannot be empty."
            )

        cleaned_texts.append(text)

    model = _load_model(model_name)

    logger.debug(
        "Generating embeddings for %d texts.",
        len(cleaned_texts)
    )

    # FastEmbed returns a generator of NumPy arrays.
    embedding_generator = model.embed(
        cleaned_texts
    )

    embeddings = [
        vector.tolist()
        for vector in embedding_generator
    ]

    # Defensive validation
    for vector in embeddings:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Unexpected embedding dimension: {len(vector)}. "
                f"Expected: {EMBEDDING_DIMENSION}."
            )

    logger.debug(
        "Generated %d embeddings with dimension %d.",
        len(embeddings),
        EMBEDDING_DIMENSION
    )

    return embeddings


# ---------------------------------------------------------
# Query Embedding
# ---------------------------------------------------------

def embed_query(
        query: str,
        model_name: str = DEFAULT_EMBEDDING_MODEL
) -> list[float]:
    """
    Generate an embedding for a single user query.

    The same model used during document ingestion must
    be used for query embedding.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    if not query.strip():
        raise ValueError(
            "Query text cannot be empty."
        )

    vectors = embed_texts(
        texts=[query],
        model_name=model_name
    )

    return vectors[0]