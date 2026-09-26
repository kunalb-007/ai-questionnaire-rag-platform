"""
config.py — Central configuration loaded from environment variables.

Changes from original:
  - Removed Qdrant settings
  - Added Pinecone settings (api_key, index_name, cloud, region)
  - Changed embedding model to text-embedding-3-small (dim=1536)
  - Added embedding_dimension (must match Pinecone index)
  - Added rerank_candidates (fetch_k before cross-encoder reranking)
  - Added request_timeout and max_retries (used by query_service)
  - Added log_level (used by main.py)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Pinecone ---
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str          # "aws" or "gcp"
    pinecone_region: str         # e.g. "us-east-1"

    # --- LLM ---
    llm_api_key: str             # OpenAI API key (used for both LLM + embeddings)
    llm_model: str               # e.g. "gpt-4o-mini"
    llm_base_url: str

    # --- Embeddings ---
    embedding_model: str         # "text-embedding-3-small"
    embedding_dimension: int     # 1536 for text-embedding-3-small

    # --- Retrieval ---
    top_k: int                   # final results to return
    rerank_candidates: int       # candidates fetched before cross-encoder reranking
    use_hybrid_search: bool      # toggle hybrid vs dense-only
    use_reranking: bool          # toggle cross-encoder reranking

    # --- Chunking ---
    chunk_size: int
    chunk_overlap: int

    # --- App ---
    request_timeout: int
    max_retries: int
    log_level: str


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing. "
            f"Copy .env.example to .env and fill in values."
        )
    return value


def load_settings() -> Settings:
    return Settings(
        # Pinecone
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "enterprise-docs"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),

        # LLM
        llm_api_key=_require("OPENROUTER_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),

        # Embeddings
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),

        # Retrieval
        top_k=int(os.getenv("TOP_K", "5")),
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
        use_hybrid_search=os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true",
        use_reranking=os.getenv("USE_RERANKING", "true").lower() == "true",

        # Chunking
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),

        # App
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = load_settings()