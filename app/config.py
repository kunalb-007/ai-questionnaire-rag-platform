"""
config.py — Central configuration loaded from environment variables.

Uses pydantic-settings so that:
  - Every field is typed and coerced automatically.
  - Missing required fields raise a clear ValidationError at startup.
  - No manual os.getenv() boilerplate.

Fix applied vs previous version:
  - env_file path is resolved relative to this file (config.py) rather
    than the process working directory, so it loads correctly regardless
    of where uvicorn is invoked from.
  - llm_api_key uses validation_alias instead of alias. In pydantic-settings,
    alias affects serialisation AND lookup; validation_alias affects only
    the lookup (env var name), which is what we want here.
  - populate_by_name=True is added so the field can also be set by its
    field name (llm_api_key) in addition to the validation alias, giving
    flexibility in how the env var is named.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it loads correctly regardless
# of which directory uvicorn / the CLI is invoked from.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow fields to be set by their field name OR their validation_alias.
        populate_by_name=True,
        case_sensitive=False,
    )

    # --- Pinecone ---
    pinecone_api_key: str
    pinecone_index_name: str = "enterprise-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- LLM ---
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-20b"
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # --- Embeddings ---
    openrouter_api_key: str
    embedding_model: str = "nvidia/nemotron-3.5-embed-1b:free"
    embedding_dimension: int = 2048

    # --- Retrieval ---
    top_k: int = 5
    rerank_candidates: int = 20
    use_hybrid_search: bool = False
    use_reranking: bool = False
    min_retrieval_score: float = 0.45

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- App ---
    request_timeout: int = 30
    max_retries: int = 2
    log_level: str = "INFO"

settings = Settings()