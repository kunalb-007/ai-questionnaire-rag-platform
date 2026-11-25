"""
pipeline.py — End-to-end RAG pipeline CLI entry point.

Run this file directly to ingest a document or query the pipeline
without starting the FastAPI server. Useful for testing and demos.

Usage:
  # Ingest a document
  python pipeline.py ingest path/to/policy.pdf

  # Ask a question
  python pipeline.py query "What is the password policy?"

  # Ingest then immediately query
  python pipeline.py ingest path/to/policy.pdf --query "What is the MFA policy?"

Ingestion pipeline (shown step-by-step):
  parse → clean → chunk → embed (OpenAI) → upsert (Pinecone)

Query pipeline:
  Delegates to run_query() in query_service so retry logic, error
  handling, and source deduplication are identical to the API path.
"""

import argparse
import logging
import sys

from app.config import settings
from app.ingestion.parser import parse_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedding_service import embed_texts
from app.vectorstore.pinecone_service import ensure_index, insert_chunks
from app.services.query_service import run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# INGESTION PIPELINE
# ---------------------------------------------------------------------------

def ingest_document(file_path: str) -> None:
    """
    Full ingestion pipeline: file → Pinecone.

    Steps:
      1. Parse   — extract raw text and page metadata from the file.
      2. Clean   — normalise whitespace, rejoin hyphenated line-breaks.
      3. Chunk   — split into overlapping character-based chunks.
      4. Embed   — generate dense vectors via OpenAI text-embedding-3-small.
      5. Store   — upsert into Pinecone with deterministic chunk IDs.

    Re-uploading the same file is safe: insert_chunks uses deterministic
    IDs derived from (doc_id, chunk_index) so upsert overwrites existing
    vectors instead of creating duplicates.
    """
    print(f"\n{'=' * 60}")
    print(f"  INGESTION PIPELINE")
    print(f"  File: {file_path}")
    print(f"{'=' * 60}\n")

    # --- Step 1: Parse ---
    print("Step 1/5 · Parsing document...")
    try:
        pages = parse_document(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ✗ Parse failed: {exc}")
        sys.exit(1)
    print(f"  ✓ Extracted {len(pages)} page(s).\n")

    # --- Step 2: Clean ---
    print("Step 2/5 · Cleaning text...")
    pages = clean_pages(pages)
    if not pages:
        print("  ✗ No text remained after cleaning. Document may be empty.")
        sys.exit(1)
    print(f"  ✓ {len(pages)} page(s) retained after cleaning.\n")

    # --- Step 3: Chunk ---
    print(
        f"Step 3/5 · Chunking "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})..."
    )
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        print("  ✗ No chunks produced. Document may be too short.")
        sys.exit(1)
    print(f"  ✓ {len(chunks)} chunk(s) produced.\n")

    # --- Step 4: Embed ---
    print(f"Step 4/5 · Generating embeddings (model={settings.embedding_model})...")
    try:
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)
    except Exception as exc:
        print(f"  ✗ Embedding failed: {exc}")
        sys.exit(1)
    print(
        f"  ✓ {len(embeddings)} embedding(s) generated "
        f"(dim={len(embeddings[0]) if embeddings else 0}).\n"
    )

    # --- Step 5: Store in Pinecone ---
    print(f"Step 5/5 · Upserting into Pinecone (index='{settings.pinecone_index_name}')...")
    try:
        ensure_index()
        insert_chunks(chunks, embeddings)
    except Exception as exc:
        print(f"  ✗ Pinecone upsert failed: {exc}")
        sys.exit(1)
    print(f"  ✓ {len(chunks)} chunk(s) stored.\n")

    print("=" * 60)
    print(f"  Ingestion complete: {file_path}")
    print(f"  Pinecone index '{settings.pinecone_index_name}' is ready to query.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# QUERY PIPELINE
# ---------------------------------------------------------------------------

def query_pipeline(question: str) -> None:
    """
    Full query pipeline: question → cited answer.

    Delegates to run_query() from query_service so this CLI uses the
    exact same retrieval, generation, retry, and error-handling logic
    as the API — no duplicated code, no drift risk.

    run_query() returns:
      answer               — LLM-generated cited answer string
      sources              — deduplicated list of {document, page}
      retrieval_latency_ms — time from query embed to Pinecone result
      generation_latency_ms — time from prompt build to LLM response
    """
    print(f"\n{'=' * 60}")
    print(f"  QUERY PIPELINE")
    print(f"  Question: {question}")
    print(f"{'=' * 60}\n")

    try:
        result = run_query(question=question, top_k=settings.top_k)
    except Exception as exc:
        print(f"  ✗ Query failed: {exc}")
        sys.exit(1)

    print(f"  Retrieval : {result['retrieval_latency_ms']:.0f} ms")
    print(f"  Generation: {result['generation_latency_ms']:.0f} ms\n")

    print("=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(result["answer"])

    if result["sources"]:
        print("\n" + "-" * 60)
        print("  SOURCES")
        print("-" * 60)
        for src in result["sources"]:
            print(f"  • {src['document']}  |  Page {src['page']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Security Questionnaire RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py ingest docs/password_policy.pdf
  python pipeline.py query "What is the minimum password length?"
  python pipeline.py ingest docs/policy.pdf --query "What is the MFA policy?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into Pinecone.")
    ingest_parser.add_argument("file_path", help="Path to a PDF or TXT file.")
    ingest_parser.add_argument(
        "--query", "-q",
        help="Optional: run a query immediately after ingestion.",
        default=None,
    )

    query_parser = subparsers.add_parser(
        "query", help="Ask a question against ingested documents."
    )
    query_parser.add_argument("question", help="Natural language question.")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_document(args.file_path)
        if args.query:
            query_pipeline(args.query)

    elif args.command == "query":
        query_pipeline(args.question)


if __name__ == "__main__":
    main()