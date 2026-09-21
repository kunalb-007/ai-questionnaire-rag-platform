"""
pipeline.py — End-to-end RAG pipeline entry point.

Run this file directly to:
  1. Ingest a document (parse → clean → chunk → embed → store).
  2. Query the document store and get a cited answer.

Usage:
  # Ingest a document
  python pipeline.py ingest path/to/policy.pdf

  # Ask a question
  python pipeline.py query "What is the password policy?"

  # Ingest then immediately query (useful for demos)
  python pipeline.py ingest path/to/policy.pdf --query "What is the MFA policy?"

This file is intentionally verbose so every step of the RAG pipeline
is visible and traceable. In an interview, walk through each step in order.
"""

import argparse
import logging
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict
from app.ingestion.parser import parse_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedding_service import embed_texts, EMBEDDING_DIMENSION
from app.vectorstore.qdrant_service import get_client, ensure_collection, insert_chunks
from app.retrieval.retriever import retrieve
from app.generation.answer_generator import generate_answer

# Configure logging — INFO level shows the pipeline steps clearly.
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
    Full ingestion pipeline: file → Qdrant.

    Steps:
      1. Parse  — extract raw text and page metadata from file.
      2. Clean  — normalise whitespace, remove noise.
      3. Chunk  — split into overlapping chunks for embedding.
      4. Embed  — generate vectors for each chunk.
      5. Store  — upsert into Qdrant with payload metadata.
    """
    print(f"\n{'='*60}")
    print(f"  INGESTION PIPELINE")
    print(f"  File: {file_path}")
    print(f"{'='*60}\n")

    # --- Step 1: Parse ---
    print("Step 1/5 · Parsing document...")
    pages = parse_document(file_path)
    print(f"  ✓ Extracted {len(pages)} page(s).\n")

    # --- Step 2: Clean ---
    print("Step 2/5 · Cleaning text...")
    pages = clean_pages(pages)
    print(f"  ✓ {len(pages)} page(s) retained after cleaning.\n")

    # --- Step 3: Chunk ---
    print(f"Step 3/5 · Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"  ✓ {len(chunks)} chunk(s) produced.\n")

    if not chunks:
        print("ERROR: No chunks produced. Document may be empty or unreadable.")
        sys.exit(1)

    # --- Step 4: Embed ---
    print(f"Step 4/5 · Generating embeddings (model={settings.embedding_model})...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, settings.embedding_model)
    print(f"  ✓ {len(embeddings)} embedding(s) generated (dim={EMBEDDING_DIMENSION}).\n")

    # --- Step 5: Store in Qdrant ---
    print(f"Step 5/5 · Storing in Qdrant (collection='{settings.collection_name}')...")
    client = get_client(settings.qdrant_url)
    ensure_collection(client, settings.collection_name, EMBEDDING_DIMENSION)
    insert_chunks(client, settings.collection_name, chunks, embeddings)
    print(f"  ✓ {len(chunks)} chunk(s) stored.\n")

    print("=" * 60)
    print(f"  Ingestion complete: {file_path}")
    print(f"  Collection '{settings.collection_name}' is ready to query.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# QUERY PIPELINE
# ---------------------------------------------------------------------------

def query_pipeline(question: str) -> None:
    """
    Full query pipeline: question → cited answer.

    Steps:
      1. Embed query  — same model used during ingestion.
      2. Retrieve     — top-K similar chunks from Qdrant.
      3. Generate     — LLM reads chunks, produces cited answer.
      4. Display      — print answer + source metadata.
    """
    print(f"\n{'='*60}")
    print(f"  QUERY PIPELINE")
    print(f"  Question: {question}")
    print(f"{'='*60}\n")

    # --- Step 1 & 2: Retrieve ---
    print(f"Step 1/2 · Retrieving top-{settings.top_k} chunks from Qdrant...")
    retrieved_chunks = retrieve(
        query=question,
        qdrant_url=settings.qdrant_url,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )

    if not retrieved_chunks:
        print("  ✗ No relevant chunks found. Is the collection populated?\n")
        sys.exit(1)

    print(f"  ✓ {len(retrieved_chunks)} chunk(s) retrieved.\n")
    print("  Retrieved chunks (preview):")
    for i, chunk in enumerate(retrieved_chunks, 1):
        preview = chunk.text[:100].replace("\n", " ")
        print(f"    [{i}] score={chunk.score} | {chunk.filename} p{chunk.page} | \"{preview}...\"")

    # --- Step 3: Generate ---
    print(f"\nStep 2/2 · Generating answer (model={settings.llm_model})...")
    result = generate_answer(
        query=question,
        retrieved_chunks=retrieved_chunks,
        llm_api_key=settings.llm_api_key,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # --- Step 4: Display ---
    print("\n" + "=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(result.answer)

    print("\n" + "-" * 60)
    print("  SOURCES USED AS CONTEXT")
    print("-" * 60)
    for src in result.sources:
        print(
            f"  • {src['filename']} | Page {src['page']} "
            f"| Chunk #{src['chunk_index']} | Similarity: {src['score']}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise Document Intelligence RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py ingest docs/password_policy.pdf
  python pipeline.py query "What is the minimum password length?"
  python pipeline.py ingest docs/policy.pdf --query "What is the MFA policy?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest sub-command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into Qdrant.")
    ingest_parser.add_argument("file_path", help="Path to PDF, DOCX, or TXT file.")
    ingest_parser.add_argument(
        "--query", "-q",
        help="Optional: run a query immediately after ingestion.",
        default=None,
    )

    # query sub-command
    query_parser = subparsers.add_parser("query", help="Ask a question against ingested documents.")
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