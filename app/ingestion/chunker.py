"""
chunker.py — Split cleaned page text into overlapping chunks for embedding.

Why chunking is required:
  LLMs have a fixed context window. Embedding models also have token limits
  (all-MiniLM-L6-v2 is capped at 256 word-piece tokens). Sending a full
  50-page policy document as one unit would either be truncated or exceed
  limits. Chunking divides text into units small enough to embed faithfully.

Why chunk size matters for retrieval:
  - Too small (e.g., 128 chars): chunks lack enough context to be meaningful.
    "The password must be..." retrieved in isolation tells us nothing about
    *which* password policy or *what* the full requirement is.
  - Too large (e.g., 2048 chars): a chunk may cover multiple topics.
    Embedding the mixed content produces a vector that is "average" of all
    topics, making it less likely to surface for any specific query.
  - Sweet spot: 400–600 characters for policy documents is a reasonable start.

Why overlap matters:
  Without overlap, a sentence that spans a chunk boundary is split. The first
  half is embedded with one context, the second half with another. Overlap
  ensures boundary content appears in at least one complete chunk.

Limitations of character-based chunking:
  - Splits on character count, not semantic boundaries. A paragraph discussing
    two topics may be split mid-sentence.
  - RecursiveCharacterTextSplitter tries to split on paragraph → sentence →
    word → character boundaries (in that order), which mitigates this.
  - It does NOT understand that "Section 4.2" headings belong with the
    text that follows — they may end up in different chunks.

Interview note: Semantic/topic-based chunking (e.g., splitting on heading
patterns or using embeddings to detect topic shifts) is more accurate but
significantly more complex. Start here, understand the trade-offs, then
extend.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    filename: str
    page: int
    chunk_index: int
    doc_id: str  # filename used as document identifier


def chunk_pages(
        pages: list[dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split each page's text into overlapping chunks.

    Args:
        pages: list of PageData dicts from the parser/cleaner.
        chunk_size: maximum character count per chunk.
        chunk_overlap: characters shared between adjacent chunks.

    Returns:
        Flat list of Chunk objects, ordered by page then chunk index.

    Interview note on RecursiveCharacterTextSplitter:
      It splits text using a hierarchy of separators:
        ["\\n\\n", "\\n", " ", ""]
      It tries "\\n\\n" first (paragraph break). If a paragraph is still
      larger than chunk_size, it falls back to "\\n", then " ", then
      character-by-character. This is more semantically aware than a naive
      character split.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Keep separators attached to the preceding text so chunks
        # don't start with a bare newline
        keep_separator=False,
    )

    all_chunks: list[Chunk] = []
    global_chunk_index = 0

    for page in pages:
        raw_chunks = splitter.split_text(page["text"])

        for local_idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    filename=page["filename"],
                    page=page["page"],
                    chunk_index=global_chunk_index,
                    doc_id=page["filename"],
                )
            )
            global_chunk_index += 1

    logger.info(
        "Chunking complete: %d chunks from %d page(s) "
        "(chunk_size=%d, overlap=%d).",
        len(all_chunks),
        len(pages),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks