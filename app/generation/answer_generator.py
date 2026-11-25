"""
answer_generator.py — Shared prompt constants and context formatting.

This module owns two things used by query_service.py and evaluate.py:
  1. SYSTEM_PROMPT        — the LLM's role and grounding rules.
  2. build_context_block  — formats retrieved chunks into a numbered,
                            source-labelled block the LLM reads for citations.

Why is generate_answer() removed?
  It was never called through the API path. query_service._generate_with_retry
  owns all LLM generation so retry logic, timeout handling, and error mapping
  stay in one place. Having two generation implementations that could silently
  drift is worse than having one that is easy to find and change.

RAG generation philosophy:
  The LLM's job is reading comprehension, not knowledge recall. The system
  prompt explicitly constrains it to the provided context. This reduces
  hallucination but does not eliminate it — the LLM can still blend context
  with training-data knowledge or misattribute content between sources.
  Do not claim this system is hallucination-proof.

Why cite sources?
  Enterprise users need to verify claims against the original document.
  Auditors need a traceable path from answer to source page.
  Citations also expose retrieval failures — if the cited chunk does not
  support the answer, something is wrong upstream.
"""

import logging

from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an enterprise security compliance assistant. \
Your task is to answer the user's question based ONLY on the context \
excerpts provided below.

Rules:
- If the answer is present in the context, provide it clearly and concisely.
- Always cite your sources using [Source: <filename>, Page <page>] inline.
- If the context does not contain enough information to answer the question,
  state clearly: "The provided documents do not contain sufficient information
  to answer this question."
- Do not use any knowledge from outside the provided context.
- Do not speculate or infer beyond what is explicitly stated.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Each chunk is labelled with its source filename and page number so the
    LLM can produce accurate inline citations. Without these labels the LLM
    has no way to know where a piece of text came from.

    Returns a fallback string when no chunks are provided so the LLM still
    invokes the insufficient-evidence rule rather than receiving an empty
    context block.

    Interview note: labelling chunks with their source *inside the prompt*
    is what enables source-grounded answers. The label format matches what
    the system prompt tells the LLM to cite — keeping them in sync matters.
    """
    if not chunks:
        return "No relevant context was found in the document store."

    sections = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] Source: {chunk.filename}, Page {chunk.page} "
            f"(score: {chunk.score:.4f})"
        )
        sections.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(sections)