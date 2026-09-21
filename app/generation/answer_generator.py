"""
answer_generator.py — Build the RAG prompt, call the LLM, return the answer
with source citations.

RAG generation philosophy:
  The LLM's job here is *reading comprehension*, not *knowledge recall*.
  We explicitly tell it to use only the provided context and to state when
  the answer is not present. This is critical for enterprise use — hallucinated
  compliance policy details could have serious consequences.

Important caveat on hallucination:
  Prompt instructions ("only use the context below") reduce hallucination
  but do not eliminate it. LLMs can still:
    - Blend provided context with training-data knowledge.
    - Confidently assert things not in the context.
    - Misread or misattribute content between sources.
  Do not claim this system is hallucination-proof. It is hallucination-
  *reduced* through grounding.

Why cite sources?
  - Enterprise users need to verify claims against the original document.
  - Auditors need a traceable path from answer to source.
  - Citations also expose retrieval failures — if the cited chunk doesn't
    support the answer, something is wrong upstream.

LangChain usage:
  We use ChatOpenAI from langchain_openai. It is OpenAI API-compatible,
  which means any OpenAI-compatible endpoint works (OpenAI, Ollama, vLLM,
  Groq, etc.) by changing LLM_BASE_URL in the config.

Interview note on prompt engineering:
  The prompt has three components:
    System — establishes the LLM's role and constraints.
    Context — the retrieved chunks, labelled with their source.
    Human  — the user's question.
  Keeping system instructions separate from context makes the prompt
  easier to iterate on without breaking the context injection logic.
"""

import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an enterprise document assistant. Your task is to answer
the user's question based ONLY on the context excerpts provided below.

Rules:
- If the answer is present in the context, provide it clearly and concisely.
- Always cite your sources using [Source: <filename>, Page <page>] inline.
- If the context does not contain enough information to answer the question,
  state clearly: "The provided documents do not contain sufficient information
  to answer this question."
- Do not use any knowledge from outside the provided context.
- Do not speculate or infer beyond what is explicitly stated.
"""


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[dict]  # [{filename, page, score, chunk_index}]
    query: str


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled with its source so the LLM can produce
    accurate inline citations.

    Interview note: labelling chunks with their source *in the prompt*
    is what enables source-grounded answers. Without labels, the LLM
    has no way to know which file or page a piece of text came from.
    """
    if not chunks:
        return "No relevant context was found in the document store."

    sections = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] Source: {chunk.filename}, Page {chunk.page} (score: {chunk.score})"
        sections.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(sections)


def generate_answer(
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        llm_api_key: str,
        llm_model: str,
        llm_base_url: str,
) -> GeneratedAnswer:
    """
    Generate a grounded answer using retrieved context and an LLM.

    Args:
        query: the user's original question.
        retrieved_chunks: top-K results from the retriever.
        llm_api_key: API key for the LLM endpoint.
        llm_model: model identifier (e.g., "gpt-4o-mini").
        llm_base_url: base URL for OpenAI-compatible API.

    Returns:
        GeneratedAnswer with answer text and source metadata list.
    """
    context_block = build_context_block(retrieved_chunks)

    human_message_content = (
        f"Context:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources inline):"
    )

    logger.info(
        "Sending prompt to LLM '%s' with %d context chunk(s).",
        llm_model,
        len(retrieved_chunks),
    )

    # LangChain's ChatOpenAI works with any OpenAI-compatible endpoint.
    # Setting base_url allows Ollama, vLLM, Groq, etc. as drop-in replacements.
    llm = ChatOpenAI(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        temperature=0,  # deterministic for factual Q&A
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    response = llm.invoke(messages)
    answer_text = response.content

    # Build source list from the chunks that were passed as context.
    # Note: we include ALL retrieved chunks as potential sources, not just
    # the ones the LLM happened to cite. In production you might parse the
    # LLM's inline citations to confirm which sources were actually used.
    sources = [
        {
            "filename": chunk.filename,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score,
        }
        for chunk in retrieved_chunks
    ]

    logger.info("Answer generated successfully.")
    return GeneratedAnswer(answer=answer_text, sources=sources, query=query)