"""
query_service.py — Orchestrates retrieval and LLM generation for a user query.

Phase 3 — Error handling:
  We distinguish three failure modes:
    1. Retrieval failure (Qdrant down / empty collection): caller returns 503.
    2. LLM timeout: caller returns 504 with a meaningful message.
    3. LLM API error (rate limit, auth): caller returns 502.
  We do NOT catch every exception silently — unhandled exceptions bubble up
  to FastAPI's global exception handler which logs them and returns 500.

Phase 7 — Latency:
  retrieval_latency_ms: embed query + Qdrant search.
  generation_latency_ms: prompt construction + LLM round-trip.
  These are included in the API response so clients can observe where
  time is spent.

Bounded retries (Phase 3):
  We retry LLM calls up to settings.max_retries times on transient errors
  (timeout, connection reset). We do NOT retry on 4xx errors (auth, bad
  request) because retrying will not fix them.

  Why bounded? An unbounded retry loop can transform a brief outage into
  a traffic amplification problem. Two retries is sufficient to handle
  a single transient timeout without compounding the problem.
"""

import logging
import time

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.retrieval.retriever import retrieve, RetrievedChunk
from app.generation.answer_generator import build_context_block, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


class LLMTimeoutError(RuntimeError):
    pass


class LLMAPIError(RuntimeError):
    pass


def run_query(question: str, top_k: int) -> dict:
    """
    Retrieve relevant chunks and generate a cited answer.

    Args:
        question: the user's validated question string.
        top_k: number of chunks to retrieve (validated by the route schema).

    Returns:
        dict with answer, sources, retrieval_latency_ms, generation_latency_ms.

    Raises:
        RetrievalError: Qdrant is unreachable or the collection does not exist.
        LLMTimeoutError: LLM did not respond within request_timeout seconds.
        LLMAPIError: LLM returned a non-transient API error.
    """
    # --- Retrieval ---
    t0 = time.perf_counter()
    try:
        chunks = retrieve(
            query=question,
            top_k=top_k,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Vector store retrieval failed: {exc}"
        ) from exc
    retrieval_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not chunks:
        logger.warning("Query returned zero chunks from Qdrant: '%s'", question[:80])
        # Return a graceful no-results answer rather than raising an error.
        # The collection may be empty or the question may be entirely off-topic.
        return {
            "answer": (
                "No relevant documents were found for your question. "
                "Please ensure the relevant documents have been uploaded."
            ),
            "sources": [],
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": 0.0,
        }

    # --- Generation with bounded retries ---
    t0 = time.perf_counter()
    answer_text = _generate_with_retry(question, chunks)
    generation_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Build deduplicated source list (same page may appear in multiple chunks)
    seen: set[tuple[str, int]] = set()
    unique_sources: list[dict] = []
    for chunk in chunks:
        key = (chunk.filename, chunk.page)
        if key not in seen:
            seen.add(key)
            unique_sources.append({"document": chunk.filename, "page": chunk.page})

    logger.info(
        "Query complete — retrieval=%.0fms generation=%.0fms sources=%d",
        retrieval_latency_ms,
        generation_latency_ms,
        len(unique_sources),
    )

    return {
        "answer": answer_text,
        "sources": unique_sources,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
    }


def _generate_with_retry(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Call the LLM with bounded retries on transient failures.

    Retries on : httpx.TimeoutException, httpx.ConnectError.
    No retry on: any other exception (auth errors, rate limits, bad requests).

    Total attempts = 1 initial + settings.max_retries retries.
    """
    context_block = build_context_block(chunks)
    human_content = (
        f"Context:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite sources inline):"
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        max_tokens=1024,
        timeout=settings.request_timeout,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    total_attempts = settings.max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = llm.invoke(messages)
            return str(response.content)

        except httpx.TimeoutException as exc:
            logger.warning(
                "LLM timeout on attempt %d/%d: %s", attempt, total_attempts, exc
            )
            if attempt == total_attempts:
                raise LLMTimeoutError(
                    f"LLM did not respond within {settings.request_timeout}s "
                    f"after {total_attempts} attempt(s)."
                ) from exc

        except httpx.ConnectError as exc:
            logger.warning(
                "LLM connection error on attempt %d/%d: %s", attempt, total_attempts, exc
            )
            if attempt == total_attempts:
                raise LLMAPIError(
                    f"Could not connect to LLM endpoint after "
                    f"{total_attempts} attempt(s): {exc}"
                ) from exc

        except Exception as exc:
            # Non-transient — do not retry
            logger.error("LLM API error (non-retryable): %s", exc)
            raise LLMAPIError(f"LLM API returned an error: {exc}") from exc

    # Unreachable — satisfies the type checker
    raise LLMAPIError("LLM generation failed after all retry attempts.")