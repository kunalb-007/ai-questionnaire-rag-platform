"""
questionnaire_service.py — Parse a CSV questionnaire and run each question
through the RAG pipeline in sequence.

Why a separate service module?
  Both the /process endpoint (returns JSON) and the /export endpoint
  (returns a CSV file) share identical parsing and batch-processing logic.
  Putting that logic here keeps both route handlers thin and makes the
  core business logic independently testable.

Processing model — synchronous sequential:
  Questions are processed one at a time in the order they appear in the CSV.
  This is intentional:
    - Keeps the implementation simple and debuggable.
    - Avoids the complexity of async task queues for a tool that processes
      tens to low-hundreds of questions per questionnaire.
    - Each question already makes two network calls (Pinecone + LLM), so
      parallelism would quickly hit rate limits on both services.
  If processing time becomes a concern, the natural next step is a background
  task with a status-polling endpoint — not in-process threading.

Error isolation:
  A failure on one question (LLM timeout, retrieval error) is caught and
  recorded as an error result rather than aborting the entire batch.
  The caller receives partial results with error details per question.
"""

import csv
import io
import logging

from app.services.query_service import (
    LLMAPIError,
    LLMTimeoutError,
    RetrievalError,
    run_query,
)
from app.config import settings

logger = logging.getLogger(__name__)


class QuestionnaireParseError(ValueError):
    """
    Raised when the uploaded CSV cannot be parsed into a valid question list.

    Mapped to HTTP 400 by the route handler.
    Covers: wrong file encoding, missing question column, empty file.
    """
    pass


def parse_questionnaire_csv(
        file_bytes: bytes,
        question_column: str = "question",
) -> list[dict]:
    """
    Parse a CSV file and extract rows with their question text.

    Args:
        file_bytes:      Raw bytes from the uploaded CSV file.
        question_column: Name of the column containing question text.
                         Case-insensitive match against the CSV header.

    Returns:
        List of dicts: [{"row": int, "question": str}, ...]
        Row numbers are 1-based and match the original CSV row (excluding header).
        Empty rows and rows where the question column is blank are skipped.

    Raises:
        QuestionnaireParseError: if the file cannot be decoded, has no
            recognisable header, the question column is not found, or
            produces zero non-empty rows after parsing.
    """
    # Decode bytes — UTF-8 with BOM fallback for files saved from Excel
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise QuestionnaireParseError(
                "Could not decode the CSV file. "
                "Save it as UTF-8 and re-upload."
            ) from exc

    reader = csv.DictReader(io.StringIO(text))

    # Validate that the CSV has a header row at all
    if reader.fieldnames is None:
        raise QuestionnaireParseError(
            "The CSV file appears to be empty or has no header row."
        )

    # Case-insensitive column lookup
    fieldnames_lower = {f.lower(): f for f in reader.fieldnames}
    matched_column = fieldnames_lower.get(question_column.lower())

    if matched_column is None:
        available = ", ".join(f'"{f}"' for f in reader.fieldnames)
        raise QuestionnaireParseError(
            f'Column "{question_column}" not found in the CSV. '
            f"Available columns: {available}. "
            f"Use the question_column query parameter to specify the correct column name."
        )

    rows = []
    for row_number, row in enumerate(reader, start=1):
        question_text = row.get(matched_column, "").strip()
        if not question_text:
            logger.debug("Row %d: empty question — skipping.", row_number)
            continue
        rows.append({"row": row_number, "question": question_text})

    if not rows:
        raise QuestionnaireParseError(
            f'No non-empty questions found in column "{question_column}". '
            "Check that the file contains data rows and the correct column is specified."
        )

    logger.info(
        "Parsed questionnaire CSV: %d question(s) from column '%s'.",
        len(rows),
        question_column,
    )
    return rows


def process_questionnaire(rows: list[dict]) -> list[dict]:
    """
    Run each question through the RAG pipeline and return structured results.

    Args:
        rows: Output of parse_questionnaire_csv — [{"row": int, "question": str}]

    Returns:
        List of result dicts, one per question:
        {
            "row":                 int,
            "question":            str,
            "answer":              str,
            "status":              str,   # answered | insufficient_evidence | no_documents | error
            "sources":             list[dict],  # [{document, page}]
            "top_retrieval_score": float | None,
        }

    Per-question errors (LLM timeout, retrieval failure) are caught and
    recorded with status="error" so the batch continues. The error message
    is included in the answer field for transparency.
    """
    results = []

    for item in rows:
        row_num = item["row"]
        question = item["question"]

        logger.info("Processing row %d: '%s'", row_num, question[:80])

        try:
            result = run_query(question=question, top_k=settings.top_k)
            results.append({
                "row": row_num,
                "question": question,
                "answer": result["answer"],
                "status": result["status"],
                "sources": result["sources"],
                "top_retrieval_score": result["top_retrieval_score"],
            })

        except (RetrievalError, LLMTimeoutError, LLMAPIError) as exc:
            # Isolate per-question failures — do not abort the entire batch
            logger.error("Row %d failed: %s", row_num, exc)
            results.append({
                "row": row_num,
                "question": question,
                "answer": f"Processing failed: {exc}",
                "status": "error",
                "sources": [],
                "top_retrieval_score": None,
            })

    logger.info(
        "Questionnaire processing complete: %d question(s) processed.",
        len(results),
    )
    return results