"""
cleaner.py — Normalize extracted text before chunking.

Philosophy: clean enough to remove noise, conservative enough to preserve
meaning and structure. Aggressive cleaning (stripping all punctuation,
lowercasing, stemming) belongs in classical NLP, not RAG pipelines where
the LLM handles language understanding.

Interview note on cleaning limitations:
  - Hyphenated line-breaks in PDFs ("secu-\nrity") look like real words
    after join but are invisible to character-based chunkers.
  - Headers extracted from PDFs often appear as isolated lines — we preserve
    them because they give context to the chunks that follow.
  - We cannot recover structure destroyed during PDF text extraction
    (e.g., multi-column layouts).
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Apply lightweight normalization to a single page's extracted text.

    Steps applied (in order):
      1. Normalize line endings to \\n.
      2. Rejoin PDF hyphenated line-breaks (e.g., "secu-\\nrity" → "security").
      3. Collapse runs of blank lines to a single blank line (paragraph boundary).
      4. Strip leading/trailing whitespace from each line.
      5. Remove lines that are purely whitespace or non-printable characters.
      6. Collapse multiple spaces within a line to a single space.

    What we deliberately do NOT do:
      - Lowercase (destroys acronym meaning: "MFA", "GDPR").
      - Remove punctuation (destroys sentence boundaries the LLM needs).
      - Strip all newlines (destroys paragraph structure).
    """
    if not text:
        return ""

    # 1. Normalize CRLF / CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Rejoin hyphenated line-breaks common in justified PDF text
    #    "authen-\ntication" → "authentication"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # 3. Collapse 3+ consecutive newlines to exactly 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4 & 5. Strip each line; drop lines that are empty after stripping
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Drop lines with only non-printable or control characters
        if stripped and any(c.isprintable() for c in stripped):
            lines.append(stripped)
        else:
            # Preserve blank line as paragraph separator
            lines.append("")

    text = "\n".join(lines)

    # 6. Collapse multiple spaces within a line
    text = re.sub(r"[ \t]+", " ", text)

    # Final strip of the whole text
    return text.strip()


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Apply clean_text to every page dict returned by the parser.

    Filters out pages that become empty after cleaning.
    Returns a new list — does not mutate input.
    """
    cleaned = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        if cleaned_text:
            cleaned.append({**page, "text": cleaned_text})
        else:
            logger.debug(
                "Page %d of '%s' became empty after cleaning — skipping.",
                page["page"],
                page["filename"],
            )

    logger.info(
        "Cleaning complete: %d/%d pages retained.", len(cleaned), len(pages)
    )
    return cleaned