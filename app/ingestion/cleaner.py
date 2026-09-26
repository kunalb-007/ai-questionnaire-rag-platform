"""
cleaner.py — Normalize extracted text before chunking.

4 operations only (interview-friendly):
  1. Normalize line endings  → consistent \n
  2. Rejoin hyphenated PDF line breaks  → "authen-\ntication" → "authentication"
  3. Remove excessive whitespace / blank lines
  4. Preserve punctuation, case, and paragraph structure

What we deliberately do NOT do:
  - Lowercase  (destroys "MFA", "GDPR")
  - Remove punctuation  (LLM needs sentence boundaries)
  - Strip all newlines  (destroys paragraph structure)
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Apply 4-step normalization to a single page's extracted text."""
    if not text:
        return ""

    # 1. Normalize line endings (CRLF / CR → LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Rejoin hyphenated PDF line breaks
    #    "authen-\ntication" → "authentication"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # 3. Remove excessive blank lines (3+ newlines → 1 blank line)
    #    and collapse multiple spaces within a line
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Strip leading/trailing whitespace — punctuation, case, and
    #    paragraph structure (\n\n boundaries) are left intact
    return text.strip()


def clean_pages(pages: list[dict]) -> list[dict]:
    """Apply clean_text to every page. Drop pages that become empty."""
    cleaned = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        if cleaned_text:
            cleaned.append({**page, "text": cleaned_text})
        else:
            logger.debug(
                "Page %d of '%s' became empty after cleaning — skipping.",
                page["page"], page["filename"],
            )

    logger.info("Cleaning complete: %d/%d pages retained.", len(cleaned), len(pages))
    return cleaned