"""
parser.py — Extract raw text from PDF, DOCX, and TXT files.

PDF  → PyMuPDF (fitz)  — fast, accurate, preserves page numbers
DOCX → Docling         — preserves page numbers (python-docx cannot)
TXT  → plain read      — single page

Each parser returns:
    [{"text": str, "page": int, "filename": str}, ...]
"""

import logging
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PageData(TypedDict):
    text: str
    page: int
    filename: str


def parse_document(file_path: str) -> list[PageData]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    parsers = {
        ".pdf": _parse_pdf,
        # ".docx": _parse_docx — disabled; see note at bottom of this file.
        ".txt": _parse_txt,
    }

    if suffix not in parsers:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(parsers.keys())}"
        )

    logger.info("Parsing %s (%s)", path.name, suffix)
    pages = parsers[suffix](path)

    if not pages:
        raise ValueError(f"Document '{path.name}' produced no extractable text.")

    logger.info("Extracted %d page(s) from '%s'", len(pages), path.name)
    return pages


def _parse_pdf(path: Path) -> list[PageData]:
    """PyMuPDF — fast, accurate, real page numbers."""
    pages: list[PageData] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(PageData(text=text, page=page_num, filename=path.name))
            else:
                logger.debug("Page %d of '%s' is empty — skipping.", page_num, path.name)
    return pages

# ---------------------------------------------------------------------------
# DOCX support — currently disabled
# ---------------------------------------------------------------------------
# DOCX parsing via Docling is not active in this deployment.
#
# Why Docling (not python-docx)?
#   python-docx has no concept of pages — it extracts paragraphs only.
#   Page breaks depend on a rendering engine (Word, LibreOffice). Docling
#   uses its own layout engine to detect page boundaries, so chunk citations
#   say "Page 3" instead of always "Page 1". This matters for the enterprise
#   use case where auditors verify citations against the original document.
#
# Why is it disabled?
#   Docling pulls in heavy ML dependencies (torch, transformers) that make
#   the Docker image significantly larger. The free-tier deployment cannot
#   accommodate the download size or the memory footprint.
#
# To re-enable:
#   1. Uncomment the Docling import at the top of this file.
#   2. Add docling to requirements.txt.
#   3. Add ".docx" back to ALLOWED_EXTENSIONS in document_service.py.
#   4. Restore the _parse_docx implementation from git history or the
#      project README.


def _parse_txt(path: Path) -> list[PageData]:
    """Plain text — single page, UTF-8 with latin-1 fallback."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed for '%s', falling back to latin-1.", path.name)
        text = path.read_text(encoding="latin-1")

    if not text.strip():
        return []

    return [PageData(text=text, page=1, filename=path.name)]