"""
parser.py — Extract raw text from PDF, DOCX, and TXT files.

Each parser returns a list of page dicts:
    [{"text": str, "page": int, "filename": str}, ...]

Why a list of pages?
  Preserving page numbers is essential for citations. Users need to know
  *where* in the source document the answer came from, not just which file.

Interview note: PyMuPDF (fitz) is faster and more accurate than pdfplumber
for most PDFs. python-docx handles .docx natively. We treat TXT as a single
page because there is no page structure to preserve.
"""

import logging
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF
from docx import Document

logger = logging.getLogger(__name__)


class PageData(TypedDict):
    text: str
    page: int
    filename: str


def parse_document(file_path: str) -> list[PageData]:
    """
    Route a file to the correct parser based on its extension.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
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
    """
    Extract text page-by-page from a PDF using PyMuPDF.

    Limitations:
      - Scanned PDFs without OCR will return empty or garbled text.
      - Complex multi-column layouts may have text extraction order issues.
      - Tables are extracted as raw text, losing their structure.
    """
    pages: list[PageData] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # type: ignore[attr-defined]

            if text.strip():
                pages.append(
                    PageData(text=text, page=page_num, filename=path.name)
                )
            else:
                logger.debug(
                    "Page %d of '%s' is empty or image-only — skipping.",
                    page_num,
                    path.name,
                )

    return pages


def _parse_docx(path: Path) -> list[PageData]:
    """
    Extract text from a DOCX file as a single logical page.

    DOCX files do not have a reliable programmatic page count in python-docx
    (page breaks depend on rendering engine). We extract all paragraphs
    and treat the document as page 1.

    Limitation: page number metadata will always be 1 for DOCX files.
    """
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        return []

    full_text = "\n".join(paragraphs)
    return [PageData(text=full_text, page=1, filename=path.name)]


def _parse_txt(path: Path) -> list[PageData]:
    """
    Read a plain text file as a single page.

    Encoding: UTF-8 with fallback to latin-1 for legacy files.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for '%s', falling back to latin-1.", path.name)
        text = path.read_text(encoding="latin-1")

    if not text.strip():
        return []

    return [PageData(text=text, page=1, filename=path.name)]