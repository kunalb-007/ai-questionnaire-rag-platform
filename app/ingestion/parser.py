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
# from docling.document_converter import DocumentConverter

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
        ".pdf":  _parse_pdf,
#         ".docx": _parse_docx,
        ".txt":  _parse_txt,
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

# Disabled for now
# def _parse_docx(path: Path) -> list[PageData]:
#     """
#     Docling — preserves real page numbers in DOCX.
#
#     Why Docling over python-docx?
#       python-docx has no concept of pages — it extracts paragraphs only.
#       Page breaks depend on a rendering engine (Word, LibreOffice).
#       Docling uses its own layout engine to detect page boundaries,
#       so chunk citations say "Page 3" instead of always "Page 1".
#
#     Docling returns a structured document. We iterate its pages,
#     extract text per page, and build the same PageData structure
#     as the PDF parser — keeping the rest of the pipeline unchanged.
#     """
#     converter = DocumentConverter()
#     result = converter.convert(str(path))
#     doc = result.document
#
#     pages: list[PageData] = []
#
#     # Docling exposes pages via doc.pages (dict keyed by page number)
#     for page_no, page_obj in doc.pages.items():
#         # Collect all text items belonging to this page
#         page_texts = []
#         for item, _ in doc.iterate_items():
#             # Each item carries a prov (provenance) list with page_no
#             for prov in getattr(item, "prov", []):
#                 if getattr(prov, "page_no", None) == page_no:
#                     text = getattr(item, "text", "")
#                     if text.strip():
#                         page_texts.append(text)
#
#         full_text = "\n".join(page_texts)
#         if full_text.strip():
#             pages.append(PageData(
#                 text=full_text,
#                 page=page_no,
#                 filename=path.name,
#             ))
#
#     # Fallback: if Docling page iteration yields nothing, use export_to_text()
#     if not pages:
#         logger.warning(
#             "Docling page-level extraction empty for '%s' — falling back to full text.",
#             path.name,
#         )
#         full_text = doc.export_to_text()
#         if full_text.strip():
#             pages.append(PageData(text=full_text, page=1, filename=path.name))
#
#     return pages


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