"""PDF text extraction with optional Tesseract OCR fallback."""

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.files import human_size

logger = get_logger(__name__)


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened or contains no usable text."""


@dataclass
class PageText:
    number: int
    text: str
    char_count: int
    used_ocr: bool = False


@dataclass
class DetectedHeading:
    text: str
    page: int
    font_size: float


@dataclass
class ExtractionResult:
    text: str
    pages: list[PageText]
    page_count: int
    char_count: int
    word_count: int
    headings: list[DetectedHeading] = field(default_factory=list)
    ocr_used: bool = False
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_scanned_like(self) -> bool:
        return self.char_count < self.page_count * settings.ocr_min_chars_per_page


_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)


def ocr_available() -> tuple[bool, str]:
    """Check whether pytesseract + the Tesseract binary are usable."""
    try:
        import pytesseract
    except Exception:
        return False, "pytesseract is not installed."
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version}"
    except Exception as exc:  # binary missing / not on PATH
        return False, f"Tesseract binary not found ({exc.__class__.__name__})."


def _clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WHITESPACE.sub(" ", line).strip()
        if not line:
            lines.append("")
            continue
        if _PAGE_NUMBER_LINE.match(line):
            continue
        # ZapfDingbats bullet glyphs are often extracted as a lone "G"/"g".
        line = re.sub(r"^[Gg]\s+(?=[A-Z][\w\- ]{1,40}:)", "", line)
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dehyphenate(text: str) -> str:
    # "informa-\ntion" -> "information" (keeps real hyphenated compounds intact).
    return re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)


def _remove_repeated_margins(pages: list[PageText]) -> None:
    """Drop lines that repeat on most pages (running headers / footers)."""
    if len(pages) < 3:
        return
    counts: dict[str, int] = {}
    for page in pages:
        seen = {
            line.strip().lower()
            for line in page.text.split("\n")
            if 3 <= len(line.strip()) <= 90
        }
        for line in seen:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in counts.items() if count >= threshold}
    if not repeated:
        return
    for page in pages:
        kept = [ln for ln in page.text.split("\n") if ln.strip().lower() not in repeated]
        page.text = "\n".join(kept).strip()
        page.char_count = len(page.text)


def _detect_headings(doc: "fitz.Document", page_texts: list[PageText]) -> list[DetectedHeading]:
    """Heading detection from font size relative to the document body text."""
    sizes: list[tuple[float, int]] = []
    raw: list[tuple[int, str, float]] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        try:
            data = page.get_text("dict")
        except Exception:
            continue
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                size = max(span.get("size", 0.0) for span in spans)
                sizes.append((round(size, 1), len(text)))
                raw.append((pno + 1, text, size))
    if not sizes:
        return []
    # Body size = size with the most characters.
    by_size: dict[float, int] = {}
    for size, length in sizes:
        by_size[size] = by_size.get(size, 0) + length
    body_size = max(by_size.items(), key=lambda item: item[1])[0]
    headings: list[DetectedHeading] = []
    seen: set[str] = set()
    for page_no, text, size in raw:
        if size < body_size * 1.12 or len(text) < 3 or len(text) > 120:
            continue
        if len(text.split()) > 16:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        headings.append(DetectedHeading(text=text, page=page_no, font_size=round(size, 1)))
    return headings[:120]


_SENTENCE_TAIL = re.compile(r'[.!?:;)\]"\']$')
_HEADING_LIKE = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[IVXLC]{1,6}\.)\s+[A-Z]")


def _probable_heading(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 90:
        return False
    words = text.split()
    if len(words) > 12:
        return False
    if _HEADING_LIKE.match(text):
        return True
    if text.endswith((".", "?", "!;,")):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return True
    return len(words) <= 8 and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1)


def _page_text(page: "fitz.Page") -> str:
    """Extract one page as paragraphs (PyMuPDF blocks), keeping headings separate."""
    try:
        blocks = page.get_text("blocks")
        parts: list[str] = []
        for block in blocks:
            # block = (x0, y0, x1, y1, text, block_no, block_type); type 1 = image
            if len(block) >= 7 and block[6] == 1:
                continue
            raw = block[4] if len(block) > 4 else ""
            cleaned = _clean_page_text(raw)
            if not cleaned:
                continue
            previous = parts[-1] if parts else ""
            continues = bool(previous) and (
                previous.endswith((",", ";", ":", "-", "\u2013", "\u2014", "(", "/"))
                or (len(previous) >= 60 and not _SENTENCE_TAIL.search(previous))
            )
            if (
                continues
                and not _probable_heading(previous)
                and not _probable_heading(cleaned)
                and (cleaned[:1].islower() or len(previous) >= 60)
            ):
                # A prose paragraph was split across blocks: stitch the sentence back.
                parts[-1] = f"{previous} {cleaned}"
            else:
                parts.append(cleaned)
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass
    try:
        return _clean_page_text(page.get_text("text") or "")
    except Exception:
        return ""


def _ocr_page(page: "fitz.Page") -> str:
    import pytesseract
    from PIL import Image

    zoom = settings.ocr_dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang="eng", config="--psm 3")


def extract_pdf(source: str | Path | bytes, *, allow_ocr: bool | None = None) -> ExtractionResult:
    """Extract text page by page, falling back to OCR for image-only pages."""
    allow_ocr = settings.ocr_enabled if allow_ocr is None else allow_ocr
    warnings: list[str] = []

    try:
        if isinstance(source, (bytes, bytearray)):
            doc = fitz.open(stream=bytes(source), filetype="pdf")
            size_bytes = len(source)
        else:
            path = Path(source)
            if not path.exists():
                raise PdfExtractionError("The PDF file could not be found on disk.")
            size_bytes = path.stat().st_size
            doc = fitz.open(str(path))
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(
            "This file could not be opened as a PDF. It may be corrupted or not a real PDF."
        ) from exc

    try:
        if doc.needs_pass:
            raise PdfExtractionError("This PDF is password protected. Please upload an unlocked copy.")

        if doc.page_count == 0:
            raise PdfExtractionError("The PDF has no pages.")
        if doc.page_count > settings.max_pdf_pages:
            raise PdfExtractionError(
                f"The PDF has {doc.page_count} pages; the limit is {settings.max_pdf_pages}."
            )

        ocr_ok, ocr_reason = ocr_available()
        page_texts: list[PageText] = []
        empty_pages = 0

        for index in range(doc.page_count):
            page = doc[index]
            cleaned = _page_text(page)

            used_ocr = False
            if len(cleaned) < settings.ocr_min_chars_per_page and allow_ocr:
                if ocr_ok:
                    try:
                        ocr_text = _clean_page_text(_ocr_page(page))
                        if len(ocr_text) > len(cleaned):
                            cleaned = ocr_text
                            used_ocr = True
                    except Exception as exc:
                        warnings.append(f"OCR failed on page {index + 1}: {exc}")
                elif index == 0:
                    warnings.append(
                        "Some pages look scanned but OCR is unavailable: "
                        f"{ocr_reason} Install Tesseract and set TESSERACT_CMD in backend/.env."
                    )

            if not cleaned:
                empty_pages += 1
            page_texts.append(
                PageText(
                    number=index + 1,
                    text=cleaned,
                    char_count=len(cleaned),
                    used_ocr=used_ocr,
                )
            )

        if empty_pages == doc.page_count:
            raise PdfExtractionError(
                "No extractable text was found in this PDF. It is probably a scanned document "
                "and OCR is unavailable or disabled."
            )
        if empty_pages:
            warnings.append(f"{empty_pages} of {doc.page_count} pages contain no extractable text.")

        _remove_repeated_margins(page_texts)
        for page in page_texts:
            page.text = _dehyphenate(page.text)

        # Join pages with paragraph breaks, but stitch a sentence back together
        # when a page ends mid-sentence (common in handouts and slide decks).
        parts: list[str] = []
        for page in page_texts:
            text = page.text.strip()
            if not text:
                continue
            if parts and not re.search(r'[.!?:;)\]"\']$', parts[-1]):
                parts[-1] = f"{parts[-1]} {text}"
            else:
                parts.append(text)
        full_text = "\n\n".join(parts).strip()
        char_count = len(full_text)
        if char_count < 80:
            raise PdfExtractionError(
                "The extracted text is too short to summarise (under 80 characters)."
            )
        if char_count > settings.max_extracted_chars:
            full_text = full_text[: settings.max_extracted_chars]
            warnings.append(
                "The document is very long; text beyond "
                f"{settings.max_extracted_chars:,} characters was ignored."
            )
            char_count = len(full_text)

        headings = _detect_headings(doc, page_texts)
        ocr_pages = sum(1 for p in page_texts if p.used_ocr)
        if ocr_pages:
            warnings.append(f"OCR was used on {ocr_pages} page(s).")

        meta = dict(doc.metadata or {})
        meta.pop("encryption", None)
        meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float))}

        logger.info(
            "Extracted %s chars / %s pages from PDF (%s, ocr_pages=%s)",
            char_count,
            doc.page_count,
            human_size(size_bytes),
            ocr_pages,
        )

        return ExtractionResult(
            text=full_text,
            pages=page_texts,
            page_count=doc.page_count,
            char_count=char_count,
            word_count=len(full_text.split()),
            headings=headings,
            ocr_used=ocr_pages > 0,
            ocr_pages=ocr_pages,
            warnings=warnings,
            metadata=meta,
        )
    finally:
        doc.close()
