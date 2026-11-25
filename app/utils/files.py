"""File-name sanitising, PDF magic-byte validation and safe path handling."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

PDF_MAGIC = b"%PDF-"
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(name: str, fallback: str = "lecture.pdf") -> str:
    """Return a harmless base name (no directories, no control characters)."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE.sub("_", name).strip(" ._-")
    if not name:
        return fallback
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Keep names readable but bounded.
    stem = name[:-4][:80]
    return f"{stem or 'lecture'}.pdf"


def looks_like_pdf_header(header: bytes) -> bool:
    return header.startswith(PDF_MAGIC) or PDF_MAGIC in header[:1024]


def is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
