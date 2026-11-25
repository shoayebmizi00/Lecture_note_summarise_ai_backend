"""Local file storage for uploads and generated exports (path-safe)."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.files import is_within, sanitize_filename

logger = get_logger(__name__)


def save_upload(user_id: str, original_name: str, data: bytes) -> tuple[str, Path]:
    """Persist the uploaded PDF and return (relative_path, absolute_path)."""
    safe_name = sanitize_filename(original_name)
    folder = settings.upload_dir / user_id
    folder.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}.pdf"
    path = folder / stored
    path.write_bytes(data)
    logger.info("Stored upload %s (%s) as %s", safe_name, len(data), path.name)
    return f"{user_id}/{stored}", path


def resolve_upload(stored_filename: str) -> Path | None:
    path = (settings.upload_dir / stored_filename).resolve()
    if not is_within(settings.upload_dir, path) or not path.exists():
        return None
    return path


def delete_stored_file(stored_filename: str) -> bool:
    path = resolve_upload(stored_filename)
    if path is None:
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)
        return False


def output_dir(user_id: str) -> Path:
    folder = settings.output_dir / user_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def safe_output_name(name: str, default: str = "summary") -> str:
    cleaned = sanitize_filename(name, fallback=f"{default}.pdf")
    return cleaned[:-4] if cleaned.lower().endswith(".pdf") else cleaned
