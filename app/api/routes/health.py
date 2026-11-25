"""Health and readiness endpoints (no authentication required)."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings
from app.database.mongo import db
from app.nlp.pdf_extractor import ocr_available
from app.services.analysis_service import model_status

router = APIRouter(tags=["system"])
STARTED_AT = time.time()

VERSION = "1.0.0"


@router.get("/health")
async def health() -> dict:
    database_ok = await db.ping()
    ocr_ok, ocr_detail = ocr_available()
    models = model_status()
    return {
        "status": "ok" if database_ok else "degraded",
        "app": settings.app_name,
        "version": VERSION,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "python": platform.python_version(),
        "database": {
            "connected": database_ok,
            "engine": "MongoDB Atlas",
            "database": settings.mongodb_db,
        },
        "ocr": {"available": ocr_ok, "detail": ocr_detail, "enabled": settings.ocr_enabled},
        "models": models,
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_pdf_pages": settings.max_pdf_pages,
            "chunk_token_budget": settings.chunk_token_budget,
        },
    }


@router.get("/models")
async def models() -> dict:
    return model_status()
