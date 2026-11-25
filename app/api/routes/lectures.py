"""Lecture upload, listing, detail and deletion."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.logging import get_logger
from app.nlp.pdf_extractor import PdfExtractionError, extract_pdf
from app.nlp.preprocessing import preprocess
from app.schemas.lecture import LectureDetail, LectureOut, UploadResponse
from app.services.lecture_service import (
    count_summaries,
    create_lecture,
    delete_lecture,
    get_lecture,
    lecture_detail,
    lecture_public,
    list_lectures,
    stats,
)
from app.services.storage import delete_stored_file, save_upload
from app.utils.files import human_size, looks_like_pdf_header, sanitize_filename

router = APIRouter(tags=["lectures"])
logger = get_logger(__name__)

CHUNK = 1024 * 1024


def _process_pdf(data: bytes, use_ocr: bool):
    extraction = extract_pdf(data, allow_ocr=use_ocr)
    preprocessed = preprocess(extraction.text)
    return extraction, preprocessed


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"The file is larger than the {settings.max_upload_mb} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_lecture(
    file: UploadFile = File(..., description="PDF lecture notes"),
    use_ocr: bool = Form(True),
    user: dict = Depends(get_current_user),
) -> UploadResponse:
    original = sanitize_filename(file.filename or "lecture.pdf")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    limit = settings.max_upload_mb * 1024 * 1024
    data = await _read_limited(file, limit)
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if not looks_like_pdf_header(data[:1024]):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="This file is not a valid PDF (missing %PDF header).",
        )

    try:
        extraction, preprocessed = await asyncio.to_thread(_process_pdf, data, use_ocr)
    except PdfExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF processing failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process this PDF: {exc}",
        ) from exc

    stored_rel, _path = await asyncio.to_thread(save_upload, str(user["_id"]), original, data)
    try:
        doc = await create_lecture(
            user["_id"],
            original_filename=original,
            stored_filename=stored_rel,
            file_size=len(data),
            extraction=extraction,
            preprocessed=preprocessed,
            use_ocr=use_ocr,
        )
    except Exception:
        delete_stored_file(stored_rel)
        raise

    logger.info(
        "User %s uploaded %s (%s, %s pages)",
        user.get("username"), original, human_size(len(data)), extraction.page_count,
    )
    return UploadResponse(
        lecture=LectureOut(**lecture_public(doc)),
        warnings=extraction.warnings + preprocessed.warnings,
        text_preview=extraction.text[:700],
    )


@router.get("/lectures")
async def list_user_lectures(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=80),
    user: dict = Depends(get_current_user),
) -> dict:
    items, total = await list_lectures(user["_id"], limit=limit, skip=skip, query=q)
    return {"items": items, "total": total, "limit": limit, "skip": skip}


@router.get("/lectures/{lecture_id}", response_model=LectureDetail)
async def get_lecture_detail(
    lecture_id: str,
    user: dict = Depends(get_current_user),
) -> LectureDetail:
    doc = await get_lecture(user["_id"], lecture_id)
    count = await count_summaries(doc["_id"])
    return LectureDetail(**lecture_detail(doc, count))


@router.delete("/lectures/{lecture_id}")
async def remove_lecture(
    lecture_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    return await delete_lecture(user["_id"], lecture_id)


@router.get("/stats")
async def dashboard_stats(user: dict = Depends(get_current_user)) -> dict:
    data = await stats(user["_id"])
    data["username"] = user.get("username")
    return data
