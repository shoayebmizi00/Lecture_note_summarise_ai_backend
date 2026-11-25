"""File exports: PDF/TXT summaries, keyword and quiz downloads, original PDF."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.nlp.topic_modeling import generate_topics
from app.services.export_service import (
    build_keywords_csv,
    build_quiz_csv,
    build_quiz_txt,
    build_summary_pdf,
    build_summary_txt,
)
from app.services.lecture_service import get_lecture
from app.services.storage import resolve_upload
from app.services.summary_service import get_keywords, get_quiz, get_summary

router = APIRouter(tags=["exports"])
logger = get_logger(__name__)


def _filename(base: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", (base or "lecture").rsplit(".", 1)[0])[:60] or "lecture"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{stem}_{stamp}.{suffix}"


def _attachment(content: bytes | str, media_type: str, name: str) -> Response:
    body = content.encode("utf-8") if isinstance(content, str) else content
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )


def _topics_for(lecture: dict):
    text = lecture.get("extracted_text") or ""
    if len(text) < 200:
        return None
    from app.nlp.preprocessing import preprocess

    doc = preprocess(text)
    if not doc.sentences:
        return None
    headings = [h.get("text", "") for h in lecture.get("headings", []) if h.get("text")]
    return generate_topics(doc.clean_text, doc.sentences, headings, max_topics=8).as_dict()


@router.get("/summary/{summary_id}/download/pdf")
async def download_summary_pdf(
    summary_id: str,
    include_topics: bool = Query(True),
    user: dict = Depends(get_current_user),
) -> Response:
    summary = await get_summary(user["_id"], summary_id)
    lecture = await get_lecture(user["_id"], str(summary["lecture_id"]))
    keywords = await get_keywords(user["_id"], lecture["_id"])

    topics = None
    if include_topics:
        try:
            topics = await asyncio.to_thread(_topics_for, lecture)
        except Exception as exc:  # export must not fail because of topics
            logger.warning("Topic generation for PDF export skipped: %s", exc)

    try:
        pdf = await asyncio.to_thread(build_summary_pdf, summary, lecture, keywords, topics)
    except Exception as exc:
        logger.exception("PDF export failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not build the PDF: {exc}",
        ) from exc

    return _attachment(
        pdf, "application/pdf",
        _filename(lecture.get("original_filename", "lecture"), "summary.pdf"),
    )


@router.get("/summary/{summary_id}/download/txt")
async def download_summary_txt(
    summary_id: str,
    include_topics: bool = Query(True),
    user: dict = Depends(get_current_user),
) -> Response:
    summary = await get_summary(user["_id"], summary_id)
    lecture = await get_lecture(user["_id"], str(summary["lecture_id"]))
    keywords = await get_keywords(user["_id"], lecture["_id"])

    topics = None
    if include_topics:
        try:
            topics = await asyncio.to_thread(_topics_for, lecture)
        except Exception as exc:
            logger.warning("Topic generation for TXT export skipped: %s", exc)

    text = await asyncio.to_thread(build_summary_txt, summary, lecture, keywords, topics)
    return _attachment(
        text, "text/plain; charset=utf-8",
        _filename(lecture.get("original_filename", "lecture"), "summary.txt"),
    )


@router.get("/lectures/{lecture_id}/download/keywords")
async def download_keywords(
    lecture_id: str,
    format: str = Query("csv", pattern="^(csv|txt|json)$"),
    user: dict = Depends(get_current_user),
) -> Response:
    lecture = await get_lecture(user["_id"], lecture_id)
    keywords = await get_keywords(user["_id"], lecture["_id"])
    if not keywords:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No keywords stored for this lecture yet. Extract keywords first.",
        )
    base = _filename(lecture.get("original_filename", "lecture"), "")
    if format == "csv":
        return _attachment(build_keywords_csv(keywords), "text/csv; charset=utf-8",
                           base + "keywords.csv")
    if format == "txt":
        lines = [f"Important keywords - {lecture.get('original_filename')}", ""]
        lines += [f"{i}. {k['keyword']} (score {k['score']:.3f}, x{k['frequency']})"
                  for i, k in enumerate(keywords, start=1)]
        return _attachment("\n".join(lines), "text/plain; charset=utf-8", base + "keywords.txt")
    return _attachment(
        __import__("json").dumps(keywords, indent=2, default=str),
        "application/json; charset=utf-8", base + "keywords.json",
    )


@router.get("/lectures/{lecture_id}/download/quiz")
async def download_quiz(
    lecture_id: str,
    format: str = Query("txt", pattern="^(csv|txt)$"),
    user: dict = Depends(get_current_user),
) -> Response:
    lecture = await get_lecture(user["_id"], lecture_id)
    questions = await get_quiz(user["_id"], lecture["_id"])
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No quiz stored for this lecture yet. Generate a quiz first.",
        )
    base = _filename(lecture.get("original_filename", "lecture"), "")
    if format == "csv":
        return _attachment(build_quiz_csv(questions), "text/csv; charset=utf-8",
                           base + "quiz.csv")
    return _attachment(
        build_quiz_txt(questions, lecture.get("original_filename", "Lecture")),
        "text/plain; charset=utf-8", base + "quiz.txt",
    )


@router.get("/lectures/{lecture_id}/download/original")
async def download_original(lecture_id: str, user: dict = Depends(get_current_user)) -> FileResponse:
    lecture = await get_lecture(user["_id"], lecture_id)
    path = resolve_upload(lecture.get("stored_filename", ""))
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The stored PDF is missing.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=lecture.get("original_filename", "lecture.pdf"),
        headers={"Cache-Control": "no-store"},
    )
