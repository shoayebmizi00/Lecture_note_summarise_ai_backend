"""Summarisation, keywords, topics, quiz and evaluation endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.database.mongo import SUMMARIES, db
from app.nlp.evaluation import EvaluationError, compare_evaluations, evaluate
from app.schemas.lecture import (
    CompareEvaluationRequest,
    CompareRequest,
    EvaluateRequest,
    JobOut,
    KeywordsRequest,
    QuizRequest,
    SummarizeRequest,
    TopicsRequest,
)
from app.services.analysis_service import run_analysis, run_keywords, run_quiz, run_topics
from app.services.jobs import job_manager
from app.services.lecture_service import get_lecture
from app.services.summary_service import (
    attach_evaluation,
    get_keywords,
    get_quiz,
    get_summary,
    persist_summaries,
    save_keywords,
    save_quiz,
)
from app.utils.serialization import serialize

router = APIRouter(tags=["analysis"])
logger = get_logger(__name__)


def _lecture_text(lecture: dict[str, Any]) -> str:
    text = (lecture.get("extracted_text") or "").strip()
    if len(text) < 80:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This lecture does not contain enough extractable text to analyse.",
        )
    return text


@router.post("/summarize", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def summarize(req: SummarizeRequest, user: dict = Depends(get_current_user)) -> JobOut:
    lecture = await get_lecture(user["_id"], req.lecture_id)
    text = _lecture_text(lecture)
    user_id: ObjectId = user["_id"]

    def work(progress):
        return run_analysis(text, [req.method], req.length, progress=progress)

    async def on_success(result: dict[str, Any]) -> dict[str, Any]:
        summaries = await persist_summaries(user_id, lecture, result["variants"], save=req.save)
        if req.save:
            await save_keywords(user_id, lecture["_id"], result["keywords"])
        result.pop("variants", None)
        result["summaries"] = summaries
        return result

    job = await job_manager.submit("summarize", str(user_id), work, req.lecture_id, on_success)
    return JobOut(**serialize(job.as_dict()))


@router.post("/summarize/compare", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def compare_summaries(req: CompareRequest, user: dict = Depends(get_current_user)) -> JobOut:
    lecture = await get_lecture(user["_id"], req.lecture_id)
    text = _lecture_text(lecture)
    user_id: ObjectId = user["_id"]

    def work(progress):
        return run_analysis(text, ["extractive", "abstractive"], req.length, progress=progress)

    async def on_success(result: dict[str, Any]) -> dict[str, Any]:
        summaries = await persist_summaries(user_id, lecture, result["variants"], save=True)
        await save_keywords(user_id, lecture["_id"], result["keywords"])
        result.pop("variants", None)
        result["summaries"] = summaries
        return result

    job = await job_manager.submit("compare", str(user_id), work, req.lecture_id, on_success)
    return JobOut(**serialize(job.as_dict()))


@router.get("/jobs/{job_id}", response_model=JobOut)
async def job_status(job_id: str, user: dict = Depends(get_current_user)) -> JobOut:
    data = await job_manager.get(job_id, str(user["_id"]))
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JobOut(**serialize(data))


@router.get("/jobs")
async def job_history(limit: int = 20, user: dict = Depends(get_current_user)) -> dict:
    items = await job_manager.recent(str(user["_id"]), limit)
    return {"items": items}


@router.post("/keywords")
async def keywords(req: KeywordsRequest, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], req.lecture_id)
    text = _lecture_text(lecture)

    try:
        payload = await asyncio.to_thread(run_keywords, text, req.top_n, req.use_keybert)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if req.save and payload["keywords"]:
        await save_keywords(user["_id"], lecture["_id"], payload["keywords"])
    payload["lecture_id"] = str(lecture["_id"])
    payload["saved"] = bool(req.save)
    return payload


@router.get("/lectures/{lecture_id}/keywords")
async def lecture_keywords(lecture_id: str, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], lecture_id)
    items = await get_keywords(user["_id"], lecture["_id"])
    return {"lecture_id": lecture_id, "keywords": items, "count": len(items)}


@router.post("/topics")
async def topics(req: TopicsRequest, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], req.lecture_id)
    text = _lecture_text(lecture)
    headings = [h.get("text", "") for h in lecture.get("headings", []) if h.get("text")]

    try:
        payload = await asyncio.to_thread(run_topics, text, headings, req.max_topics)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    payload["lecture_id"] = str(lecture["_id"])
    return payload


@router.post("/quiz")
async def quiz(req: QuizRequest, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], req.lecture_id)
    text = _lecture_text(lecture)

    try:
        payload = await asyncio.to_thread(run_quiz, text, req.count, req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if req.save and payload["questions"]:
        stored = await save_quiz(user["_id"], lecture["_id"], payload, text)
        payload["stored"] = len(stored)
    else:
        payload["stored"] = 0
    payload["lecture_id"] = str(lecture["_id"])
    return payload


@router.get("/lectures/{lecture_id}/quiz")
async def lecture_quiz(lecture_id: str, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], lecture_id)
    items = await get_quiz(user["_id"], lecture["_id"])
    return {"lecture_id": lecture_id, "questions": items, "count": len(items)}


@router.post("/evaluate")
async def evaluate_summary(req: EvaluateRequest, user: dict = Depends(get_current_user)) -> dict:
    summary_doc = None
    summary_text = (req.summary_text or "").strip()
    lecture_id = req.lecture_id

    if req.summary_id:
        summary_doc = await get_summary(user["_id"], req.summary_id)
        summary_text = summary_doc.get("summary_text", "")
        lecture_id = lecture_id or str(summary_doc["lecture_id"])
    if not summary_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No summary text to evaluate.")

    lecture = None
    if lecture_id:
        lecture = await get_lecture(user["_id"], lecture_id)

    try:
        evaluation = await asyncio.to_thread(
            evaluate,
            summary_text,
            req.reference_summary,
            lecture.get("extracted_text") if lecture else None,
            (summary_doc or {}).get("processing_time"),
            (summary_doc or {}).get("method"),
        )
    except EvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    payload = evaluation.as_dict()
    payload["summary_preview"] = summary_text[:400]
    if summary_doc is not None:
        await attach_evaluation(user["_id"], str(summary_doc["_id"]), payload)
        payload["summary_id"] = str(summary_doc["_id"])
        payload["lecture_id"] = str(summary_doc["lecture_id"])
    return payload


@router.post("/evaluate/compare")
async def compare_evaluation(req: CompareEvaluationRequest, user: dict = Depends(get_current_user)) -> dict:
    lecture = await get_lecture(user["_id"], req.lecture_id)

    async def latest(method: str) -> dict | None:
        return await db[SUMMARIES].find_one(
            {"lecture_id": lecture["_id"], "user_id": user["_id"], "method": method},
            sort=[("created_at", -1)],
        )

    entries: list[dict] = []
    missing: list[str] = []
    for method in ("extractive", "abstractive"):
        doc = await latest(method)
        if doc is None:
            missing.append(method)
            continue
        entries.append({
            "method": method,
            "summary_id": str(doc["_id"]),
            "summary_text": doc.get("summary_text", ""),
            "summary_length": doc.get("summary_length"),
            "processing_time": doc.get("processing_time"),
        })
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Generate both a TextRank and a BART summary for this lecture first. "
                f"Missing: {', '.join(missing)}."
            ),
        )

    try:
        comparison = await asyncio.to_thread(
            compare_evaluations, entries, req.reference_summary, lecture.get("extracted_text")
        )
    except EvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    comparison["lecture_id"] = req.lecture_id
    comparison["lecture_filename"] = lecture.get("original_filename")
    return comparison
