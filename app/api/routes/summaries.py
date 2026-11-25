"""Summary history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_user
from app.database.mongo import LECTURES, db
from app.services.lecture_service import get_lecture, lecture_public
from app.services.summary_service import delete_summary, get_keywords, get_summary, list_summaries
from app.utils.serialization import serialize

router = APIRouter(tags=["summaries"])


@router.get("/summaries")
async def summaries(
    lecture_id: str | None = Query(None),
    method: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    user: dict = Depends(get_current_user),
) -> dict:
    items, total = await list_summaries(
        user["_id"], lecture_id=lecture_id, method=method, limit=limit, skip=skip
    )
    return {"items": items, "total": total, "limit": limit, "skip": skip}


@router.get("/summaries/{summary_id}")
async def summary_detail(summary_id: str, user: dict = Depends(get_current_user)) -> dict:
    doc = await get_summary(user["_id"], summary_id)
    lecture = await db[LECTURES].find_one({"_id": doc["lecture_id"], "user_id": user["_id"]})
    if lecture is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found.")
    keywords = await get_keywords(user["_id"], lecture["_id"])
    summary = serialize(doc)
    summary["lecture_filename"] = lecture.get("original_filename")
    return {
        "summary": summary,
        "lecture": lecture_public(lecture),
        "keywords": keywords,
        "lecture_id": str(lecture["_id"]),
    }


@router.delete("/summaries/{summary_id}")
async def remove_summary(summary_id: str, user: dict = Depends(get_current_user)) -> dict:
    await delete_summary(user["_id"], summary_id)
    return {"message": "Summary deleted.", "id": summary_id}
