"""Lecture persistence and ownership-aware lookups."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import HTTPException, status

from app.database.mongo import JOBS, KEYWORDS, LECTURES, QUIZZES, SUMMARIES, db
from app.nlp.pdf_extractor import ExtractionResult
from app.nlp.preprocessing import PreprocessedDocument
from app.services.storage import delete_stored_file
from app.utils.serialization import serialize


def lecture_public(doc: dict[str, Any], summary_count: int = 0) -> dict[str, Any]:
    preview = (doc.get("extracted_text") or "")[:320]
    return {
        "id": str(doc["_id"]),
        "original_filename": doc.get("original_filename", "lecture.pdf"),
        "page_count": doc.get("page_count", 0),
        "char_count": doc.get("char_count", 0),
        "word_count": doc.get("word_count", 0),
        "sentence_count": doc.get("sentence_count", 0),
        "file_size": doc.get("file_size", 0),
        "upload_date": doc.get("upload_date"),
        "processing_status": doc.get("processing_status", "ready"),
        "ocr_used": doc.get("ocr_used", False),
        "warnings": doc.get("warnings", []),
        "preview": preview,
        "summary_count": summary_count,
    }


def lecture_detail(doc: dict[str, Any], summary_count: int = 0) -> dict[str, Any]:
    data = lecture_public(doc, summary_count)
    data.update(
        {
            "extracted_text": doc.get("extracted_text", ""),
            "headings": doc.get("headings", []),
            "pages": doc.get("pages", []),
            "metadata_info": doc.get("metadata_info", {}),
        }
    )
    return data


async def create_lecture(
    user_id: ObjectId,
    *,
    original_filename: str,
    stored_filename: str,
    file_size: int,
    extraction: ExtractionResult,
    preprocessed: PreprocessedDocument,
    use_ocr: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "extracted_text": extraction.text,
        "page_count": extraction.page_count,
        "char_count": extraction.char_count,
        "word_count": extraction.word_count,
        "sentence_count": preprocessed.sentence_count,
        "file_size": file_size,
        "upload_date": now,
        "processing_status": "ready",
        "ocr_used": extraction.ocr_used,
        "ocr_pages": extraction.ocr_pages,
        "ocr_requested": use_ocr,
        "warnings": extraction.warnings + preprocessed.warnings,
        "headings": [h.__dict__ for h in extraction.headings],
        "pages": [
            {"number": p.number, "char_count": p.char_count, "used_ocr": p.used_ocr}
            for p in extraction.pages
        ],
        "metadata_info": extraction.metadata,
        "duplicates_removed": preprocessed.duplicates_removed,
        "noise_removed": preprocessed.noise_removed,
    }
    result = await db[LECTURES].insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def get_lecture(user_id: ObjectId, lecture_id: str) -> dict[str, Any]:
    if not ObjectId.is_valid(lecture_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found.")
    doc = await db[LECTURES].find_one({"_id": ObjectId(lecture_id), "user_id": user_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found.")
    return doc


async def count_summaries(lecture_id: ObjectId) -> int:
    return await db[SUMMARIES].count_documents({"lecture_id": lecture_id})


async def list_lectures(user_id: ObjectId, *, limit: int = 20, skip: int = 0,
                        query: str | None = None) -> tuple[list[dict[str, Any]], int]:
    match: dict[str, Any] = {"user_id": user_id}
    if query:
        match["original_filename"] = {"$regex": query.strip()[:80].replace("\\", ""), "$options": "i"}

    total = await db[LECTURES].count_documents(match)
    cursor = (
        db[LECTURES]
        .find(match)
        .sort("upload_date", -1)
        .skip(max(0, skip))
        .limit(max(1, min(limit, 100)))
    )
    docs = [doc async for doc in cursor]
    if not docs:
        return [], total

    ids = [doc["_id"] for doc in docs]
    counts: dict[ObjectId, int] = {}
    pipeline = [
        {"$match": {"lecture_id": {"$in": ids}}},
        {"$group": {"_id": "$lecture_id", "count": {"$sum": 1}}},
    ]
    cursor = await db[SUMMARIES].aggregate(pipeline)
    async for row in cursor:
        counts[row["_id"]] = row["count"]

    return [lecture_public(doc, counts.get(doc["_id"], 0)) for doc in docs], total


async def delete_lecture(user_id: ObjectId, lecture_id: str) -> dict[str, Any]:
    doc = await get_lecture(user_id, lecture_id)
    oid = doc["_id"]

    summary_ids = [s["_id"] async for s in db[SUMMARIES].find(
        {"lecture_id": oid, "user_id": user_id}, {"_id": 1}
    )]

    delete_stored_file(doc.get("stored_filename", ""))
    await db[SUMMARIES].delete_many({"lecture_id": oid, "user_id": user_id})
    await db[KEYWORDS].delete_many({"lecture_id": oid, "user_id": user_id})
    await db[QUIZZES].delete_many({"lecture_id": oid, "user_id": user_id})
    await db[JOBS].delete_many({"lecture_id": lecture_id, "user_id": user_id})
    await db[LECTURES].delete_one({"_id": oid, "user_id": user_id})

    return {
        "message": "Lecture deleted.",
        "deleted_summaries": len(summary_ids),
        "deleted_lecture_id": lecture_id,
    }


async def stats(user_id: ObjectId) -> dict[str, Any]:
    lecture_count = await db[LECTURES].count_documents({"user_id": user_id})
    summary_count = await db[SUMMARIES].count_documents({"user_id": user_id})
    keyword_count = await db[KEYWORDS].count_documents({"user_id": user_id})
    quiz_count = await db[QUIZZES].count_documents({"user_id": user_id})

    pipeline = [
        {"$match": {"user_id": user_id, "processing_time": {"$gt": 0}}},
        {"$group": {"_id": None, "avg": {"$avg": "$processing_time"}, "total": {"$sum": "$processing_time"}}},
    ]
    timing = {"avg": 0.0, "total": 0.0}
    cursor = await db[SUMMARIES].aggregate(pipeline)
    async for row in cursor:
        timing = {"avg": row.get("avg", 0.0), "total": row.get("total", 0.0)}

    by_method: dict[str, int] = {"extractive": 0, "abstractive": 0}
    method_cursor = await db[SUMMARIES].aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$method", "count": {"$sum": 1}}},
    ])
    async for row in method_cursor:
        by_method[str(row["_id"])] = row["count"]

    recent = [serialize(doc) async for doc in db[SUMMARIES].find({"user_id": user_id})
              .sort("created_at", -1).limit(5)]
    if recent:
        lecture_ids = list({r["lecture_id"] for r in recent})
        names = {
            l["_id"]: l.get("original_filename", "")
            async for l in db[LECTURES].find({"_id": {"$in": lecture_ids}}, {"original_filename": 1})
        }
        for item in recent:
            item["lecture_filename"] = names.get(item.get("lecture_id"), "")

    return {
        "lecture_count": lecture_count,
        "summary_count": summary_count,
        "keyword_count": keyword_count,
        "quiz_count": quiz_count,
        "avg_processing_time": round(timing["avg"], 2),
        "total_processing_time": round(timing["total"], 2),
        "summaries_by_method": by_method,
        "recent_summaries": recent,
    }
