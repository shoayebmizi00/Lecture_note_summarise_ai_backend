"""Persistence for summaries, keywords, quizzes and evaluations (MongoDB)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import HTTPException, status

from app.core.logging import get_logger
from app.database.mongo import KEYWORDS, LECTURES, QUIZZES, SUMMARIES, db
from app.utils.serialization import serialize

logger = get_logger(__name__)


def summary_public(doc: dict[str, Any], lecture_filename: str | None = None) -> dict[str, Any]:
    data = serialize(doc)
    data["lecture_filename"] = lecture_filename or doc.get("lecture_filename")
    return data


async def save_keywords(user_id: ObjectId, lecture_id: ObjectId,
                        keywords: list[dict[str, Any]]) -> None:
    await db[KEYWORDS].delete_many({"lecture_id": lecture_id, "user_id": user_id})
    if not keywords:
        return
    now = datetime.now(timezone.utc)
    docs = [
        {
            "user_id": user_id,
            "lecture_id": lecture_id,
            "keyword": item.get("keyword", ""),
            "score": float(item.get("score", 0.0)),
            "frequency": int(item.get("frequency", 0)),
            "kind": item.get("kind", "term"),
            "method": item.get("method", "tfidf"),
            "created_at": now,
        }
        for item in keywords
        if item.get("keyword")
    ]
    if docs:
        await db[KEYWORDS].insert_many(docs)


async def get_keywords(user_id: ObjectId, lecture_id: ObjectId) -> list[dict[str, Any]]:
    cursor = db[KEYWORDS].find({"lecture_id": lecture_id, "user_id": user_id}).sort("score", -1)
    return [serialize(doc) async for doc in cursor]


async def persist_summaries(user_id: ObjectId, lecture: dict[str, Any],
                            variants: list[dict[str, Any]], *,
                            save: bool = True) -> list[dict[str, Any]]:
    """Insert one summary document per generated variant and return API payloads."""
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for variant in variants:
        doc = {
            "user_id": user_id,
            "lecture_id": lecture["_id"],
            "lecture_filename": lecture.get("original_filename", "lecture.pdf"),
            "method": variant["method"],
            "algorithm": variant.get("algorithm"),
            "summary_text": variant["summary_text"],
            "summary_length": variant.get("length", "medium"),
            "processing_time": float(variant.get("processing_time", 0.0)),
            "created_at": now,
            "model_name": variant.get("model_name"),
            "requested_model": variant.get("requested_model"),
            "fallback_used": bool(variant.get("fallback_used", False)),
            "device": variant.get("device"),
            "chunk_count": variant.get("chunk_count", 0),
            "chunk_summaries": variant.get("chunk_summaries", []),
            "compression_ratio": variant.get("compression_ratio"),
            "summary_word_count": variant.get("summary_word_count"),
            "source_word_count": variant.get("source_word_count"),
            "source_sentence_count": variant.get("source_sentence_count"),
            "selected_sentences": variant.get("selected_sentences", []),
            "sentence_scores": variant.get("sentence_scores", []),
            "warnings": variant.get("warnings", []),
            "saved": save,
        }
        if save:
            result = await db[SUMMARIES].insert_one(doc)
            doc["_id"] = result.inserted_id
        else:
            doc["_id"] = ObjectId()
        out.append(summary_public(doc))
    return out


async def get_summary(user_id: ObjectId, summary_id: str) -> dict[str, Any]:
    if not ObjectId.is_valid(summary_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found.")
    doc = await db[SUMMARIES].find_one({"_id": ObjectId(summary_id), "user_id": user_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found.")
    return doc


async def list_summaries(user_id: ObjectId, *, lecture_id: str | None = None,
                         method: str | None = None,
                         limit: int = 50, skip: int = 0) -> tuple[list[dict[str, Any]], int]:
    query: dict[str, Any] = {"user_id": user_id}
    if lecture_id:
        if not ObjectId.is_valid(lecture_id):
            raise HTTPException(status_code=404, detail="Lecture not found.")
        query["lecture_id"] = ObjectId(lecture_id)
    if method in ("extractive", "abstractive"):
        query["method"] = method

    total = await db[SUMMARIES].count_documents(query)
    cursor = (
        db[SUMMARIES].find(query).sort("created_at", -1)
        .skip(max(0, skip)).limit(max(1, min(limit, 100)))
    )
    docs = [doc async for doc in cursor]

    if docs:
        ids = list({doc["lecture_id"] for doc in docs})
        names = {
            lecture["_id"]: lecture.get("original_filename", "")
            async for lecture in db[LECTURES].find({"_id": {"$in": ids}}, {"original_filename": 1})
        }
        return [summary_public(doc, names.get(doc["lecture_id"])) for doc in docs], total
    return [], total


async def delete_summary(user_id: ObjectId, summary_id: str) -> None:
    doc = await get_summary(user_id, summary_id)
    await db[SUMMARIES].delete_one({"_id": doc["_id"], "user_id": user_id})


async def attach_evaluation(user_id: ObjectId, summary_id: str,
                            evaluation: dict[str, Any]) -> None:
    await db[SUMMARIES].update_one(
        {"_id": ObjectId(summary_id), "user_id": user_id},
        {
            "$set": {
                "last_evaluation": evaluation,
                "last_evaluation_at": datetime.now(timezone.utc),
            }
        },
    )


async def save_quiz(user_id: ObjectId, lecture_id: ObjectId,
                    quiz: dict[str, Any], raw_text: str) -> list[dict[str, Any]]:
    await db[QUIZZES].delete_many({"lecture_id": lecture_id, "user_id": user_id})
    now = datetime.now(timezone.utc)
    docs = [
        {
            "user_id": user_id,
            "lecture_id": lecture_id,
            "question": item["question"],
            "options": item["options"],
            "correct_index": item["correct_index"],
            "correct_answer": item["correct_answer"],
            "explanation": item["explanation"],
            "source_sentence": item["source_sentence"],
            "strategy": item["strategy"],
            "difficulty": item.get("difficulty", "medium"),
            "created_at": now,
        }
        for item in quiz.get("questions", [])
    ]
    if docs:
        await db[QUIZZES].insert_many(docs)
    return [serialize(doc) for doc in docs]


async def get_quiz(user_id: ObjectId, lecture_id: ObjectId) -> list[dict[str, Any]]:
    cursor = db[QUIZZES].find({"lecture_id": lecture_id, "user_id": user_id}).sort("created_at", -1)
    return [serialize(doc) async for doc in cursor]
