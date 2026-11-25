"""Typed documentation of the MongoDB document shapes.

MongoDB is schemaless, so these Pydantic models describe the contract that the
services write. They are used by `docs/DATABASE.md` and are handy for future
migrations; the API schemas in `app/schemas` define what leaves the server.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SummaryMethod = Literal["extractive", "abstractive"]
SummaryLength = Literal["short", "medium", "long"]


class UserDocument(BaseModel):
    id: str | None = None
    username: str
    username_lower: str
    email: str
    password_hash: str
    created_at: datetime
    last_login_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class LectureDocument(BaseModel):
    id: str | None = None
    user_id: str
    original_filename: str
    stored_filename: str
    extracted_text: str
    page_count: int
    char_count: int
    word_count: int
    sentence_count: int
    file_size: int
    upload_date: datetime
    processing_status: Literal["ready", "failed"] = "ready"
    ocr_used: bool = False
    ocr_pages: int = 0
    warnings: list[str] = Field(default_factory=list)
    headings: list[dict[str, Any]] = Field(default_factory=list)
    pages: list[dict[str, Any]] = Field(default_factory=list)
    metadata_info: dict[str, Any] = Field(default_factory=dict)


class SummaryDocument(BaseModel):
    id: str | None = None
    user_id: str
    lecture_id: str
    lecture_filename: str
    method: SummaryMethod
    algorithm: str | None = None
    summary_text: str
    summary_length: SummaryLength
    processing_time: float
    created_at: datetime
    model_name: str | None = None
    requested_model: str | None = None
    fallback_used: bool = False
    device: str | None = None
    chunk_count: int = 0
    chunk_summaries: list[str] = Field(default_factory=list)
    compression_ratio: float | None = None
    summary_word_count: int | None = None
    source_word_count: int | None = None
    selected_sentences: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    last_evaluation: dict[str, Any] | None = None


class KeywordDocument(BaseModel):
    id: str | None = None
    user_id: str
    lecture_id: str
    keyword: str
    score: float
    frequency: int
    kind: str = "term"
    method: str = "tfidf"
    created_at: datetime


class QuizDocument(BaseModel):
    id: str | None = None
    user_id: str
    lecture_id: str
    question: str
    options: list[str]  # stored as a structured array (4 options)
    correct_index: int
    correct_answer: str
    explanation: str
    source_sentence: str
    strategy: str
    difficulty: str = "medium"
    created_at: datetime


class JobDocument(BaseModel):
    job_id: str
    user_id: str
    type: Literal["summarize", "compare"]
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = 0
    stage: str = "queued"
    message: str | None = None
    lecture_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
