"""Schemas for lectures, jobs and analysis requests."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SummaryMethod = Literal["extractive", "abstractive"]
SummaryLength = Literal["short", "medium", "long"]


class LectureOut(BaseModel):
    id: str
    original_filename: str
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    file_size: int = 0
    upload_date: datetime | str | None = None
    processing_status: str = "ready"
    ocr_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    preview: str | None = None
    summary_count: int = 0


class LectureDetail(LectureOut):
    extracted_text: str = ""
    headings: list[dict] = Field(default_factory=list)
    pages: list[dict] = Field(default_factory=list)
    metadata_info: dict = Field(default_factory=dict)


class SummaryOut(BaseModel):
    id: str
    lecture_id: str
    method: str
    summary_text: str
    summary_length: str
    processing_time: float
    created_at: datetime | str | None = None
    model_name: str | None = None
    algorithm: str | None = None
    compression_ratio: float | None = None
    summary_word_count: int | None = None
    source_word_count: int | None = None
    chunk_count: int | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    lecture_filename: str | None = None


class KeywordOut(BaseModel):
    keyword: str
    score: float
    frequency: int
    kind: str = "term"
    method: str = "tfidf"


class QuizQuestionOut(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    correct_answer: str
    explanation: str
    source_sentence: str
    strategy: str
    difficulty: str = "medium"


class JobOut(BaseModel):
    id: str
    type: str
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = 0
    stage: str = "queued"
    message: str | None = None
    lecture_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class UploadResponse(BaseModel):
    lecture: LectureOut
    warnings: list[str] = Field(default_factory=list)
    text_preview: str = ""


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    method: SummaryMethod = "extractive"
    length: SummaryLength = "medium"
    save: bool = True


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    length: SummaryLength = "medium"


class KeywordsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    top_n: int = Field(default=15, ge=3, le=40)
    use_keybert: bool | None = None
    save: bool = True


class TopicsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    max_topics: int = Field(default=6, ge=2, le=12)


class QuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=20)
    seed: int | None = Field(default=None, ge=0, le=10_000_000)
    save: bool = True


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_summary: str = Field(min_length=10, max_length=20_000)
    summary_id: str | None = None
    summary_text: str | None = Field(default=None, max_length=100_000)
    lecture_id: str | None = None

    @model_validator(mode="after")
    def _need_summary(self) -> "EvaluateRequest":
        if not self.summary_id and not (self.summary_text and self.summary_text.strip()):
            raise ValueError("Provide either summary_id or summary_text.")
        return self


class CompareEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str = Field(min_length=1)
    reference_summary: str = Field(min_length=10, max_length=20_000)


class MessageResponse(BaseModel):
    message: str
