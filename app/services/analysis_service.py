"""Orchestrates preprocessing -> summarisation -> keywords for a lecture."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from app.core.logging import get_logger
from app.nlp.abstractive import ModelUnavailableError, get_summarizer
from app.nlp.extractive import ExtractiveResult, summarize
from app.nlp.keywords import extract_keywords
from app.nlp.preprocessing import PreprocessedDocument, preprocess

logger = get_logger(__name__)

ProgressFn = Callable[[int, str, str], None]

ALGORITHM_LABELS = {
    "extractive": "TextRank (TF-IDF cosine graph + PageRank)",
}


def document_info(doc: PreprocessedDocument, preprocessing_seconds: float) -> dict[str, Any]:
    avg_words = (doc.word_count / doc.sentence_count) if doc.sentence_count else 0.0
    return {
        "word_count": doc.word_count,
        "char_count": doc.char_count,
        "sentence_count": doc.sentence_count,
        "average_sentence_words": round(avg_words, 1),
        "duplicates_removed": doc.duplicates_removed,
        "noise_removed": doc.noise_removed,
        "preprocessing_time": round(preprocessing_seconds, 3),
        "reading_time_minutes": round(doc.word_count / 200, 1),
        "warnings": doc.warnings,
    }


def _extractive_payload(result: ExtractiveResult) -> dict[str, Any]:
    payload = result.as_dict()
    payload.update(
        {
            "algorithm": ALGORITHM_LABELS["extractive"],
            "model_name": None,
            "requested_model": None,
            "fallback_used": False,
            "device": "cpu",
            "chunk_count": 0,
            "chunk_summaries": [],
            "source_sentence_count": result.source_sentence_count,
        }
    )
    # Keep the stored document compact: top 120 sentence scores are enough for the UI.
    payload["sentence_scores"] = payload["sentence_scores"][:120]
    return payload


def run_analysis(
    text: str,
    methods: Iterable[str],
    length: str,
    *,
    top_n: int = 15,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Blocking NLP pipeline (executed in a worker thread by the job manager)."""
    started = time.perf_counter()
    methods = [m for m in methods if m in ("extractive", "abstractive")]
    if not methods:
        raise ValueError("At least one summarisation method is required.")

    def report(percent: int, stage: str, message: str) -> None:
        if progress:
            progress(percent, stage, message)

    report(5, "preprocessing", "Cleaning text and segmenting sentences")
    preprocess_started = time.perf_counter()
    doc = preprocess(text)
    preprocessing_seconds = time.perf_counter() - preprocess_started
    if not doc.sentences:
        raise ValueError(
            "No usable sentences were found in this lecture. The PDF may contain only images."
        )
    report(12, "preprocessing", f"Found {doc.sentence_count} sentences")

    variants: list[dict[str, Any]] = []
    span = 76 / len(methods)

    for i, method in enumerate(methods):
        base = 14 + i * span

        def sub_progress(p: int, stage: str, message: str, base=base) -> None:
            report(int(base + span * max(0, min(100, p)) / 100), stage, message)

        if method == "extractive":
            sub_progress(20, "textrank", "Building TF-IDF similarity graph")
            result = summarize(doc.sentences, length)
            sub_progress(100, "textrank", "TextRank ranking complete")
            variants.append(_extractive_payload(result))
        else:
            summarizer = get_summarizer()
            result = summarizer.summarize(doc.sentences, length, progress=sub_progress)
            payload = result.as_dict()
            payload["algorithm"] = f"Abstractive (BART seq2seq: {result.model_name})"
            payload["source_sentence_count"] = doc.sentence_count
            payload["sentence_scores"] = []
            variants.append(payload)

    report(94, "keywords", "Extracting keywords")
    keyword_result = extract_keywords(doc.sentences, doc.clean_text, top_n=top_n)
    keywords = [k.as_dict() for k in keyword_result.keywords]

    total = time.perf_counter() - started
    report(99, "finalizing", "Preparing results")
    return {
        "variants": variants,
        "keywords": keywords,
        "keyword_method": keyword_result.method,
        "document": document_info(doc, preprocessing_seconds),
        "total_processing_time": round(total, 3),
    }


def run_keywords(text: str, top_n: int = 15, use_keybert: bool | None = None) -> dict[str, Any]:
    doc = preprocess(text)
    if not doc.sentences:
        raise ValueError("No usable text was found in this lecture.")
    result = extract_keywords(doc.sentences, doc.clean_text, top_n=top_n, use_keybert=use_keybert)
    return {
        "keywords": [k.as_dict() for k in result.keywords],
        "method": result.method,
        "total_candidates": result.total_candidates,
        "warnings": result.warnings,
        "document": document_info(doc, 0.0),
    }


def run_topics(text: str, font_headings: list[str] | None = None,
               max_topics: int = 6) -> dict[str, Any]:
    from app.nlp.topic_modeling import generate_topics

    doc = preprocess(text)
    if not doc.sentences:
        raise ValueError("No usable text was found in this lecture.")
    result = generate_topics(doc.clean_text, doc.sentences, font_headings, max_topics)
    return result.as_dict()


def run_quiz(text: str, count: int = 5, seed: int | None = None,
             top_n_keywords: int = 18) -> dict[str, Any]:
    from app.nlp.quiz_generator import generate_quiz

    doc = preprocess(text)
    if not doc.sentences:
        raise ValueError("No usable text was found in this lecture.")
    kw = extract_keywords(doc.sentences, doc.clean_text, top_n=top_n_keywords)
    result = generate_quiz(
        doc.sentences,
        kw.keywords,
        count=count,
        seed=seed if seed is not None else 42,
    )
    payload = result.as_dict()
    payload["document"] = {
        "sentence_count": doc.sentence_count,
        "word_count": doc.word_count,
    }
    return payload


def model_status() -> dict[str, Any]:
    try:
        return get_summarizer().status()
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not read model status: %s", exc)
        return {"loaded": False, "last_error": str(exc)}


__all__ = [
    "ModelUnavailableError",
    "document_info",
    "model_status",
    "run_analysis",
    "run_keywords",
    "run_quiz",
    "run_topics",
]
