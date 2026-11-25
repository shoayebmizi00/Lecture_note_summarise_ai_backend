"""ROUGE evaluation plus non-reference statistics.

ROUGE measures n-gram / longest-common-subsequence overlap between a generated
summary and a human reference summary. It is a proxy for content overlap, not
proof of factual correctness - the API states this explicitly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rouge_score import rouge_scorer

from app.core.logging import get_logger

logger = get_logger(__name__)

READING_WPM = 200


class EvaluationError(Exception):
    pass


@dataclass
class RougeMetric:
    precision: float
    recall: float
    fmeasure: float

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "fmeasure": round(self.fmeasure, 4),
        }


@dataclass
class SummaryStats:
    summary_words: int
    source_words: int | None
    reference_words: int | None
    compression_ratio: float | None
    summary_sentences: int
    reading_time_saved_minutes: float | None
    coverage_of_reference: float | None


@dataclass
class EvaluationResult:
    rouge1: RougeMetric
    rouge2: RougeMetric
    rougeL: RougeMetric
    stats: SummaryStats
    processing_time: float
    method: str | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "rouge1": self.rouge1.as_dict(),
            "rouge2": self.rouge2.as_dict(),
            "rougeL": self.rougeL.as_dict(),
            "rougeLsum": self.rougeL.as_dict(),
            "stats": {
                "summary_words": self.stats.summary_words,
                "source_words": self.stats.source_words,
                "reference_words": self.stats.reference_words,
                "compression_ratio": (
                    round(self.stats.compression_ratio, 4)
                    if self.stats.compression_ratio is not None else None
                ),
                "summary_sentences": self.stats.summary_sentences,
                "reading_time_saved_minutes": (
                    round(self.stats.reading_time_saved_minutes, 1)
                    if self.stats.reading_time_saved_minutes is not None else None
                ),
                "coverage_of_reference": (
                    round(self.stats.coverage_of_reference, 4)
                    if self.stats.coverage_of_reference is not None else None
                ),
            },
            "processing_time": round(self.processing_time, 3),
            "warnings": self.warnings,
            "notes": self.notes,
        }


_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()])


def non_reference_stats(summary_text: str, source_text: str | None) -> dict:
    summary_words = _words(summary_text)
    source_words = _words(source_text) if source_text else None
    compression = (summary_words / source_words) if source_words else None
    saved = ((source_words - summary_words) / READING_WPM) if source_words else None
    return {
        "summary_words": summary_words,
        "summary_sentences": _sentences(summary_text),
        "source_words": source_words,
        "compression_ratio": round(compression, 4) if compression is not None else None,
        "reading_time_saved_minutes": round(saved, 1) if saved is not None and saved > 0 else 0.0,
        "reference_required": True,
        "message": (
            "ROUGE-1/2/L require a human-written reference summary. "
            "Only non-reference statistics are shown for this summary."
        ),
    }


def evaluate(summary_text: str, reference_text: str,
             source_text: str | None = None,
             processing_time: float | None = None,
             method: str | None = None) -> EvaluationResult:
    started = time.perf_counter()
    summary_text = (summary_text or "").strip()
    reference_text = (reference_text or "").strip()
    if not summary_text:
        raise EvaluationError("The generated summary is empty.")
    if not reference_text:
        raise EvaluationError("A reference summary is required for ROUGE evaluation.")
    if len(reference_text.split()) < 5:
        raise EvaluationError("The reference summary is too short to score reliably (minimum 5 words).")

    scores = _scorer.score(target=reference_text, prediction=summary_text)
    summary_words = _words(summary_text)
    source_words = _words(source_text) if source_text else None
    reference_words = _words(reference_text)

    compression = (summary_words / source_words) if source_words else None
    saved = ((source_words - summary_words) / READING_WPM) if source_words else None
    # Share of reference words that appear in the summary (lexical coverage).
    ref_tokens = set(re.findall(r"[a-z0-9]+", reference_text.lower()))
    sum_tokens = set(re.findall(r"[a-z0-9]+", summary_text.lower()))
    coverage = (len(ref_tokens & sum_tokens) / len(ref_tokens)) if ref_tokens else None

    warnings: list[str] = []
    if summary_words < 20:
        warnings.append("The summary is very short, which makes ROUGE scores unstable.")
    if reference_words < summary_words / 4:
        warnings.append("The reference summary is much shorter than the generated summary.")

    notes = [
        "ROUGE measures overlap with the reference summary (unigram, bigram and "
        "longest common subsequence). It does not by itself prove factual correctness.",
        "The reference summary must be written by a human for these scores to be meaningful.",
    ]

    return EvaluationResult(
        rouge1=RougeMetric(**{k: float(v) for k, v in scores["rouge1"]._asdict().items()
                              if k in ("precision", "recall", "fmeasure")}),
        rouge2=RougeMetric(**{k: float(v) for k, v in scores["rouge2"]._asdict().items()
                              if k in ("precision", "recall", "fmeasure")}),
        rougeL=RougeMetric(**{k: float(v) for k, v in scores["rougeL"]._asdict().items()
                              if k in ("precision", "recall", "fmeasure")}),
        stats=SummaryStats(
            summary_words=summary_words,
            source_words=source_words,
            reference_words=reference_words,
            compression_ratio=compression,
            summary_sentences=_sentences(summary_text),
            reading_time_saved_minutes=saved,
            coverage_of_reference=coverage,
        ),
        processing_time=max(0.0, (time.perf_counter() - started) + (processing_time or 0.0)),
        method=method,
        warnings=warnings,
        notes=notes,
    )


def compare_evaluations(entries: list[dict], reference_text: str,
                        source_text: str | None = None) -> dict:
    """Score several summaries (e.g. TextRank vs BART) against one reference."""
    results: list[dict] = []
    for entry in entries:
        evaluation = evaluate(
            summary_text=entry["summary_text"],
            reference_text=reference_text,
            source_text=source_text,
            processing_time=entry.get("processing_time"),
            method=entry.get("method"),
        )
        payload = evaluation.as_dict()
        payload["summary_id"] = entry.get("summary_id")
        payload["summary_length"] = entry.get("summary_length")
        results.append(payload)

    best = max(results, key=lambda item: item["rougeL"]["fmeasure"]) if results else None
    return {
        "results": results,
        "best_by_rougeL": best["method"] if best else None,
        "reference_words": _words(reference_text),
        "notes": [
            "All summaries were scored against the same reference summary.",
            "A higher ROUGE score means closer lexical overlap, not necessarily a better or "
            "more factually correct summary.",
        ],
    }
