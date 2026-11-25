"""Genuine TextRank extractive summarisation (TF-IDF + cosine graph + PageRank)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.logging import get_logger

logger = get_logger(__name__)

LENGTH_PRESETS: dict[str, dict[str, float | int]] = {
    "short": {"ratio": 0.10, "max_sentences": 8, "min_sentences": 2},
    "medium": {"ratio": 0.20, "max_sentences": 16, "min_sentences": 3},
    "long": {"ratio": 0.30, "max_sentences": 28, "min_sentences": 4},
}

SIMILARITY_FLOOR = 0.05
REDUNDANCY_THRESHOLD = 0.82


class TextRankError(Exception):
    pass


@dataclass
class ScoredSentence:
    index: int
    text: str
    score: float
    selected: bool = False

    def as_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "score": round(self.score, 5),
                "selected": self.selected}


@dataclass
class ExtractiveResult:
    summary_text: str
    method: str
    length: str
    selected: list[ScoredSentence]
    sentences: list[ScoredSentence]
    target_sentences: int
    ratio: float
    source_sentence_count: int
    source_word_count: int
    summary_word_count: int
    compression_ratio: float
    processing_time: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "length": self.length,
            "summary_text": self.summary_text,
            "target_sentences": self.target_sentences,
            "ratio": self.ratio,
            "source_sentence_count": self.source_sentence_count,
            "source_word_count": self.source_word_count,
            "summary_word_count": self.summary_word_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "processing_time": round(self.processing_time, 3),
            "selected_sentences": [s.as_dict() for s in self.selected],
            "sentence_scores": [s.as_dict() for s in self.sentences],
            "warnings": self.warnings,
        }


def similarity_matrix(sentences: list[str]) -> np.ndarray:
    """TF-IDF vectors -> cosine similarity, with weak edges pruned to zero."""
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0))
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w'-]{1,}\b",
        )
        matrix = vectorizer.fit_transform(sentences)
        sim = cosine_similarity(matrix)
    except ValueError:
        # Empty vocabulary after stop-word removal (e.g. a formula-only page).
        sim = np.zeros((n, n))
    if sim.shape != (n, n):
        sim = np.zeros((n, n))
    np.fill_diagonal(sim, 0.0)
    sim[sim < SIMILARITY_FLOOR] = 0.0
    return sim


def textrank_scores(sentences: list[str], sim: np.ndarray | None = None,
                    damping: float = 0.85) -> np.ndarray:
    n = len(sentences)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    sim = similarity_matrix(sentences) if sim is None else sim

    graph = nx.from_numpy_array(sim)
    if graph.number_of_edges() == 0:
        # No lexical overlap - fall back to a positional/centrality-free ranking.
        return np.ones(n) / n
    try:
        pr = nx.pagerank(graph, alpha=damping, weight="weight", max_iter=300, tol=1.0e-6)
        scores = np.array([pr.get(i, 0.0) for i in range(n)], dtype=float)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge; using degree centrality")
        scores = np.array([d for _, d in graph.degree(weight="weight")], dtype=float)
    max_score = scores.max(initial=0.0)
    if max_score <= 0:
        return np.ones(n) / n
    return scores / max_score


def summarize(sentences: list[str], length: str = "medium",
              *, min_sentences: int | None = None,
              max_sentences: int | None = None,
              redundancy_threshold: float = REDUNDANCY_THRESHOLD) -> ExtractiveResult:
    """Rank sentences with TextRank, pick the top-k, restore document order."""
    started = time.perf_counter()
    length = (length or "medium").lower()
    if length not in LENGTH_PRESETS:
        length = "medium"
    preset = LENGTH_PRESETS[length]

    sentences = [s.strip() for s in sentences if s and s.strip()]
    n = len(sentences)
    warnings: list[str] = []
    if n == 0:
        raise TextRankError("There are no sentences to summarise.")

    ratio = float(preset["ratio"])
    max_k = int(max_sentences if max_sentences is not None else preset["max_sentences"])
    min_k = int(min_sentences if min_sentences is not None else preset["min_sentences"])

    if n <= 3:
        target = n
        warnings.append("The document is very short, so almost every sentence was kept.")
    else:
        target = int(round(n * ratio))
        target = max(min_k, min(max_k, target))
        target = min(target, n)

    sim = similarity_matrix(sentences)
    scores = textrank_scores(sentences, sim)

    ranked = np.argsort(-scores)
    selected_idx: list[int] = []
    for idx in ranked:
        if len(selected_idx) >= target:
            break
        candidate_vec = sim[idx] if sim.size else None
        redundant = False
        if candidate_vec is not None:
            for chosen in selected_idx:
                if candidate_vec[chosen] >= redundancy_threshold:
                    redundant = True
                    break
        if redundant and len(ranked) - len(selected_idx) > target:
            continue
        selected_idx.append(int(idx))

    if not selected_idx:
        selected_idx = [int(i) for i in ranked[:target]]

    selected_idx.sort()  # restore original document order
    selected_set = set(selected_idx)

    scored = [
        ScoredSentence(index=i, text=sentences[i], score=float(scores[i]),
                       selected=i in selected_set)
        for i in range(n)
    ]
    selected = [s for s in scored if s.selected]
    summary_text = " ".join(s.text for s in selected)
    source_words = sum(len(s.split()) for s in sentences)
    summary_words = len(summary_text.split())

    elapsed = time.perf_counter() - started
    return ExtractiveResult(
        summary_text=summary_text,
        method="textrank",
        length=length,
        selected=selected,
        sentences=scored,
        target_sentences=target,
        ratio=ratio,
        source_sentence_count=n,
        source_word_count=source_words,
        summary_word_count=summary_words,
        compression_ratio=(summary_words / source_words) if source_words else 0.0,
        processing_time=elapsed,
        warnings=warnings,
    )
