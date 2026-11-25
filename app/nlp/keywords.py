"""Keyword extraction: TF-IDF over sentences, optional KeyBERT semantic boost."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_EXTRA_STOPWORDS = {
    "figure", "figures", "table", "tables", "page", "pages", "slide", "slides",
    "chapter", "section", "lecture", "note", "notes", "example", "examples",
    "following", "shown", "figure", "above", "below", "also", "may", "can",
    "one", "two", "three", "first", "second", "third", "etc", "ie", "eg",
    "using", "used", "use", "based", "given", "however", "therefore", "thus",
    "will", "would", "could", "should", "shall", "might", "must", "many", "much",
    "within", "without", "among", "between", "across", "during", "since",
    "these", "those", "this", "that", "there", "their", "them", "they",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | _EXTRA_STOPWORDS

_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


@dataclass
class Keyword:
    keyword: str
    score: float
    frequency: int
    kind: str = "term"
    method: str = "tfidf"

    def as_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "score": round(self.score, 4),
            "frequency": self.frequency,
            "kind": self.kind,
            "method": self.method,
        }


@dataclass
class KeywordResult:
    keywords: list[Keyword]
    method: str
    total_candidates: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "keywords": [k.as_dict() for k in self.keywords],
            "method": self.method,
            "total_candidates": self.total_candidates,
            "warnings": self.warnings,
        }


def _valid_term(term: str) -> bool:
    term = term.strip()
    if len(term) < 3 or len(term) > 60:
        return False
    words = [w for w in re.split(r"[\s\-]+", term) if w]
    if not words or len(words) > 3:
        return False
    if not any(len(w) >= 3 for w in words):
        return False
    if all(w.lower() in STOPWORDS for w in words):
        return False
    if words[0].lower() in STOPWORDS or words[-1].lower() in STOPWORDS:
        return False
    if term.lower() in STOPWORDS:
        return False
    if not re.search(r"[A-Za-z]{3,}", term):
        return False
    # Reject pure numbers / roman numerals / formula fragments.
    if not re.search(r"[A-Za-z]", term):
        return False
    return True


def _count_occurrences(term: str, text: str) -> int:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _stem_key(term: str) -> str:
    key = term.lower().strip()
    for suffix in ("ies", "es", "s"):
        if len(key) > 4 and key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _keybert_candidates(text: str, top_n: int) -> list[tuple[str, float]] | None:
    if not settings.enable_keybert:
        return None
    try:
        from keybert import KeyBERT

        model = KeyBERT(model=settings.keybert_model)
        pairs = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n * 2,
        )
        return [(term, float(score)) for term, score in pairs]
    except Exception as exc:
        logger.warning("KeyBERT unavailable, continuing with TF-IDF only: %s", exc)
        return None


def extract_keywords(sentences: list[str], full_text: str | None = None,
                     top_n: int = 15, use_keybert: bool | None = None) -> KeywordResult:
    sentences = [s for s in sentences if s and s.strip()]
    text = full_text or " ".join(sentences)
    warnings: list[str] = []
    if not sentences:
        return KeywordResult(keywords=[], method="tfidf", total_candidates=0,
                             warnings=["No text available for keyword extraction."])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.92,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9'\-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
        terms = vectorizer.get_feature_names_out()
        scores = np.asarray(matrix.sum(axis=0)).ravel()
    except ValueError:
        return KeywordResult(keywords=[], method="tfidf", total_candidates=0,
                             warnings=["The text did not contain enough distinct words for keywords."])

    # Phrase bonus: multi-word terms carry more topical information.
    adjusted = []
    for term, score in zip(terms, scores):
        if not _valid_term(term):
            continue
        word_count = len(term.split())
        bonus = 1.0 + 0.08 * (word_count - 1)
        adjusted.append((term, float(score) * bonus))

    if not adjusted:
        return KeywordResult(keywords=[], method="tfidf", total_candidates=0,
                             warnings=["No meaningful keywords were found."])

    max_score = max(score for _, score in adjusted) or 1.0
    # Drop singular/plural duplicates, keeping the higher scoring form.
    best: dict[str, tuple[str, float]] = {}
    for term, score in adjusted:
        key = _stem_key(term)
        if key not in best or score > best[key][1]:
            best[key] = (term, score)

    candidates = sorted(best.values(), key=lambda item: item[1], reverse=True)[: top_n * 3]

    keybert = _keybert_candidates(text, top_n) if (use_keybert is None or use_keybert) else None
    semantic: dict[str, float] = {}
    method = "tfidf"
    if keybert:
        method = "tfidf+keybert"
        semantic = {term.lower(): score for term, score in keybert}

    keywords: list[Keyword] = []
    seen: set[str] = set()
    raw_scores: list[tuple[Keyword, float]] = []
    for term, score in candidates:
        norm = _stem_key(term)
        if norm in seen:
            continue
        seen.add(norm)
        frequency = _count_occurrences(term, text)
        if frequency == 0:
            continue
        tfidf_score = score / max_score
        if semantic:
            sem = semantic.get(term.lower(), 0.0)
            final = 0.65 * tfidf_score + 0.35 * max(0.0, min(1.0, sem))
        else:
            freq_weight = 1.0 + math.log1p(frequency) * 0.12
            final = tfidf_score * freq_weight
        raw_scores.append((
            Keyword(
                keyword=term,
                score=0.0,
                frequency=frequency,
                kind="phrase" if " " in term else "term",
                method="tfidf+keybert" if semantic else "tfidf",
            ),
            final,
        ))

    top_raw = max((score for _, score in raw_scores), default=1.0) or 1.0
    for keyword, raw in raw_scores:
        keyword.score = round(min(1.0, raw / top_raw), 4)
        keywords.append(keyword)

    keywords.sort(key=lambda k: k.score, reverse=True)
    keywords = keywords[:top_n]
    if not keywords:
        warnings.append("No meaningful keywords were found in this document.")
    return KeywordResult(keywords=keywords, method=method,
                         total_candidates=len(adjusted), warnings=warnings)
