"""Topic-wise summarisation: heading segmentation with TF-IDF/K-Means fallback.

Titles come either from real headings found in the document or from the top
TF-IDF terms of a cluster. Cluster labels are *derived*, not invented headings -
the API reports which source was used so the UI can say so honestly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.logging import get_logger
from app.nlp.extractive import summarize
from app.nlp.preprocessing import split_sentences

logger = get_logger(__name__)

_HEADING_PREFIX = re.compile(
    r"^(?:chapter|section|unit|lecture|module|part|topic|lesson|week)\s+\d+",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?|[IVXLC]{1,6}\.)\s+[A-Z]")
_SENTENCE_END = (".", "?", "!", ";", ",")


@dataclass
class TopicSection:
    index: int
    title: str
    summary: str
    keywords: list[str]
    sentence_count: int
    word_count: int
    source: str  # "heading" | "cluster" | "document"
    start_sentence: int = 0
    sentences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "source": self.source,
            "start_sentence": self.start_sentence,
        }


@dataclass
class TopicResult:
    topics: list[TopicSection]
    method: str
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "topics": [t.as_dict() for t in self.topics],
            "method": self.method,
            "topic_count": len(self.topics),
            "notes": self.notes,
            "warnings": self.warnings,
        }


def _is_heading(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text or len(text) > 100:
        return False
    if text[:1].islower():
        return False
    words = text.split()
    if not 1 <= len(words) <= 14:
        return False
    if _HEADING_PREFIX.match(text) or _NUMBERED.match(text):
        return True
    if text.endswith(":") and len(words) <= 10:
        return True
    if text.endswith(_SENTENCE_END):
        return False
    letters = [c for c in text if c.isalpha()]
    if letters and (sum(c.isupper() for c in letters) / len(letters)) > 0.8:
        return True
    # Title Case short lines, e.g. "Gradient Descent" / "Evaluation"
    capitalised = sum(1 for w in words if w[:1].isupper())
    if len(words) >= 2 and capitalised >= max(1, len(words) - 1) and not text.endswith("."):
        return True
    if len(words) == 1 and text[:1].isupper() and len(text) <= 40:
        return True
    return False


def _clean_title(paragraph: str) -> str:
    title = re.sub(r"\s+", " ", paragraph).strip().rstrip(":")
    title = re.sub(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+\.)\s*", "", title)
    return title[:90].strip() or "Untitled section"


def _section_summary(sentences: list[str]) -> str:
    if not sentences:
        return ""
    if len(sentences) <= 4:
        return " ".join(sentences)
    try:
        result = summarize(sentences, length="short", min_sentences=2, max_sentences=6)
        return result.summary_text
    except Exception as exc:
        logger.warning("Section summarisation failed: %s", exc)
        return " ".join(sentences[:3])


def _section_keywords(sentences: list[str], limit: int = 5) -> list[str]:
    if not sentences:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9'\-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = np.argsort(-scores)
        out: list[str] = []
        for idx in order:
            term = str(terms[idx])
            if len(term) < 4 or term.isdigit():
                continue
            if any(term.lower() in existing.lower() for existing in out):
                continue
            out.append(term)
            if len(out) >= limit:
                break
        return out
    except ValueError:
        return []


def _build_sections_from_headings(paragraphs: list[str], font_set: set[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []

    def flush():
        nonlocal current_title, current_body
        if current_body:
            sections.append((current_title or "Opening", current_body))
            current_body = []

    for para in paragraphs:
        norm = re.sub(r"\s+", " ", para).strip().lower()
        # Font-based headings only count on an exact paragraph match; substring
        # matching would promote ordinary sentences to headings.
        is_font_heading = norm in font_set and len(para) < 120
        if _is_heading(para) or is_font_heading:
            if current_body:
                flush()
            current_title = _clean_title(para)
            continue
        current_body.append(para)
    flush()
    return sections


def _cluster_sections(sentences: list[str], max_topics: int = 6) -> list[tuple[str, list[str], int]]:
    n = len(sentences)
    if n < 8:
        return [("Full document", sentences, 0)]

    k = max(2, min(max_topics, int(round(math.sqrt(n / 3))), max(2, n // 4)))
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9'\-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
        if matrix.shape[1] < k:
            return [("Full document", sentences, 0)]
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        terms = vectorizer.get_feature_names_out()
    except Exception as exc:
        logger.warning("Clustering failed (%s); using a single section", exc)
        return [("Full document", sentences, 0)]

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    ordered = sorted(groups.items(), key=lambda item: min(item[1]))
    out: list[tuple[str, list[str], int]] = []
    for cluster_id, indices in ordered:
        indices.sort()
        centroid = km.cluster_centers_[cluster_id]
        top_idx = np.argsort(-centroid)[:8]
        label_terms: list[str] = []
        for idx in top_idx:
            term = str(terms[idx])
            if len(term) < 4 or term.isdigit():
                continue
            if any(term.lower() in existing.lower() for existing in label_terms):
                continue
            label_terms.append(term)
            if len(label_terms) == 3:
                break
        title = " · ".join(word.capitalize() for word in label_terms) if label_terms else f"Topic {len(out) + 1}"
        out.append((title, [sentences[i] for i in indices], indices[0]))
    return out


def generate_topics(clean_text: str, sentences: list[str],
                    font_headings: list[str] | None = None,
                    max_topics: int = 6) -> TopicResult:
    notes: list[str] = []
    warnings: list[str] = []
    if not sentences:
        return TopicResult(topics=[], method="none", notes=["No text available."])

    font_set = {re.sub(r"\s+", " ", h).strip().lower() for h in (font_headings or [])}
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    heading_sections = _build_sections_from_headings(paragraphs, font_set) if len(paragraphs) > 1 else []

    topics: list[TopicSection] = []
    meaningful = [s for s in heading_sections if sum(len(p.split()) for p in s[1]) >= 30]

    if len(meaningful) >= 2:
        method = "headings"
        notes.append("Sections follow headings detected in the document (font size / numbering).")
        for title, body in meaningful[:14]:
            body_sentences = split_sentences(" ".join(body))
            if not body_sentences:
                continue
            topics.append(
                TopicSection(
                    index=len(topics),
                    title=title,
                    summary=_section_summary(body_sentences),
                    keywords=_section_keywords(body_sentences),
                    sentence_count=len(body_sentences),
                    word_count=sum(len(s.split()) for s in body_sentences),
                    source="heading",
                )
            )
    else:
        method = "clusters"
        notes.append(
            "No reliable headings were found, so sentences were grouped by TF-IDF similarity "
            "(K-Means). Titles are derived from the most distinctive terms of each group, "
            "not authored headings."
        )
        clusters = _cluster_sections(sentences, max_topics)
        cursor = 0
        for title, cluster_sentences, start in clusters:
            if not cluster_sentences:
                continue
            topics.append(
                TopicSection(
                    index=len(topics),
                    title=title,
                    summary=_section_summary(cluster_sentences),
                    keywords=_section_keywords(cluster_sentences),
                    sentence_count=len(cluster_sentences),
                    word_count=sum(len(s.split()) for s in cluster_sentences),
                    source="cluster" if len(clusters) > 1 else "document",
                    start_sentence=start,
                )
            )
            cursor += len(cluster_sentences)

    if not topics:
        warnings.append("Topic segmentation produced no sections.")
    if len(topics) == 1:
        notes.append("The document appears to cover a single topic.")
    return TopicResult(topics=topics, method=method, notes=notes, warnings=warnings)
