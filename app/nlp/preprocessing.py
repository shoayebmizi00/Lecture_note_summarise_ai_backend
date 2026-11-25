"""Text cleaning, sentence segmentation and noise/duplicate removal.

Design note: we do NOT strip stopwords or punctuation from the text that is
given to transformer models. Only layout noise (line breaks, running headers,
page numbers, duplicated boilerplate) is removed, so technical terms, numbers,
formulas, definitions and negations survive intact.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00ad": "", "\ufeff": "",
}
_BULLET = re.compile(r"^\s*(?:[\u2022\u25cf\u25aa\u2023\u2043\-\*]\s+|\d+[.)]\s+)")
_WS = re.compile(r"[ \t\u00a0]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PAGE_JUNK = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.IGNORECASE)
_SPACED_OUT = re.compile(r"\b(?:\w\s){4,}\w\b")  # "T h e   c a t"

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "fig", "figs", "eq", "eqs", "no", "vol", "ch", "sec",
    "dr", "mr", "mrs", "ms", "prof", "al", "approx", "dept", "univ", "inc", "ltd",
    "ref", "refs", "app", "tab", "pp", "cf", "viz", "ie", "eg",
}

_nltk_ready = False


def ensure_nltk() -> bool:
    """Make sure the punkt tokenizer data is available (NLTK >= 3.9 needs punkt_tab)."""
    global _nltk_ready
    if _nltk_ready:
        return True
    try:
        import nltk
        from nltk.data import find

        for resource, pkg in (("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")):
            try:
                find(resource)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception as exc:  # offline
                    logger.warning("Could not download NLTK resource %s: %s", pkg, exc)
        _nltk_ready = True
    except Exception as exc:
        logger.warning("NLTK unavailable, using regex sentence splitter: %s", exc)
        _nltk_ready = False
    return _nltk_ready


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _WS.sub(" ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        if _PAGE_JUNK.match(line):
            continue
        if _SPACED_OUT.search(line) and len(line) < 120:
            line = re.sub(r"\s+", "", line)
        line = _BULLET.sub("", line)
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Join wrapped lines: keep paragraph breaks, merge soft line breaks.
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    joined = []
    for para in paragraphs:
        merged = " ".join(line for line in para.split("\n") if line.strip())
        merged = re.sub(r"\s+", " ", merged).strip()
        if merged:
            joined.append(merged)
    text = "\n\n".join(joined)
    # Re-attach sentence fragments broken across line wraps.
    text = re.sub(r"([a-z,;])\s*\n\s*([a-z])", r"\1 \2", text)
    text = _WS.sub(" ", text)
    return text.strip()


def _regex_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation that tolerates abbreviations, decimals and formulas."""
    text = text.strip()
    if not text:
        return []
    if ensure_nltk():
        try:
            from nltk.tokenize import sent_tokenize

            pieces = sent_tokenize(text, language="english")
        except Exception:
            pieces = _regex_split(text)
    else:
        pieces = _regex_split(text)

    # NLTK sometimes splits after "e.g." / "Fig." - repair obvious cases.
    repaired: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if repaired:
            last_word = re.split(r"[\s(]", repaired[-1])[-2].lower().strip(".,;:()")
            prev_ends_abbrev = re.search(r"\b([A-Za-z]{1,5})\.$", repaired[-1]) is not None
            if prev_ends_abbrev and last_word in _ABBREVIATIONS:
                repaired[-1] = f"{repaired[-1]} {piece}"
                continue
            if repaired[-1].endswith(("Fig.", "Eq.", "No.", "vs.")):
                repaired[-1] = f"{repaired[-1]} {piece}"
                continue
        repaired.append(piece)
    return repaired


def _sentence_key(sentence: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", sentence.lower())
    return key[:220]


def _token_set(sentence: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if len(w) > 2}


def is_noise(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < 20:
        return True
    if _PAGE_JUNK.match(stripped):
        return True
    # Standalone heading lines ("Supervised Learning") are not summary sentences.
    if len(stripped.split()) <= 10 and not stripped.endswith((".", "!", "?", ";", ":")):
        if stripped[:1].isupper() or stripped[:1].isdigit():
            return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < len(stripped) * 0.35:
        return True
    if len(set(stripped.lower().split())) <= 1 and len(stripped) < 80:
        return True
    low = stripped.lower()
    junk_prefixes = ("figure ", "table ", "source:", "downloaded from", "copyright",
                     "all rights reserved", "slide ", "lecture slide")
    if len(stripped) < 90 and low.startswith(junk_prefixes):
        return True
    return False


@dataclass
class PreprocessedDocument:
    clean_text: str
    sentences: list[str]
    word_count: int
    char_count: int
    duplicates_removed: int = 0
    noise_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def preprocess(text: str) -> PreprocessedDocument:
    warnings: list[str] = []
    clean = normalize_text(text)
    raw_sentences: list[str] = []
    # Split paragraph by paragraph: headings never get glued onto body sentences.
    for paragraph in clean.split("\n\n"):
        if paragraph.strip():
            raw_sentences.extend(split_sentences(paragraph))

    sentences: list[str] = []
    seen_keys: set[str] = set()
    seen_token_sets: list[set[str]] = []
    noise_removed = 0
    duplicates_removed = 0

    for sentence in raw_sentences:
        if is_noise(sentence):
            noise_removed += 1
            continue
        key = _sentence_key(sentence)
        if key and key in seen_keys:
            duplicates_removed += 1
            continue
        tokens = _token_set(sentence)
        if tokens and len(tokens) >= 5:
            duplicate = False
            for prev in seen_token_sets[-120:]:
                if not prev:
                    continue
                overlap = len(tokens & prev) / len(tokens | prev)
                if overlap >= 0.9:
                    duplicate = True
                    break
            if duplicate:
                duplicates_removed += 1
                continue
            seen_token_sets.append(tokens)
        seen_keys.add(key)
        sentences.append(sentence)

    if len(sentences) < 4:
        warnings.append(
            "Very little usable text was found - the summary may be unreliable."
        )

    return PreprocessedDocument(
        clean_text=clean,
        sentences=sentences,
        word_count=len(clean.split()),
        char_count=len(clean),
        duplicates_removed=duplicates_removed,
        noise_removed=noise_removed,
        warnings=warnings,
    )
