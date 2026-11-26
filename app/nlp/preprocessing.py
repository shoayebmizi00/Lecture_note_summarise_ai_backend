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
_CODE_HINT = re.compile(
    r"(?:^|\s)(?:import|from)\s+[A-Za-z_]|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:(]|"
    r"\bprint\s*\(|>>>|->|::|\u2192|\bfor\s+\w+\s+in\s+|"
    r"\b\w+\s*=\s*[A-Za-z_][\w.]*\s*\(|\b\w+\s*=\s*[\[{(\"']",
    re.IGNORECASE,
)
_POS_TAG = re.compile(r"\b[A-Za-z][\w'-]*/[A-Z$]{1,6}\b")
_PARSE_TREE = re.compile(r"\(\s*[A-Z$]{1,5}\s")
_GRAMMAR_RULE = re.compile(r"\b[A-Z]{1,4}:\s+[A-Z][A-Za-z]*\b")
_CODE_SYMBOLS = re.compile(r"[(){}\[\]<>=|\\/*#^~]")
_NON_LATIN = re.compile(r"[^\x00-\x7f\u00c0-\u024f\u2000-\u206f]")

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


def _merge_block_lines(lines: list[str]) -> str:
    """Join PDF line wraps but keep slide bullets / headings on their own line.

    A wrapped prose line usually ends in a lowercase word, a comma or an opening
    bracket. A slide bullet or heading usually ends in a capitalised word without
    terminal punctuation and is short. Keeping those as separate lines stops a
    whole slide being glued into one 200-word pseudo-sentence.
    """
    merged = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not merged:
            merged = line
            continue
        previous = merged.rsplit("\n", 1)[-1]
        if previous.endswith((",", ";", ":", "-", "\u2013", "\u2014", "(", "/")):
            merged = f"{merged} {line}"
        elif len(previous) >= 60 and not re.search(r"[.!?]$", previous):
            # Long line without a terminator: almost certainly a wrapped sentence.
            merged = f"{merged} {line}"
        elif line[:1].islower() and len(previous) >= 40:
            merged = f"{merged} {line}"
        else:
            merged = f"{merged}\n{line}"
    return re.sub(r"[ \t]+", " ", merged).strip()


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
        merged = _merge_block_lines(para.split("\n"))
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
    """Sentence segmentation that tolerates abbreviations, decimals and formulas.

    Lines that survived ``normalize_text`` as separate units (slide bullets,
    headings) are tokenised independently, so two bullets are never merged into
    one sentence just because neither ends with a full stop.
    """
    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    for block in (part.strip() for part in text.split("\n")):
        if not block:
            continue
        if ensure_nltk():
            try:
                from nltk.tokenize import sent_tokenize

                pieces.extend(sent_tokenize(block, language="english"))
                continue
            except Exception:
                pass
        pieces.extend(_regex_split(block))

    # NLTK sometimes splits after "e.g." / "Fig." - repair obvious cases.
    repaired: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if repaired:
            tail = re.search(r"([A-Za-z]{1,5}(?:\.[A-Za-z]{1,5})*)\.$", repaired[-1])
            prev_ends_abbrev = tail is not None
            if prev_ends_abbrev and tail.group(1).lower() in _ABBREVIATIONS:
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


def is_code_like(text: str) -> bool:
    """Detect code, grammar rules and parse trees extracted from slide decks.

    These are real content in a lecture PDF but they make terrible summary
    sentences (``(S (NP (DT The) (NN cat)) ...)``), so they are excluded from
    candidate sentences while staying in the stored extracted text.
    """
    if _CODE_HINT.search(text):
        return True
    if len(_POS_TAG.findall(text)) >= 3:
        return True
    if len(_PARSE_TREE.findall(text)) >= 2:
        return True
    if len(_GRAMMAR_RULE.findall(text)) >= 3:
        return True
    if len(text) >= 30:
        symbols = len(_CODE_SYMBOLS.findall(text))
        if symbols / len(text) > 0.12:
            return True
    return False


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
    # This is an English-language summariser; foreign-script example lines from
    # translation slides are not useful summary content.
    if len(_NON_LATIN.findall(stripped)) >= 3:
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
        if is_code_like(sentence):
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
