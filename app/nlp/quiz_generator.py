"""Extractive, template-based MCQ generation grounded in the uploaded document.

Every question is derived from a real sentence of the lecture. A candidate is
discarded when four unambiguous options cannot be built from the document, so
the generator skips rather than inventing unrelated questions.

Strategies
----------
1. definition  - "X is/are/refers to/means ..." -> "What is X?"
2. acronym     - "Full Name (FNL)" -> "What does FNL stand for?"
3. numeric     - a factual sentence with a single number -> fill in the value
4. cloze       - blank a salient document keyword in a real sentence

No neural question-generation model is used; this is a documented, deterministic
extractive approach (as allowed by the project brief).
"""

from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.nlp.keywords import Keyword

logger = get_logger(__name__)

_DEF_RE = re.compile(
    r"^(?P<term>[A-Z][A-Za-z0-9][\w\-,/&'’ ]{1,55}?)\s+"
    r"(?P<verb>is|are|refers to|refer to|means|is defined as|are defined as|"
    r"is known as|are known as|is called|are called|denotes|describes)\s+"
    r"(?P<definition>.{25,300})$"
)
_ACRONYM_FIRST = re.compile(
    r"\b(?P<expansion>[A-Z][A-Za-z]{2,}(?:\s+[A-Za-z][A-Za-z\-]*){0,7})\s*"
    r"\((?P<acronym>[A-Z]{2,8})\)"
)
_ACRONYM_SECOND = re.compile(
    r"\b(?P<acronym>[A-Z]{2,8})\s*\(\s*(?P<expansion>[A-Z][A-Za-z]{2,}"
    r"(?:\s+[A-Za-z][A-Za-z\-]*){0,7})\s*\)"
)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?%?)(?![\w.])")

_PRONOUN_TERMS = {
    "this", "that", "these", "those", "there", "it", "they", "he", "she", "we",
    "which", "who", "what", "when", "where", "how", "his", "her", "their",
    "the", "a", "an", "in", "on", "for", "with", "as", "if", "while", "also",
    "however", "therefore", "then", "there", "here",
}
_GENERIC_TERMS = {"figure", "table", "page", "chapter", "section", "example", "note", "notes"}

# Tokens that are poor cloze answers even when they are frequent document words.
_WEAK_CLOZE_WORDS = {
    "long", "short", "large", "small", "many", "much", "used", "using", "use",
    "correct", "incorrect", "important", "different", "similar", "common",
    "possible", "necessary", "called", "known", "given", "shown", "based",
    "also", "often", "usually", "generally", "first", "second", "third",
    "model", "models", "data", "system", "systems", "method", "methods",
    "approach", "approaches", "process", "processes", "result", "results",
    "example", "part", "type", "types", "value", "values", "case", "cases",
    "time", "number", "way", "thing", "things", "problem", "idea",
}

_pos_ready: bool | None = None


def _ensure_pos() -> bool:
    """NLTK POS tagger data (used only to keep cloze answers noun-like)."""
    global _pos_ready
    if _pos_ready is not None:
        return _pos_ready
    try:
        import nltk
        from nltk.data import find

        for resource, pkg in (
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ):
            try:
                find(resource)
            except LookupError:
                nltk.download(pkg, quiet=True)
        _pos_ready = True
    except Exception:
        _pos_ready = False
    return _pos_ready


def _noun_tokens(sentence: str) -> set[str] | None:
    """Lowercased noun-ish tokens of a sentence, or None if POS is unavailable."""
    if not _ensure_pos():
        return None
    try:
        from nltk import pos_tag, word_tokenize

        return {
            word.lower()
            for word, tag in pos_tag(word_tokenize(sentence))
            if tag.startswith("NN")
        }
    except Exception:
        return None


@dataclass
class QuizItem:
    question: str
    options: list[str]
    correct_index: int
    correct_answer: str
    explanation: str
    source_sentence: str
    strategy: str
    difficulty: str = "medium"

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "options": self.options,
            "correct_index": self.correct_index,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "source_sentence": self.source_sentence,
            "strategy": self.strategy,
            "difficulty": self.difficulty,
        }


@dataclass
class QuizResult:
    items: list[QuizItem]
    requested: int
    generated: int
    skipped: int
    strategies: dict[str, int]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "questions": [q.as_dict() for q in self.items],
            "requested": self.requested,
            "generated": self.generated,
            "skipped": self.skipped,
            "strategies": self.strategies,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def _clean(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\-\u2022\*\d.)\s]+", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if len(text) > limit:
        cut = text[:limit]
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        text = cut.rstrip(",;: ") + "…"
    return text


def _first_sentence(text: str, limit: int = 180) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return _clean(parts[0] if parts else text, limit)


def _is_duplicate_option(a: str, b: str) -> bool:
    a_norm, b_norm = a.lower().strip(), b.lower().strip()
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio() > 0.82


def _build_options(correct: str, distractors: list[str], rng: random.Random) -> tuple[list[str], int] | None:
    options = [correct]
    for candidate in distractors:
        if len(options) >= 4:
            break
        if not candidate or len(candidate) < 2:
            continue
        if any(_is_duplicate_option(candidate, existing) for existing in options):
            continue
        options.append(candidate)
    if len(options) < 4:
        return None
    rng.shuffle(options)
    return options, options.index(correct)


def _valid_term(term: str) -> bool:
    term = term.strip()
    if not (2 < len(term) <= 60):
        return False
    first = re.split(r"[\s\-]", term)[0].lower()
    if first in _PRONOUN_TERMS or first in _GENERIC_TERMS:
        return False
    if term.lower() in _PRONOUN_TERMS or term.lower() in _GENERIC_TERMS:
        return False
    if len(term.split()) > 6:
        return False
    if not re.search(r"[A-Za-z]{2,}", term):
        return False
    return True


# ---------------------------------------------------------------------------
# strategy implementations
# ---------------------------------------------------------------------------
def _definition_candidates(sentences: list[str]) -> list[dict]:
    definitions: list[tuple[int, str, str, str]] = []
    for idx, sentence in enumerate(sentences):
        match = _DEF_RE.match(sentence.strip())
        if not match:
            continue
        term = match.group("term").strip(" ,;:")
        verb = match.group("verb")
        definition = _first_sentence(match.group("definition"))
        if not _valid_term(term) or len(definition.split()) < 5:
            continue
        if definition.lower().startswith(("not ", "n't")):
            continue
        definitions.append((idx, term, verb, definition))

    candidates: list[dict] = []
    for idx, term, verb, definition in definitions:
        distractors = [
            other_def for other_idx, _, _, other_def in definitions
            if other_idx != idx
        ]
        if len(distractors) < 3:
            continue
        question = f"What {'are' if verb.startswith('are') else 'is'} {term}?"
        candidates.append(
            {
                "question": question,
                "correct": definition,
                "distractors": distractors,
                "explanation": f"The lecture notes define {term} as: “{_clean(sentences[idx], 240)}”",
                "source": sentences[idx],
                "sentence_index": idx,
                "strategy": "definition",
                "difficulty": "easy",
            }
        )
    return candidates


def _acronym_candidates(sentences: list[str]) -> list[dict]:
    pairs: list[tuple[int, str, str]] = []
    for idx, sentence in enumerate(sentences):
        for pattern, first_is_expansion in ((_ACRONYM_FIRST, True), (_ACRONYM_SECOND, False)):
            for match in pattern.finditer(sentence):
                expansion = match.group("expansion").strip()
                acronym = match.group("acronym").strip()
                if not _valid_term(expansion) or len(expansion.split()) < 2:
                    continue
                pairs.append((idx, acronym, expansion))
    # De-duplicate by acronym.
    unique: dict[str, tuple[int, str, str]] = {}
    for idx, acronym, expansion in pairs:
        unique.setdefault(acronym, (idx, acronym, expansion))

    candidates: list[dict] = []
    for acronym, (idx, _, expansion) in unique.items():
        distractors = [exp for acr, (_, _, exp) in unique.items() if acr != acronym]
        if len(distractors) < 3:
            continue
        candidates.append(
            {
                "question": f"In these lecture notes, what does “{acronym}” stand for?",
                "correct": _clean(expansion, 120),
                "distractors": [_clean(d, 120) for d in distractors],
                "explanation": f"The notes state: “{_clean(sentences[idx], 240)}”",
                "source": sentences[idx],
                "sentence_index": idx,
                "strategy": "acronym",
                "difficulty": "medium",
            }
        )
    return candidates


def _numeric_pool(sentences: list[str]) -> dict[str, int]:
    pool: dict[str, int] = {}
    for sentence in sentences:
        for value in _NUMBER.findall(sentence):
            pool[value] = pool.get(value, 0) + 1
    return pool


def _numeric_candidates(sentences: list[str], pool: dict[str, int]) -> list[dict]:
    candidates: list[dict] = []
    for idx, sentence in enumerate(sentences):
        # Titles such as "Lecture 5 - ..." are not factual statements.
        if re.match(r"^(lecture|chapter|section|unit|week|module|lesson|topic|part)\b",
                    sentence.strip(), re.IGNORECASE):
            continue
        numbers = list(_NUMBER.finditer(sentence))
        if len(numbers) != 1:
            continue
        if len(sentence.split()) < 9:
            continue
        match = numbers[0]
        # "5 - Text Representation" is a heading fragment, not a value in a fact.
        if match.start() <= 2 and re.match(r"\s*[-\u2013\u2014]", sentence[match.end():]):
            continue
        value = match.group(1)
        distractors = [v for v in pool if v != value][:12]
        if len(distractors) < 3:
            continue
        blanked = sentence[: match.start()] + "______" + sentence[match.end():]
        if len(blanked.split()) < 8:
            continue
        candidates.append(
            {
                "question": f"According to the notes, complete the statement: “{_clean(blanked, 240)}”",
                "correct": value,
                "distractors": distractors,
                "explanation": f"The original sentence is: “{_clean(sentence, 240)}”",
                "source": sentence,
                "sentence_index": idx,
                "strategy": "numeric",
                "difficulty": "hard",
            }
        )
    return candidates


def _term_pattern(term: str) -> str:
    """Word-boundary pattern that will not match inside a hyphenated compound."""
    return r"(?<![A-Za-z0-9\-'])" + re.escape(term) + r"(?![A-Za-z0-9\-'])"


def _cloze_candidates(sentences: list[str], keywords: list[Keyword]) -> list[dict]:
    candidates: list[dict] = []
    terms = [k.keyword for k in keywords if k.keyword and not k.keyword.isdigit()]
    if len(terms) < 4:
        return candidates

    for idx, sentence in enumerate(sentences):
        if len(sentence.split()) < 10:
            continue
        nouns = _noun_tokens(sentence)
        used_here: set[str] = set()
        for term in terms:
            if term.lower() in used_here:
                continue
            if term.lower() in _WEAK_CLOZE_WORDS:
                continue
            pattern = _term_pattern(term)
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not match:
                continue
            present = match.group(0)
            # Skip sentence-initial blanks and blanks inside a capitalised name.
            if match.start() == 0:
                continue
            before = sentence[: match.start()].rstrip()
            if re.search(r"\b[A-Z][a-z]+$", before):
                continue
            # Do not split a name/compound: "Long| Short-Term Memory", "hyper|parameter".
            tail = sentence[match.end():]
            if re.match(r"[A-Z\-]", tail[:1] or ""):
                continue
            if nouns is not None:
                term_tokens = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", term)]
                if not term_tokens or not all(
                    token in nouns or token.endswith(("tion", "ment", "ity", "ness", "ism"))
                    for token in term_tokens
                ):
                    continue
            # Do not blank a term that also appears elsewhere in the same sentence.
            if len(re.findall(pattern, sentence, flags=re.IGNORECASE)) > 1:
                continue
            blanked = sentence[: match.start()] + "______" + sentence[match.end():]
            if len(blanked.split()) < 8:
                continue
            distractors = [
                t for t in terms
                if t.lower() != term.lower()
                and t.lower() not in present.lower()
                and not re.search(_term_pattern(t), sentence, flags=re.IGNORECASE)
            ]
            if len(distractors) < 3:
                continue
            options_correct = present if present[:1].isupper() or present.isupper() else present.lower()
            candidates.append(
                {
                    "question": f"Fill in the blank: “{_clean(blanked, 250)}”",
                    "correct": options_correct,
                    "distractors": distractors,
                    "explanation": f"The original sentence is: “{_clean(sentence, 240)}”",
                    "source": sentence,
                    "sentence_index": idx,
                    "strategy": "cloze",
                    "difficulty": "medium",
                }
            )
            used_here.add(term.lower())
            break  # one cloze per sentence
    return candidates


def generate_quiz(sentences: list[str], keywords: list[Keyword] | None = None,
                  count: int = 5, seed: int = 42) -> QuizResult:
    count = max(1, min(int(count), 20))
    sentences = [s.strip() for s in sentences if s and s.strip()]
    notes = [
        "Questions are produced by a template-based extractive method that uses only "
        "sentences from the uploaded document. No external question-generation model is used.",
    ]
    warnings: list[str] = []
    if len(sentences) < 4:
        return QuizResult(items=[], requested=count, generated=0, skipped=count, strategies={},
                          notes=notes, warnings=["Not enough text to generate a quiz."])

    numeric_pool = _numeric_pool(sentences)
    pools = {
        "definition": _definition_candidates(sentences),
        "acronym": _acronym_candidates(sentences),
        "numeric": _numeric_candidates(sentences, numeric_pool),
        "cloze": _cloze_candidates(sentences, keywords or []),
    }

    rng = random.Random(seed)
    strategy_order = ["definition", "acronym", "cloze", "numeric"]
    # Shuffle within each pool so a re-run with a different seed gives different items.
    for name in strategy_order:
        rng.shuffle(pools[name])

    selected: list[dict] = []
    used_questions: set[str] = set()
    used_sources: dict[int, int] = {}
    strategy_counts: dict[str, int] = {name: 0 for name in strategy_order}

    # Round-robin so the quiz mixes question types.
    index = 0
    while len(selected) < count and any(pools[name] for name in strategy_order):
        name = strategy_order[index % len(strategy_order)]
        index += 1
        if not pools[name]:
            continue
        candidate = pools[name].pop(0)
        question_key = re.sub(r"\W+", "", candidate["question"].lower())[:160]
        if question_key in used_questions:
            continue
        if used_sources.get(candidate["sentence_index"], 0) >= 1:
            continue
        built = _build_options(candidate["correct"], candidate["distractors"], rng)
        if built is None:
            continue
        options, correct_index = built
        used_questions.add(question_key)
        used_sources[candidate["sentence_index"]] = used_sources.get(candidate["sentence_index"], 0) + 1
        strategy_counts[name] += 1
        selected.append(
            QuizItem(
                question=candidate["question"],
                options=options,
                correct_index=correct_index,
                correct_answer=candidate["correct"],
                explanation=candidate["explanation"],
                source_sentence=_clean(candidate["source"], 260),
                strategy=candidate["strategy"],
                difficulty=candidate["difficulty"],
            )
        )

    selected.sort(key=lambda item: sentences.index(item.source_sentence)
                  if item.source_sentence in sentences else 999)

    skipped = max(0, count - len(selected))
    if skipped:
        warnings.append(
            f"{skipped} question(s) could not be generated with four unambiguous options "
            "and were skipped instead of producing unreliable items."
        )
    if not selected:
        warnings.append("No reliable multiple-choice question could be built from this document.")

    return QuizResult(
        items=selected,
        requested=count,
        generated=len(selected),
        skipped=skipped,
        strategies={k: v for k, v in strategy_counts.items() if v},
        notes=notes,
        warnings=warnings,
    )
