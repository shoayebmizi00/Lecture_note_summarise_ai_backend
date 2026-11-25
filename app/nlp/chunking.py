"""Tokenizer-aware chunking with sentence boundaries and overlap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.core.config import settings


@dataclass
class Chunk:
    index: int
    text: str
    start_sentence: int
    end_sentence: int
    token_count: int
    sentence_count: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "token_count": self.token_count,
            "sentence_count": self.sentence_count,
            "preview": self.text[:180],
        }


@dataclass
class ChunkSet:
    chunks: list[Chunk]
    token_counter: str
    total_tokens: int
    empty_chunks_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    """Conservative fallback: ~1.35 tokens per whitespace word for English prose."""
    return max(1, int(len(text.split()) * 1.35))


def build_chunks(
    sentences: list[str],
    *,
    token_counter: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
    overlap_sentences: int | None = None,
) -> ChunkSet:
    """Pack whole sentences into chunks that fit the model's token window.

    `token_counter` should be the model tokenizer's encode length so the limit is
    real, not guessed. Overlap keeps context across chunk borders.
    """
    max_tokens = max_tokens or settings.chunk_token_budget
    overlap = settings.chunk_overlap_sentences if overlap_sentences is None else max(0, overlap_sentences)
    counter = token_counter or _estimate_tokens
    counter_name = "model-tokenizer" if token_counter else "estimate(word*1.35)"

    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        return ChunkSet(chunks=[], token_counter=counter_name, total_tokens=0,
                        warnings=["No sentences available for chunking."])

    # Hard budget: leave room for the generated tokens.
    budget = max(120, min(max_tokens, settings.max_input_tokens - 80))

    chunks: list[Chunk] = []
    empty_skipped = 0
    warnings: list[str] = []

    current: list[str] = []
    current_start = 0
    current_tokens = 0
    i = 0
    total_tokens = 0

    while i < len(sentences):
        sentence = sentences[i]
        sent_tokens = counter(sentence)

        if sent_tokens > budget:
            # A single monster sentence: split it on word boundaries.
            if current:
                chunks.append(_make_chunk(len(chunks), current, current_start, i - 1, counter))
                current, current_tokens = [], 0
            for piece in _split_long_sentence(sentence, budget, counter):
                if not piece.strip():
                    empty_skipped += 1
                    continue
                chunks.append(_make_chunk(len(chunks), [piece], i, i, counter))
            i += 1
            current_start = i
            continue

        if current and current_tokens + sent_tokens > budget:
            end = i - 1
            chunks.append(_make_chunk(len(chunks), current, current_start, end, counter))
            # Carry overlapping sentences into the next chunk.
            carry = current[-overlap:] if overlap else []
            current = list(carry)
            current_tokens = sum(counter(s) for s in current)
            current_start = max(0, i - len(carry))
            continue

        if not current:
            current_start = i
            current_tokens = 0
        current.append(sentence)
        current_tokens += sent_tokens
        i += 1

    if current:
        chunks.append(_make_chunk(len(chunks), current, current_start, len(sentences) - 1, counter))

    # Drop empty chunks (defensive) and reindex.
    usable = [c for c in chunks if c.text.strip()]
    empty_skipped += len(chunks) - len(usable)
    for idx, chunk in enumerate(usable):
        chunk.index = idx

    total_tokens = sum(c.token_count for c in usable)
    if len(usable) > 1:
        warnings.append(
            f"Document split into {len(usable)} chunks ({total_tokens:,} tokens) "
            f"to stay inside the model's context window."
        )
    return ChunkSet(
        chunks=usable,
        token_counter=counter_name,
        total_tokens=total_tokens,
        empty_chunks_skipped=empty_skipped,
        warnings=warnings,
    )


def _make_chunk(index: int, sentences: list[str], start: int, end: int,
                counter: Callable[[str], int]) -> Chunk:
    text = " ".join(s.strip() for s in sentences).strip()
    return Chunk(
        index=index,
        text=text,
        start_sentence=start,
        end_sentence=end,
        token_count=counter(text),
        sentence_count=len(sentences),
    )


def _split_long_sentence(sentence: str, budget: int, counter: Callable[[str], int]) -> list[str]:
    words = sentence.split()
    if not words:
        return []
    approx_tokens_per_word = max(1.0, counter(sentence) / len(words))
    words_per_piece = max(20, int(budget / approx_tokens_per_word) - 10)
    return [" ".join(words[i:i + words_per_piece]) for i in range(0, len(words), words_per_piece)]
