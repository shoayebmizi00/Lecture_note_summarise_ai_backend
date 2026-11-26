"""Unit tests for the accuracy-critical NLP pieces (no model download needed)."""

from __future__ import annotations

from app.nlp.abstractive import AbstractiveSummarizer, GENERATION_PRESETS
from app.nlp.evaluation import evaluate
from app.nlp.extractive import LENGTH_PRESETS, summarize


def test_extractive_returns_source_sentences_in_document_order() -> None:
    sentences = [
        "Natural language processing analyses human language.",
        "Tokenisation splits text into tokens.",
        "TF-IDF weights terms by frequency and rarity.",
        "TextRank ranks sentences with a graph.",
        "BART generates abstractive summaries.",
        "ROUGE measures overlap with a reference summary.",
    ]
    result = summarize(sentences, "short")
    assert result.method == "extractive"
    assert result.selected
    indices = [sentence.index for sentence in result.selected]
    assert indices == sorted(indices), "selected sentences must keep document order"
    assert all(sentences[index] == sentence.text for index, sentence in zip(indices, result.selected))
    assert 0 < result.compression_ratio < 1


def test_mmr_selection_prefers_a_diverse_sentence() -> None:
    import numpy as np

    from app.nlp.extractive import _mmr_select

    scores = np.array([1.0, 0.9, 0.5])
    sim = np.array([
        [0.0, 0.9, 0.0],
        [0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    # Sentence 0 wins on relevance; sentence 1 is nearly identical to it, so the
    # diverse sentence 2 should be chosen second even though its score is lower.
    assert _mmr_select(scores, sim, target=2, lambda_=0.65) == [0, 2]


def test_extractive_selection_respects_target_and_order() -> None:
    sentences = [
        "TF-IDF weights terms by frequency and rarity in a document.",
        "TF-IDF weights terms by frequency and rarity in a corpus.",
        "TF-IDF weights terms by frequency and rarity for ranking.",
        "TextRank ranks sentences using a similarity graph and PageRank.",
        "BART is a denoising sequence-to-sequence model for summarisation.",
        "ROUGE measures overlap with a human reference summary.",
    ]
    result = summarize(sentences, "medium", min_sentences=3, max_sentences=3)
    indices = [sentence.index for sentence in result.selected]
    assert result.target_sentences == 3
    assert len(indices) == 3
    assert indices == sorted(indices)


def test_extractive_short_document_keeps_everything() -> None:
    sentences = ["One sentence about NLP.", "Another sentence about BART."]
    result = summarize(sentences, "short")
    assert result.summary_word_count > 0
    assert any("short" in warning.lower() for warning in result.warnings)


def test_generation_presets_are_sane() -> None:
    for name, preset in GENERATION_PRESETS.items():
        assert 0 < float(preset["ratio"]) < 1, name
        assert int(preset["min_length"]) < int(preset["max_length"]), name
        assert float(preset["length_penalty"]) > 0, name
    assert set(GENERATION_PRESETS) == set(LENGTH_PRESETS)


def test_numeric_hallucination_guard_drops_unsupported_sentences() -> None:
    source = "BERT is a bidirectional encoder. GPT is an autoregressive decoder."
    generated = "BERT uses 2048 attention heads. GPT is an autoregressive decoder."
    cleaned, warnings = AbstractiveSummarizer._drop_unsupported_numbers(generated, source)
    assert "2048" not in cleaned
    assert "GPT is an autoregressive decoder." in cleaned
    assert warnings and "numbers" in warnings[0].lower()


def test_numeric_guard_keeps_numbers_that_exist_in_the_source() -> None:
    source = "BART has a 1024 token input window."
    generated = "BART accepts 1024 tokens at a time."
    cleaned, warnings = AbstractiveSummarizer._drop_unsupported_numbers(generated, source)
    assert "1024" in cleaned
    assert warnings == []


def test_rouge_is_one_for_an_identical_summary() -> None:
    text = "TextRank ranks sentences with a similarity graph and PageRank."
    result = evaluate(text, text, text, 0.0, "extractive")
    assert result.rouge1.fmeasure == 1.0
    assert result.rouge2.fmeasure == 1.0
    assert result.rougeL.fmeasure == 1.0
    assert result.stats.compression_ratio == 1.0


def test_rouge_rejects_a_too_short_reference() -> None:
    from app.nlp.evaluation import EvaluationError
    import pytest

    with pytest.raises(EvaluationError):
        evaluate("A valid generated summary.", "short", None, None, None)
