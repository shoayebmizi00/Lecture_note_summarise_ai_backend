"""Abstractive summarisation with BART (facebook/bart-large-cnn) + distilbart fallback.

The model is loaded once and cached for the process lifetime. Long documents are
chunked with the model's own tokenizer and summarised hierarchically
(map -> reduce), never by pushing a whole PDF into BART at once.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.core.config import settings
from app.core.logging import get_logger
from app.nlp.chunking import ChunkSet, build_chunks

logger = get_logger(__name__)

ProgressFn = Callable[[int, str, str], None]

GENERATION_PRESETS: dict[str, dict[str, int | float]] = {
    "short": {"max_length": 150, "min_length": 45, "length_penalty": 2.0},
    "medium": {"max_length": 260, "min_length": 80, "length_penalty": 2.0},
    "long": {"max_length": 420, "min_length": 120, "length_penalty": 1.8},
}


class ModelUnavailableError(RuntimeError):
    """Raised when the transformer model cannot be loaded for this request."""


@dataclass
class ModelInfo:
    model_name: str
    requested_model: str
    fallback_used: bool
    device: str
    dtype: str
    max_input_tokens: int
    load_seconds: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "requested_model": self.requested_model,
            "fallback_used": self.fallback_used,
            "device": self.device,
            "dtype": self.dtype,
            "max_input_tokens": self.max_input_tokens,
            "load_seconds": round(self.load_seconds, 2),
            "warnings": self.warnings,
        }


@dataclass
class AbstractiveResult:
    summary_text: str
    method: str
    length: str
    model_name: str
    requested_model: str
    fallback_used: bool
    device: str
    chunk_count: int
    chunk_summaries: list[str]
    compression_ratio: float
    source_word_count: int
    summary_word_count: int
    processing_time: float
    chunks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "length": self.length,
            "summary_text": self.summary_text,
            "model_name": self.model_name,
            "requested_model": self.requested_model,
            "fallback_used": self.fallback_used,
            "device": self.device,
            "chunk_count": self.chunk_count,
            "chunk_summaries": self.chunk_summaries,
            "compression_ratio": round(self.compression_ratio, 4),
            "source_word_count": self.source_word_count,
            "summary_word_count": self.summary_word_count,
            "processing_time": round(self.processing_time, 3),
            "chunks": self.chunks,
            "warnings": self.warnings,
        }


_BAD_SENTENCE_END = re.compile(r"\b(the|a|an|and|or|of|to|in|for|with|that|which|is|are|was|were)$", re.I)


class AbstractiveSummarizer:
    """Process-wide singleton holding one tokenizer + one seq2seq model."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._info: ModelInfo | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # --- loading --------------------------------------------------------
    def ensure_loaded(self, progress: ProgressFn | None = None) -> ModelInfo:
        if self._info is not None and self._model is not None:
            return self._info
        with self._lock:
            if self._info is not None and self._model is not None:
                return self._info
            if self._load_error and not settings.allow_model_download:
                raise ModelUnavailableError(self._load_error)
            self._info = self._load()
            return self._info

    def _candidate_models(self) -> list[tuple[str, bool]]:
        requested = settings.bart_model
        if settings.prefer_light_model:
            primary = settings.fallback_model
            candidates = [(primary, primary != requested)]
            if requested != primary:
                candidates.append((requested, False))
            return candidates
        candidates = [(requested, False)]
        if settings.fallback_model and settings.fallback_model != requested:
            candidates.append((settings.fallback_model, True))
        return candidates

    def _load(self) -> ModelInfo:
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            self._load_error = (
                "PyTorch/Transformers are not installed. Abstractive summarisation is "
                "unavailable - use the TextRank method instead."
            )
            raise ModelUnavailableError(self._load_error) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        errors: list[str] = []
        cache_dir = str(settings.hf_home / "hub")

        for name, is_fallback in self._candidate_models():
            try:
                logger.info("Loading summarisation model %s (device=%s)", name, device)
                tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    cache_dir=cache_dir,
                    local_files_only=not settings.allow_model_download,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    name,
                    cache_dir=cache_dir,
                    local_files_only=not settings.allow_model_download,
                )
                model.eval()
                model.to(device)
                use_half = device == "cuda"
                if use_half:
                    model.half()
                if device == "cpu":
                    try:
                        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
                    except Exception:
                        pass

                max_input = min(settings.max_input_tokens, getattr(model.config, "max_position_embeddings", 1024))
                info = ModelInfo(
                    model_name=name,
                    requested_model=settings.bart_model,
                    fallback_used=is_fallback,
                    device=device,
                    dtype="float16" if use_half else "float32",
                    max_input_tokens=max_input,
                    load_seconds=time.perf_counter() - started,
                    warnings=errors,
                )
                self._model = model
                self._tokenizer = tokenizer
                self._load_error = None
                logger.info(
                    "Model %s ready in %.1fs (%s, %s)",
                    name, info.load_seconds, device, info.dtype,
                )
                return info
            except Exception as exc:
                err = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(err)
                logger.warning("Failed to load model %s: %s", name, exc)

        self._load_error = (
            "Could not load an abstractive summarisation model. "
            + " | ".join(errors)
            + " Check your internet connection (first run downloads weights) or run "
            "'python download_models.py' in the model folder. You can still use TextRank."
        )
        raise ModelUnavailableError(self._load_error)

    # --- introspection --------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        device = "unknown"
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        cached = (settings.hf_home / "hub" / f"models--{settings.bart_model.replace('/', '--')}").exists()
        fallback_cached = (
            settings.hf_home / "hub" / f"models--{settings.fallback_model.replace('/', '--')}"
        ).exists()
        return {
            "loaded": self.loaded,
            "loaded_model": self._info.model_name if self._info else None,
            "device": self._info.device if self._info else device,
            "primary_model": settings.bart_model,
            "primary_model_cached": cached,
            "fallback_model": settings.fallback_model,
            "fallback_model_cached": fallback_cached,
            "prefer_light_model": settings.prefer_light_model,
            "allow_model_download": settings.allow_model_download,
            "last_error": self._load_error,
        }

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return max(1, int(len(text.split()) * 1.35))
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, int(len(text.split()) * 1.35))

    # --- inference ------------------------------------------------------
    def _generate(self, texts: list[str], length: str) -> list[str]:
        import torch

        assert self._model is not None and self._tokenizer is not None
        preset = GENERATION_PRESETS.get(length, GENERATION_PRESETS["medium"])
        beams = 2 if self._info and self._info.device == "cpu" else 4
        batch_size = max(1, min(settings.summary_batch_size, len(texts)))
        outputs: list[str] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=settings.max_input_tokens - 8,
                )
                encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    generated = self._model.generate(
                        **encoded,
                        num_beams=beams,
                        max_length=int(preset["max_length"]),
                        min_length=int(preset["min_length"]),
                        length_penalty=float(preset["length_penalty"]),
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
                outputs.extend(self._clean(text) for text in decoded)
            except (RuntimeError, MemoryError) as exc:
                if len(batch) > 1:
                    logger.warning("Batch generation failed (%s); retrying one-by-one", exc)
                    outputs.extend(self._generate(batch, length))
                else:
                    raise
        return outputs

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"^[.,;:\-–—\s]+", "", text)
        if text and text[-1] not in ".!?\"'":
            text += "."
        return text

    @staticmethod
    def _is_usable(summary: str, source: str) -> bool:
        if len(summary.split()) < 4:
            return False
        if summary.strip().lower() == source.strip().lower():
            return False
        # Reject degenerate repetition ("the the the ...").
        words = summary.lower().split()
        if len(words) > 12 and len(set(words)) / len(words) < 0.35:
            return False
        return True

    def _summarise_chunks(self, chunk_texts: list[str], length: str,
                          progress: ProgressFn | None) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        summaries: list[str] = []
        total = len(chunk_texts)
        for i, chunk in enumerate(chunk_texts):
            if progress:
                progress(
                    10 + int(70 * i / max(1, total)),
                    "summarizing",
                    f"Summarising chunk {i + 1} of {total}",
                )
            try:
                out = self._generate([chunk], length)[0]
            except Exception as exc:
                logger.warning("Chunk %s failed: %s", i + 1, exc)
                warnings.append(f"Chunk {i + 1} could not be summarised ({type(exc).__name__}); "
                                "its lead sentences were used instead.")
                out = " ".join(chunk.split(". ")[:2]).strip()
                if out and not out.endswith("."):
                    out += "."
            if not self._is_usable(out, chunk):
                warnings.append(f"Model output for chunk {i + 1} was degenerate; "
                                "fell back to its lead sentences.")
                out = " ".join(re.split(r"(?<=[.!?])\s+", chunk)[:2]).strip()
            if out:
                summaries.append(out)
        return summaries, warnings

    def _reduce(self, summaries: list[str], length: str,
                progress: ProgressFn | None) -> tuple[str, list[str]]:
        """Hierarchically merge chunk summaries until they fit one model window."""
        warnings: list[str] = []
        combined = self._dedupe_join(summaries)
        token_count = self.count_tokens(combined)
        budget = settings.chunk_token_budget
        rounds = 0
        while token_count > budget and rounds < 2 and len(summaries) > 1:
            rounds += 1
            if progress:
                progress(85, "merging", f"Reducing {len(summaries)} partial summaries (round {rounds})")
            groups: list[str] = []
            current: list[str] = []
            current_tokens = 0
            for summary in summaries:
                tokens = self.count_tokens(summary)
                if current and current_tokens + tokens > budget:
                    groups.append(" ".join(current))
                    current, current_tokens = [], 0
                current.append(summary)
                current_tokens += tokens
            if current:
                groups.append(" ".join(current))
            merged: list[str] = []
            for group in groups:
                try:
                    merged.append(self._generate([group], length)[0])
                except Exception as exc:
                    warnings.append(f"Reduce step failed ({type(exc).__name__}); kept partial text.")
                    merged.append(group)
            if len(merged) >= len(summaries):
                break
            summaries = merged
            combined = self._dedupe_join(summaries)
            token_count = self.count_tokens(combined)
        return combined, warnings

    @staticmethod
    def _dedupe_join(summaries: list[str]) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for summary in summaries:
            for sentence in re.split(r"(?<=[.!?])\s+", summary.strip()):
                key = re.sub(r"[^a-z0-9]+", "", sentence.lower())[:180]
                if not key or key in seen:
                    continue
                seen.add(key)
                parts.append(sentence.strip())
        return " ".join(parts).strip()

    # --- public API -----------------------------------------------------
    def summarize(self, sentences: list[str], length: str = "medium",
                  progress: ProgressFn | None = None) -> AbstractiveResult:
        started = time.perf_counter()
        length = (length or "medium").lower()
        if length not in GENERATION_PRESETS:
            length = "medium"

        if progress:
            progress(3, "loading_model", "Loading summarisation model (first run may download weights)")
        info = self.ensure_loaded(progress)
        if progress:
            progress(10, "chunking", "Splitting document into model-sized chunks")

        chunk_set: ChunkSet = build_chunks(
            sentences,
            token_counter=self.count_tokens,
            max_tokens=min(settings.chunk_token_budget, info.max_input_tokens - 80),
            overlap_sentences=settings.chunk_overlap_sentences,
        )
        if not chunk_set.chunks:
            raise ModelUnavailableError("The document produced no usable chunks for summarisation.")

        chunk_texts = [c.text for c in chunk_set.chunks]
        warnings: list[str] = list(chunk_set.warnings)
        if len(chunk_texts) > settings.abstractive_max_chunks:
            warnings.append(
                f"The document needs {len(chunk_texts)} chunks but only the first "
                f"{settings.abstractive_max_chunks} were summarised to keep runtime reasonable. "
                "Increase ABSTRACTIVE_MAX_CHUNKS in backend/.env for full coverage."
            )
            chunk_texts = chunk_texts[: settings.abstractive_max_chunks]

        chunk_summaries, chunk_warnings = self._summarise_chunks(chunk_texts, length, progress)
        warnings.extend(chunk_warnings)
        if not chunk_summaries:
            raise ModelUnavailableError("The model did not return any usable summary text.")

        final_text, reduce_warnings = self._reduce(chunk_summaries, length, progress)
        warnings.extend(reduce_warnings)
        final_text = self._clean(final_text)

        if progress:
            progress(95, "finalizing", "Finalising summary")

        source_words = sum(len(s.split()) for s in sentences)
        summary_words = len(final_text.split())
        elapsed = time.perf_counter() - started

        return AbstractiveResult(
            summary_text=final_text,
            method="abstractive",
            length=length,
            model_name=info.model_name,
            requested_model=info.requested_model,
            fallback_used=info.fallback_used,
            device=info.device,
            chunk_count=len(chunk_texts),
            chunk_summaries=chunk_summaries,
            compression_ratio=(summary_words / source_words) if source_words else 0.0,
            source_word_count=source_words,
            summary_word_count=summary_words,
            processing_time=elapsed,
            chunks=[c.as_dict() for c in chunk_set.chunks[: len(chunk_texts)]],
            warnings=warnings,
        )


_summarizer: AbstractiveSummarizer | None = None
_summarizer_lock = threading.Lock()


def get_summarizer() -> AbstractiveSummarizer:
    global _summarizer
    if _summarizer is None:
        with _summarizer_lock:
            if _summarizer is None:
                _summarizer = AbstractiveSummarizer()
    return _summarizer
