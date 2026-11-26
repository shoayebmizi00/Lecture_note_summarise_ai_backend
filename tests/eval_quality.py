"""Measure summarisation quality on the bundled sample lecture.

This is a manual evaluation harness, not part of pytest, because the abstractive
run loads BART and can take minutes on a CPU.

Usage (from backend/):
    .venv\\Scripts\\python.exe tests\\eval_quality.py
    .venv\\Scripts\\python.exe tests\\eval_quality.py --abstractive
    .venv\\Scripts\\python.exe tests\\eval_quality.py --abstractive --length short

It reports ROUGE-1/2/L against a hand-written reference summary for:
  * lead-3 (a naive baseline)
  * TextRank without MMR (ablation)
  * TextRank with MMR (the default)
  * BART abstractive (optional, --abstractive)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
PDF = ROOT / "dataset" / "sample_lecture_notes.pdf"
OUT = ROOT / "docs" / "quality_report.json"

REFERENCE = (
    "Natural Language Processing combines computer science and linguistics to analyse and generate "
    "human language. Text is prepared with tokenisation and lemmatisation, and represented with "
    "TF-IDF weights and cosine similarity. Word2Vec, GloVe and FastText learn dense embeddings, while "
    "BERT and GPT are Transformer models pre-trained on large corpora. TextRank ranks sentences in a "
    "graph for extractive summarisation, whereas BART generates abstractive summaries from overlapping "
    "chunks. ROUGE-1, ROUGE-2 and ROUGE-L measure overlap with a human reference and do not prove "
    "factual correctness."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--abstractive", action="store_true", help="also run BART (slow on CPU)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from app.nlp.evaluation import evaluate
    from app.nlp.extractive import summarize
    from app.nlp.pdf_extractor import extract_pdf
    from app.nlp.preprocessing import preprocess

    extraction = extract_pdf(args.pdf.read_bytes(), allow_ocr=False)
    doc = preprocess(extraction.text)
    source = extraction.text
    print(f"Document: {args.pdf.name} · {extraction.page_count} pages · "
          f"{doc.sentence_count} sentences · {doc.word_count} words")
    print(f"Reference: {len(REFERENCE.split())} words · summary length preset: {args.length}\n")

    runs: list[dict] = []

    def score(name: str, text: str, elapsed: float, method: str | None) -> None:
        result = evaluate(text, REFERENCE, source, elapsed, method)
        row = {
            "method": name,
            "words": result.stats.summary_words,
            "compression": result.stats.compression_ratio,
            "seconds": round(result.processing_time, 3),
            "rouge1": result.rouge1.fmeasure,
            "rouge2": result.rouge2.fmeasure,
            "rougeL": result.rougeL.fmeasure,
            "summary": text,
        }
        runs.append(row)
        print(f"{name:24s} R1={row['rouge1']:.4f}  R2={row['rouge2']:.4f}  "
              f"RL={row['rougeL']:.4f}  words={row['words']:3d}  "
              f"compression={row['compression']:.3f}  time={row['seconds']:.2f}s")

    # Naive baseline: the first three sentences.
    lead = " ".join(doc.sentences[:3])
    score("lead-3 baseline", lead, 0.0, None)

    # Ablation: the previous pure top-k selection.
    started = time.perf_counter()
    no_mmr = summarize(doc.sentences, args.length, use_mmr=False)
    score("TextRank (no MMR)", no_mmr.summary_text, time.perf_counter() - started, "extractive")

    # Default: MMR-selected TextRank.
    started = time.perf_counter()
    mmr = summarize(doc.sentences, args.length, use_mmr=True)
    score("TextRank + MMR", mmr.summary_text, time.perf_counter() - started, "extractive")

    if args.abstractive:
        from app.nlp.abstractive import get_summarizer

        started = time.perf_counter()
        bart = get_summarizer().summarize(doc.sentences, args.length)
        score(f"BART ({bart.model_name.split('/')[-1]})", bart.summary_text,
              time.perf_counter() - started, "abstractive")
        if bart.warnings:
            print("\nBART warnings:")
            for warning in bart.warnings:
                print(f"  - {warning}")

    print()
    best = max(runs, key=lambda row: row["rougeL"])
    print(f"Best ROUGE-L: {best['method']} ({best['rougeL']:.4f})")
    print("Reminder: ROUGE measures overlap with one reference; inspect summaries for faithfulness.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document": str(args.pdf),
        "length": args.length,
        "reference": REFERENCE,
        "runs": runs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
