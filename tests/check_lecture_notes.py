"""Run the whole NLP stack over a folder of real lecture PDFs.

This is a manual robustness check (it depends on your own PDFs), not part of
pytest. It exercises extraction -> preprocessing -> extractive summary ->
keywords -> topics -> quiz and reports failures instead of stopping.

Usage (from backend/):
    .venv\\Scripts\\python.exe tests\\check_lecture_notes.py
    .venv\\Scripts\\python.exe tests\\check_lecture_notes.py --folder ..\\Lecture_notes
    .venv\\Scripts\\python.exe tests\\check_lecture_notes.py --sample 2 --show
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=ROOT / "Lecture_notes")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--show", type=int, default=0, help="print this many summaries")
    args = parser.parse_args()

    from app.nlp.extractive import summarize
    from app.nlp.keywords import extract_keywords
    from app.nlp.pdf_extractor import extract_pdf
    from app.nlp.preprocessing import preprocess
    from app.nlp.quiz_generator import generate_quiz
    from app.nlp.topic_modeling import generate_topics

    pdfs = sorted(args.folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.folder}")
        return 1

    failures = 0
    shown = 0
    print(f"{'file':50s} {'pg':>3s} {'words':>6s} {'sent':>5s} {'sum':>5s} {'kw':>4s} "
          f"{'top':>4s} {'quiz':>7s}  status")
    print("-" * 110)

    for path in pdfs:
        try:
            extraction = extract_pdf(path.read_bytes(), allow_ocr=False)
            doc = preprocess(extraction.text)
            summary = summarize(doc.sentences, args.length)
            keywords = extract_keywords(doc.sentences, doc.clean_text, top_n=12)
            topics = build_topics(doc.sentences, extraction.headings, max_topics=5)
            quiz = generate_quiz(doc.sentences, keywords, count=5)
            status = "ok"
            if doc.word_count < 100:
                status = "low text (scanned/slides?)"
            print(f"{path.name[:50]:50s} {extraction.page_count:3d} {doc.word_count:6d} "
                  f"{doc.sentence_count:5d} {len(summary.selected):5d} {len(keywords):4d} "
                  f"{len(topics.topics):4d} {quiz.generated:3d}/{quiz.requested:<3d}  {status}")
            if shown < args.show:
                shown += 1
                print("\n  SUMMARY:", summary.summary_text[:600].replace("\n", " "), "\n")
                print("  KEYWORDS:", [k.keyword for k in keywords[:8]])
                print("  TOPICS:", [(t.title, t.source) for t in topics.topics])
                if quiz.questions:
                    q = quiz.questions[0]
                    print("  QUIZ:", q.question[:120])
                    print("        options:", q.options)
                print()
        except Exception as exc:  # keep going so one bad file does not hide the rest
            failures += 1
            print(f"{path.name[:50]:50s} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)

    print("-" * 110)
    print(f"{len(pdfs) - failures}/{len(pdfs)} PDFs processed successfully")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
