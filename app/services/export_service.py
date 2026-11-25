"""Export summaries, keywords and quizzes as PDF (ReportLab), TXT and CSV."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#2563eb")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
LIGHT = colors.HexColor("#f1f5f9")


def _safe(value: Any) -> str:
    """Standard PDF fonts are WinAnsi: drop characters they cannot render."""
    text = str(value or "")
    try:
        return text.encode("cp1252", "replace").decode("cp1252")
    except Exception:
        return text.encode("ascii", "replace").decode("ascii")


def _esc(value: Any) -> str:
    return escape(_safe(value)).replace("\n", "<br/>")


def _fmt_duration(seconds: Any) -> str:
    try:
        seconds = float(seconds or 0)
    except (TypeError, ValueError):
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds // 60:.0f} m {seconds % 60:.0f} s"


def _date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    return str(value or datetime.now(timezone.utc).strftime("%d %b %Y"))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=2),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                   fontSize=9.5, textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=ACCENT,
                             spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=10, leading=15.5, textColor=colors.HexColor("#1e293b"),
                               spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["Normal"], fontName="Helvetica",
                                fontSize=8.5, leading=11, textColor=MUTED),
        "topic": ParagraphStyle("Topic", parent=base["Heading3"], fontName="Helvetica-Bold",
                                fontSize=10.5, leading=14, textColor=INK, spaceBefore=8,
                                spaceAfter=3),
    }


def _meta_table(rows: list[tuple[str, str]], styles) -> Table:
    data = [[Paragraph(f"<b>{_esc(k)}</b>", styles["small"]),
             Paragraph(_esc(v), styles["small"])] for k, v in rows]
    table = Table(data, colWidths=[38 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
    ]))
    return table


def build_summary_pdf(summary: dict[str, Any], lecture: dict[str, Any],
                      keywords: list[dict[str, Any]],
                      topics: dict[str, Any] | None = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"Summary - {lecture.get('original_filename', 'Lecture')}",
        author="LectureNote AI",
    )
    styles = _styles()
    story: list[Any] = []

    filename = lecture.get("original_filename", "lecture.pdf")
    method = "TextRank (extractive)" if summary.get("method") == "extractive" else "BART (abstractive)"
    if summary.get("model_name"):
        method = f"BART (abstractive) - {summary['model_name']}"

    story.append(Paragraph("Lecture Summary", styles["title"]))
    story.append(Paragraph(_esc(filename), styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=10))

    story.append(_meta_table([
        ("Lecture file", filename),
        ("Method", method),
        ("Summary length", str(summary.get("summary_length", "medium")).title()),
        ("Generated", _date(summary.get("created_at"))),
        ("Pages", str(lecture.get("page_count", "-"))),
        ("Source words", f"{summary.get('source_word_count') or lecture.get('word_count') or '-'}"),
        ("Summary words", str(summary.get("summary_word_count") or len(str(summary.get("summary_text", "")).split()))),
        ("Compression", f"{(summary.get('compression_ratio') or 0) * 100:.1f}%"),
        ("Processing time", _fmt_duration(summary.get("processing_time"))),
    ], styles))

    story.append(Paragraph("Summary", styles["h2"]))
    for paragraph in str(summary.get("summary_text", "")).split("\n\n"):
        if paragraph.strip():
            story.append(Paragraph(_esc(paragraph.strip()), styles["body"]))

    if keywords:
        story.append(Paragraph("Important keywords", styles["h2"]))
        chips = " &nbsp;·&nbsp; ".join(
            f"<b>{_esc(k.get('keyword'))}</b>" for k in keywords[:24] if k.get("keyword")
        )
        story.append(Paragraph(chips, styles["body"]))

    if topics and topics.get("topics"):
        story.append(Paragraph("Topic-wise summaries", styles["h2"]))
        for topic in topics["topics"][:12]:
            story.append(Paragraph(_esc(topic.get("title", "Topic")), styles["topic"]))
            story.append(Paragraph(_esc(topic.get("summary", "")), styles["body"]))
        story.append(Paragraph(
            "Titles are detected headings or labels derived from each group's most distinctive "
            "terms; they are not authored section headings.", styles["small"]))

    evaluation = summary.get("last_evaluation")
    if evaluation:
        story.append(Paragraph("ROUGE evaluation", styles["h2"]))
        rows = [["Metric", "Precision", "Recall", "F-Measure"]]
        for key, label in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")):
            metric = evaluation.get(key, {})
            rows.append([label, f"{metric.get('precision', 0):.3f}",
                         f"{metric.get('recall', 0):.3f}", f"{metric.get('fmeasure', 0):.3f}"])
        table = Table(rows, colWidths=[45 * mm, 35 * mm, 35 * mm, 35 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Paragraph(
            "ROUGE measures overlap with a human reference summary; it does not prove "
            "factual correctness.", styles["small"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        f"Generated by LectureNote AI on {_date(datetime.now(timezone.utc))}. "
        "Summaries are machine generated - verify important details against the original notes.",
        styles["small"]))

    def _footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 10 * mm, "LectureNote AI")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def build_summary_txt(summary: dict[str, Any], lecture: dict[str, Any],
                      keywords: list[dict[str, Any]],
                      topics: dict[str, Any] | None = None) -> str:
    lines: list[str] = []
    title = lecture.get("original_filename", "lecture.pdf")
    lines += [
        "=" * 74,
        "LECTURENOTE AI - LECTURE SUMMARY",
        "=" * 74,
        f"Lecture file   : {title}",
        f"Method         : {summary.get('method')}",
        f"Model          : {summary.get('model_name') or 'TextRank (TF-IDF + PageRank)'}",
        f"Summary length : {summary.get('summary_length')}",
        f"Generated      : {_date(summary.get('created_at'))}",
        f"Pages          : {lecture.get('page_count', '-')}",
        f"Source words   : {summary.get('source_word_count') or '-'}",
        f"Summary words  : {summary.get('summary_word_count') or len(str(summary.get('summary_text', '')).split())}",
        f"Processing time: {_fmt_duration(summary.get('processing_time'))}",
        "",
        "-" * 74,
        "SUMMARY",
        "-" * 74,
        str(summary.get("summary_text", "")).strip(),
        "",
    ]
    if keywords:
        lines += ["-" * 74, "IMPORTANT KEYWORDS", "-" * 74]
        lines += [f"- {k.get('keyword')} (score {k.get('score', 0):.2f}, x{k.get('frequency', 0)})"
                  for k in keywords]
        lines.append("")
    if topics and topics.get("topics"):
        lines += ["-" * 74, "TOPIC-WISE SUMMARIES", "-" * 74]
        for topic in topics["topics"]:
            lines.append(f"\n[{topic.get('title')}]")
            lines.append(topic.get("summary", ""))
        lines.append("")
    if summary.get("warnings"):
        lines += ["-" * 74, "NOTES", "-" * 74]
        lines += [f"- {w}" for w in summary["warnings"]]
        lines.append("")
    lines.append("Generated by LectureNote AI. Verify important details against the original notes.")
    return "\n".join(lines)


def build_keywords_csv(keywords: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "keyword", "score", "frequency", "kind"])
    for index, item in enumerate(keywords, start=1):
        writer.writerow([index, item.get("keyword", ""), item.get("score", ""),
                         item.get("frequency", ""), item.get("kind", "")])
    return "\ufeff" + buffer.getvalue()


def build_quiz_csv(questions: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["#", "question", "option_a", "option_b", "option_c", "option_d",
                     "correct_answer", "explanation", "strategy"])
    for index, item in enumerate(questions, start=1):
        options = list(item.get("options", [])) + ["", "", "", ""]
        writer.writerow([
            index, item.get("question", ""), options[0], options[1], options[2], options[3],
            item.get("correct_answer", ""), item.get("explanation", ""), item.get("strategy", ""),
        ])
    return "\ufeff" + buffer.getvalue()


def build_quiz_txt(questions: list[dict[str, Any]], title: str = "Lecture") -> str:
    lines = [
        "=" * 74,
        f"QUIZ - {title}",
        "=" * 74,
        "Generated by LectureNote AI from the uploaded lecture notes.",
        "",
    ]
    letters = ["A", "B", "C", "D", "E", "F"]
    for index, item in enumerate(questions, start=1):
        lines.append(f"{index}. {item.get('question', '')}")
        for opt_index, option in enumerate(item.get("options", [])):
            lines.append(f"   {letters[opt_index]}) {option}")
        lines.append("")
    lines += ["-" * 74, "ANSWER KEY", "-" * 74]
    for index, item in enumerate(questions, start=1):
        correct = item.get("correct_index", 0)
        letter = letters[correct] if 0 <= correct < len(letters) else "?"
        lines.append(f"{index}. {letter}) {item.get('correct_answer', '')}")
        explanation = item.get("explanation")
        if explanation:
            lines.append(f"   {explanation}")
    return "\n".join(lines)
