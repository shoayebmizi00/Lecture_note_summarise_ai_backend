"""Pytest fixtures for the LectureNote AI backend.

The suite talks to a real MongoDB Atlas database, but always a dedicated test
database (`lecturenote_ai_test`) so your development data is never touched.
The test database is dropped when the session finishes.

Run from `backend/`:
    .venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

# Must be set before importing the application settings.
os.environ["MONGODB_DB"] = "lecturenote_ai_test"
os.environ["ALLOW_MODEL_DOWNLOAD"] = "false"
os.environ["WARMUP_MODELS"] = "false"
os.environ["OCR_ENABLED"] = "false"
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.core.config import settings
from app.main import app

UPLOADS = Path(__file__).resolve().parents[1] / "uploads"

SECTIONS = [
    (
        "1. Introduction to Natural Language Processing",
        [
            "Natural Language Processing (NLP) is the field of computer science concerned with the "
            "interaction between computers and human language. The goal of NLP is to design algorithms "
            "that analyse, understand and generate natural language text.",
            "A corpus is a structured collection of text used for training and evaluation. Tokenisation "
            "splits text into tokens such as words or subwords. Lemmatisation reduces a word to its "
            "dictionary form.",
        ],
    ),
    (
        "2. Text Representation",
        [
            "Term Frequency-Inverse Document Frequency (TF-IDF) weights each term by how often it occurs "
            "in a document and how rare it is across the corpus. Cosine similarity measures the angle "
            "between two document vectors.",
            "A unigram is a single token and a bigram is a sequence of two tokens. Language models "
            "estimate the probability of a sequence of words.",
        ],
    ),
    (
        "3. Automatic Summarisation",
        [
            "Extractive summarisation selects sentences from the source document, while abstractive "
            "summarisation generates new sentences. TextRank is a graph-based extractive algorithm in "
            "which sentences are nodes and cosine similarity provides edge weights.",
            "ROUGE evaluates summarisation by measuring overlap with a human reference summary. "
            "ROUGE-L measures the longest common subsequence between summary and reference.",
        ],
    ),
]


def make_pdf(path: Path) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 60
    for heading, paragraphs in SECTIONS:
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(60, y, heading)
        y -= 22
        pdf.setFont("Helvetica", 10)
        for paragraph in paragraphs:
            for line in _wrap(paragraph, 95):
                pdf.drawString(60, y, line)
                y -= 14
            y -= 8
        if y < 120:
            pdf.showPage()
            y = height - 60
    pdf.save()
    return path


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
    # Drop the isolated test database and any uploaded test PDFs.
    try:
        import pymongo

        sync_client = pymongo.MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
        sync_client.drop_database(settings.mongodb_db)
        sync_client.close()
    except Exception:
        pass
    if UPLOADS.exists():
        for child in UPLOADS.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


@pytest.fixture(scope="session")
def account(client: TestClient) -> dict:
    suffix = uuid.uuid4().hex[:8]
    payload = {
        "username": f"pytest_{suffix}",
        "email": f"pytest_{suffix}@example.com",
        "password": "TestPass123",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201, response.text
    payload["id"] = response.json()["user"]["id"]
    return payload


@pytest.fixture(scope="session")
def pdf_path(tmp_path_factory) -> Path:
    return make_pdf(tmp_path_factory.mktemp("pdf") / "lecture.pdf")


@pytest.fixture(scope="session")
def lecture(client: TestClient, account: dict, pdf_path: Path) -> dict:
    with pdf_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": (pdf_path.name, handle, "application/pdf")},
            data={"use_ocr": "false"},
        )
    assert response.status_code == 201, response.text
    return response.json()["lecture"]


def wait_for_job(client: TestClient, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.25)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s")


@pytest.fixture(scope="session")
def summary(client: TestClient, lecture: dict) -> dict:
    response = client.post(
        "/api/summarize",
        json={"lecture_id": lecture["id"], "method": "extractive", "length": "short", "save": True},
    )
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["id"])
    assert job["status"] == "succeeded", job.get("error")
    assert job["result"]["summaries"], "no summary returned"
    return job["result"]["summaries"][0]
