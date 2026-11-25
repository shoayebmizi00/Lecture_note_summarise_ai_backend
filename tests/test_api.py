"""API integration tests for LectureNote AI.

These tests exercise the real FastAPI app, the real NLP pipeline (TextRank,
TF-IDF, topics, quiz, ROUGE) and the real MongoDB test database. The BART
abstractive path is intentionally not covered here because CPU inference takes
several minutes; `tests/e2e_api.py --method both` covers it manually.

Run from `backend/`:
    .venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app

REFERENCE = (
    "Natural Language Processing combines computer science and linguistics to analyse and generate "
    "human language. Text is prepared with tokenisation and lemmatisation, and represented with "
    "TF-IDF weights and cosine similarity. TextRank ranks sentences in a graph for extractive "
    "summarisation, while BART generates abstractive summaries. ROUGE measures overlap with a human "
    "reference summary."
)


def test_health_is_public(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"]["connected"] is True
    assert body["models"]["primary_model"] == "facebook/bart-large-cnn"
    assert body["limits"]["max_upload_mb"] > 0


def test_auth_required_for_private_routes(client: TestClient) -> None:
    anonymous = TestClient(app)
    assert anonymous.get("/api/auth/me").status_code == 401
    assert anonymous.get("/api/lectures").status_code == 401
    assert anonymous.get("/api/stats").status_code == 401
    anonymous.close()


def test_register_login_and_me(client: TestClient, account: dict) -> None:
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == account["email"]
    assert me.json()["settings"]["default_method"] == "extractive"

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401

    login = client.post(
        "/api/auth/login",
        json={"identifier": account["email"], "password": account["password"]},
    )
    assert login.status_code == 200, login.text
    assert login.json()["user"]["username"] == account["username"]
    assert client.get("/api/auth/me").status_code == 200


def test_duplicate_registration_rejected(client: TestClient, account: dict) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": account["username"],
            "email": f"other_{uuid.uuid4().hex[:6]}@example.com",
            "password": "TestPass123",
        },
    )
    assert response.status_code == 409

    response = client.post(
        "/api/auth/register",
        json={
            "username": f"other_{uuid.uuid4().hex[:6]}",
            "email": account["email"],
            "password": "TestPass123",
        },
    )
    assert response.status_code == 409


def test_weak_password_and_bad_email_rejected(client: TestClient) -> None:
    weak = client.post(
        "/api/auth/register",
        json={"username": "weakuser", "email": "weak@example.com", "password": "onlyletters"},
    )
    assert weak.status_code == 422

    bad_email = client.post(
        "/api/auth/register",
        json={"username": "badmail", "email": "not-an-email", "password": "TestPass123"},
    )
    assert bad_email.status_code == 422


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("notes.pdf", b"this is definitely not a pdf", "application/pdf")},
        data={"use_ocr": "false"},
    )
    assert response.status_code == 415


def test_upload_extracts_text(client: TestClient, lecture: dict) -> None:
    assert lecture["page_count"] >= 1
    assert lecture["word_count"] > 100
    assert lecture["sentence_count"] > 5
    assert lecture["processing_status"] == "ready"

    detail = client.get(f"/api/lectures/{lecture['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert "TextRank" in body["extracted_text"]
    assert len(body["extracted_text"]) > 500
    assert body["headings"], "expected heading detection on the generated PDF"


def test_lecture_listing_and_stats(client: TestClient, lecture: dict, summary: dict) -> None:
    listing = client.get("/api/lectures", params={"limit": 10})
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1
    assert any(item["id"] == lecture["id"] for item in listing.json()["items"])

    search = client.get("/api/lectures", params={"q": "lecture"})
    assert search.status_code == 200

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["lecture_count"] >= 1
    assert body["summary_count"] >= 1  # the summary fixture has run
    assert "summaries_by_method" in body


def test_extractive_summary_job(client: TestClient, lecture: dict, summary: dict) -> None:
    assert summary["method"] == "extractive"
    assert summary["summary_text"].strip()
    assert summary["summary_word_count"] > 0
    assert summary["compression_ratio"] is not None
    assert 0 < summary["compression_ratio"] < 1
    assert summary["processing_time"] >= 0
    assert summary["selected_sentences"], "TextRank must report the selected sentences"

    selected = [s for s in summary["selected_sentences"] if s.get("selected", True)]
    assert selected, "at least one selected sentence expected"
    assert all(s["text"].strip() for s in selected)
    # Extractive output must be made of source sentences.
    source = client.get(f"/api/lectures/{lecture['id']}").json()["extracted_text"]
    for sentence in selected[:3]:
        assert sentence["text"][:40] in source


def test_job_unknown_id(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_summary_detail_and_history(client: TestClient, summary: dict, lecture: dict) -> None:
    detail = client.get(f"/api/summaries/{summary['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["summary"]["id"] == summary["id"]
    assert body["lecture"]["id"] == lecture["id"]
    assert body["lecture_id"] == lecture["id"]

    history = client.get("/api/summaries", params={"lecture_id": lecture["id"]})
    assert history.status_code == 200
    assert history.json()["total"] >= 1

    filtered = client.get("/api/summaries", params={"method": "abstractive"})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 0


def test_keywords(client: TestClient, lecture: dict) -> None:
    response = client.post("/api/keywords", json={"lecture_id": lecture["id"], "top_n": 12, "save": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"] in ("tfidf", "keybert")
    assert 5 <= len(body["keywords"]) <= 12
    scores = [item["score"] for item in body["keywords"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["frequency"] >= 1 for item in body["keywords"])

    stored = client.get(f"/api/lectures/{lecture['id']}/keywords")
    assert stored.status_code == 200
    assert stored.json()["count"] == len(body["keywords"])


def test_topics(client: TestClient, lecture: dict) -> None:
    response = client.post("/api/topics", json={"lecture_id": lecture["id"], "max_topics": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["topic_count"] >= 1
    assert body["method"] in ("headings", "clusters")
    assert len(body["topics"]) == body["topic_count"]
    for topic in body["topics"]:
        assert topic["title"].strip()
        assert topic["source"] in ("heading", "cluster", "document")
        assert topic["summary"].strip()
        assert topic["sentence_count"] >= 1


def test_quiz(client: TestClient, lecture: dict) -> None:
    response = client.post("/api/quiz", json={"lecture_id": lecture["id"], "count": 5, "save": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generated"] >= 1
    assert body["generated"] <= 5
    assert len(body["questions"]) == body["generated"]
    for question in body["questions"]:
        assert len(question["options"]) == 4
        assert len(set(question["options"])) == 4
        assert 0 <= question["correct_index"] < 4
        assert question["correct_answer"] == question["options"][question["correct_index"]]
        assert question["source_sentence"].strip()
        assert question["explanation"].strip()

    stored = client.get(f"/api/lectures/{lecture['id']}/quiz")
    assert stored.status_code == 200
    assert stored.json()["count"] == body["generated"]


def test_evaluation(client: TestClient, summary: dict) -> None:
    response = client.post(
        "/api/evaluate",
        json={"summary_id": summary["id"], "reference_summary": REFERENCE},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    for key in ("rouge1", "rouge2", "rougeL"):
        metric = body[key]
        assert 0.0 <= metric["fmeasure"] <= 1.0
        assert 0.0 <= metric["precision"] <= 1.0
        assert 0.0 <= metric["recall"] <= 1.0
    assert body["stats"]["summary_words"] > 0
    assert body["stats"]["source_words"] > 0
    assert body["summary_id"] == summary["id"]

    too_short = client.post(
        "/api/evaluate",
        json={"summary_id": summary["id"], "reference_summary": "too short"},
    )
    assert too_short.status_code == 422


def test_compare_evaluation_requires_both_methods(client: TestClient, lecture: dict) -> None:
    response = client.post(
        "/api/evaluate/compare",
        json={"lecture_id": lecture["id"], "reference_summary": REFERENCE},
    )
    assert response.status_code == 409
    assert "abstractive" in response.json()["detail"]


def test_downloads(client: TestClient, summary: dict, lecture: dict) -> None:
    pdf = client.get(f"/api/summary/{summary['id']}/download/pdf")
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"
    assert len(pdf.content) > 1000

    txt = client.get(f"/api/summary/{summary['id']}/download/txt")
    assert txt.status_code == 200
    assert "SUMMARY" in txt.text.upper()
    assert summary["summary_text"][:40] in txt.text

    keywords_csv = client.get(f"/api/lectures/{lecture['id']}/download/keywords", params={"format": "csv"})
    assert keywords_csv.status_code == 200
    assert "keyword" in keywords_csv.text.lower()

    quiz_csv = client.get(f"/api/lectures/{lecture['id']}/download/quiz", params={"format": "csv"})
    assert quiz_csv.status_code == 200

    original = client.get(f"/api/lectures/{lecture['id']}/download/original")
    assert original.status_code == 200
    assert original.content[:4] == b"%PDF"


def test_ownership_isolation(client: TestClient, lecture: dict, summary: dict) -> None:
    suffix = uuid.uuid4().hex[:8]
    intruder = TestClient(app)
    register = intruder.post(
        "/api/auth/register",
        json={
            "username": f"intruder_{suffix}",
            "email": f"intruder_{suffix}@example.com",
            "password": "TestPass123",
        },
    )
    assert register.status_code == 201, register.text

    assert intruder.get(f"/api/lectures/{lecture['id']}").status_code == 404
    assert intruder.get(f"/api/summaries/{summary['id']}").status_code == 404
    assert intruder.get(f"/api/summary/{summary['id']}/download/txt").status_code == 404
    assert intruder.post(
        "/api/summarize",
        json={"lecture_id": lecture["id"], "method": "extractive", "length": "short"},
    ).status_code == 404

    intruder.delete("/api/auth/account")
    intruder.close()


def test_settings_update(client: TestClient) -> None:
    response = client.patch(
        "/api/auth/settings",
        json={"theme": "dark", "default_method": "abstractive", "default_length": "long", "use_ocr": False},
    )
    assert response.status_code == 200, response.text
    settings = response.json()["settings"]
    assert settings["theme"] == "dark"
    assert settings["default_method"] == "abstractive"

    reset = client.patch("/api/auth/settings", json={"theme": "light", "default_method": "extractive"})
    assert reset.status_code == 200


def test_delete_summary_then_lecture(client: TestClient, lecture: dict) -> None:
    created = client.post(
        "/api/summarize",
        json={"lecture_id": lecture["id"], "method": "extractive", "length": "short", "save": True},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    from tests.conftest import wait_for_job

    job = wait_for_job(client, job_id)
    assert job["status"] == "succeeded"
    summary_id = job["result"]["summaries"][0]["id"]

    deleted = client.delete(f"/api/summaries/{summary_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/summaries/{summary_id}").status_code == 404

    removed = client.delete(f"/api/lectures/{lecture['id']}")
    assert removed.status_code == 200
    assert client.get(f"/api/lectures/{lecture['id']}").status_code == 404
    assert client.get(f"/api/lectures/{lecture['id']}/download/original").status_code == 404
