"""End-to-end API smoke test against a running backend.

Usage (server must be running on 127.0.0.1:8000):
    .venv\\Scripts\\python.exe tests\\e2e_api.py
Optional:  --method abstractive   to also exercise BART.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
PDF = Path(__file__).resolve().parents[2] / "dataset" / "sample_lecture_notes.pdf"
REFERENCE = (
    "Natural Language Processing combines linguistics, statistics and deep learning to analyse "
    "and generate human language. Corpora are collections of text used for training, and "
    "tokenisation splits text into tokens. TF-IDF weights terms by frequency and rarity, while "
    "cosine similarity compares document vectors. Word2Vec, GloVe and FastText learn dense word "
    "embeddings. BERT and GPT are Transformer models pre-trained on large corpora. Extractive "
    "summarisation selects source sentences, whereas abstractive summarisation generates new "
    "text with BART. ROUGE measures overlap with a human reference summary and does not prove "
    "factual correctness."
)


def wait_health(client: httpx.Client, timeout: float = 40.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = client.get(f"{BASE}/api/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            last = response.text
        except Exception as exc:  # server not up yet
            last = str(exc)
        time.sleep(1)
    raise SystemExit(f"Backend did not become healthy: {last}")


def poll_job(client: httpx.Client, job_id: str, timeout: float = 1800) -> dict:
    deadline = time.time() + timeout
    last_progress = -1
    while time.time() < deadline:
        response = client.get(f"{BASE}/api/jobs/{job_id}", timeout=15)
        response.raise_for_status()
        job = response.json()
        if job["progress"] != last_progress:
            print(f"    job {job['status']:9s} {job['progress']:3d}%  {job['stage']:16s} {job.get('message') or ''}")
            last_progress = job["progress"]
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(1.5)
    raise SystemExit(f"Job {job_id} timed out")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["extractive", "abstractive", "both"], default="extractive")
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:8]
    email = f"e2e_{suffix}@example.com"
    username = f"e2e_{suffix}"
    client = httpx.Client(follow_redirects=True, timeout=120)

    health = wait_health(client)
    print("health:", health["status"], "| db:", health["database"]["connected"],
          "| ocr:", health["ocr"]["available"], "| model loaded:", health["models"]["loaded"])

    r = client.post(f"{BASE}/api/auth/register", json={
        "username": username, "email": email, "password": "TestPass123",
    })
    assert r.status_code == 201, (r.status_code, r.text)
    print("register:", r.json()["user"]["username"])

    r = client.get(f"{BASE}/api/auth/me")
    assert r.status_code == 200 and r.json()["email"] == email
    print("me: ok")

    with PDF.open("rb") as handle:
        r = client.post(f"{BASE}/api/upload", files={"file": (PDF.name, handle, "application/pdf")},
                        data={"use_ocr": "true"})
    assert r.status_code == 201, (r.status_code, r.text)
    upload = r.json()
    lecture_id = upload["lecture"]["id"]
    print(f"upload: {upload['lecture']['page_count']} pages, "
          f"{upload['lecture']['word_count']} words, id={lecture_id}")
    if upload["warnings"]:
        print("  warnings:", upload["warnings"])

    r = client.get(f"{BASE}/api/lectures/{lecture_id}")
    assert r.status_code == 200 and r.json()["extracted_text"]
    print("lecture detail: ok")

    methods = ["extractive", "abstractive"] if args.method == "both" else [args.method]
    print(f"summarising with {methods} ...")
    r = client.post(f"{BASE}/api/summarize", json={
        "lecture_id": lecture_id, "method": methods[0], "length": "medium",
    })
    assert r.status_code == 202, (r.status_code, r.text)
    job = poll_job(client, r.json()["id"])
    assert job["status"] == "succeeded", job.get("error")
    summaries = job["result"]["summaries"]
    summary_id = summaries[0]["id"]
    print("  summary:", summaries[0]["summary_text"][:300].replace("\n", " "), "...")
    print("  keywords:", [k["keyword"] for k in job["result"]["keywords"][:8]])
    print("  processing time:", summaries[0]["processing_time"], "s")

    if args.method in ("abstractive", "both"):
        r = client.post(f"{BASE}/api/summarize/compare", json={"lecture_id": lecture_id, "length": "short"})
        assert r.status_code == 202, (r.status_code, r.text)
        job = poll_job(client, r.json()["id"])
        assert job["status"] == "succeeded", job.get("error")
        for variant in job["result"]["summaries"]:
            print(f"  compare[{variant['method']}]: {len(variant['summary_text'].split())} words in "
                  f"{variant['processing_time']:.1f}s")

    r = client.post(f"{BASE}/api/keywords", json={"lecture_id": lecture_id, "top_n": 12})
    assert r.status_code == 200, r.text
    print("keywords:", [(k["keyword"], k["score"]) for k in r.json()["keywords"][:6]])

    r = client.post(f"{BASE}/api/topics", json={"lecture_id": lecture_id, "max_topics": 5})
    assert r.status_code == 200, r.text
    topics = r.json()
    print(f"topics ({topics['method']}, {topics['topic_count']}):",
          [t["title"] for t in topics["topics"]])

    r = client.post(f"{BASE}/api/quiz", json={"lecture_id": lecture_id, "count": 5})
    assert r.status_code == 200, r.text
    quiz = r.json()
    print(f"quiz: generated {quiz['generated']}/{quiz['requested']} ({quiz['strategies']})")
    for q in quiz["questions"][:3]:
        print("   -", q["question"][:100], "| correct:", q["options"][q["correct_index"]])

    r = client.post(f"{BASE}/api/evaluate", json={
        "summary_id": summary_id, "reference_summary": REFERENCE,
    })
    assert r.status_code == 200, r.text
    ev = r.json()
    print("rouge:", {k: round(ev[k]["fmeasure"], 4) for k in ("rouge1", "rouge2", "rougeL")},
          "| compression:", ev["stats"]["compression_ratio"])

    r = client.get(f"{BASE}/api/summary/{summary_id}/download/pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF", r.status_code
    (Path(__file__).parent / "_e2e_summary.pdf").write_bytes(r.content)
    print(f"pdf download: {len(r.content)} bytes")

    r = client.get(f"{BASE}/api/summary/{summary_id}/download/txt")
    assert r.status_code == 200 and "SUMMARY" in r.text
    print(f"txt download: {len(r.text)} chars")

    r = client.get(f"{BASE}/api/lectures/{lecture_id}/download/keywords?format=csv")
    assert r.status_code == 200 and "keyword" in r.text
    print("keywords csv: ok")

    r = client.get(f"{BASE}/api/lectures/{lecture_id}/download/quiz?format=csv")
    assert r.status_code == 200
    print("quiz csv: ok")

    r = client.get(f"{BASE}/api/summaries")
    assert r.status_code == 200 and r.json()["total"] >= 1
    print("history:", r.json()["total"], "summaries")

    r = client.get(f"{BASE}/api/stats")
    assert r.status_code == 200
    print("stats:", json.dumps({k: r.json()[k] for k in (
        "lecture_count", "summary_count", "avg_processing_time")}))

    r = client.delete(f"{BASE}/api/lectures/{lecture_id}")
    assert r.status_code == 200, r.text
    print("cleanup:", r.json()["message"])

    r = client.delete(f"{BASE}/api/auth/account")
    assert r.status_code == 200, r.text
    print("account deleted")
    print("\nE2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
