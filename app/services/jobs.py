"""Async job manager for long-running NLP work.

Jobs run in a worker thread (so BART inference never blocks the event loop) and
their progress is tracked in memory while the final state is persisted to MongoDB
for history / restart resilience.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.logging import get_logger
from app.database.mongo import JOBS, db
from app.utils.serialization import serialize

logger = get_logger(__name__)

JobFn = Callable[[Callable[[int, str, str], None]], dict[str, Any]]


class Job:
    __slots__ = (
        "id", "user_id", "type", "status", "progress", "stage", "message",
        "lecture_id", "result", "error", "created_at", "updated_at",
    )

    def __init__(self, job_type: str, user_id: str, lecture_id: str | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.user_id = user_id
        self.type = job_type
        self.status = "queued"
        self.progress = 0
        self.stage = "queued"
        self.message: str | None = None
        self.lecture_id = lecture_id
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "lecture_id": self.lecture_id,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()

    async def submit(self, job_type: str, user_id: str, fn: JobFn,
                     lecture_id: str | None = None,
                     on_success: Callable[[dict[str, Any]], Any] | None = None) -> Job:
        job = Job(job_type, user_id, lecture_id)
        with self._lock:
            self._jobs[job.id] = job
        await self._persist(job)
        task = asyncio.create_task(self._execute(job, fn, on_success))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _execute(self, job: Job, fn: JobFn, on_success=None) -> None:
        job.status = "running"
        job.stage = "starting"
        job.progress = 1
        job.message = "Preparing document"
        await self._persist(job)

        def progress(percent: int, stage: str, message: str) -> None:
            job.progress = max(0, min(99, int(percent)))
            job.stage = stage
            job.message = message
            job.updated_at = datetime.now(timezone.utc)

        try:
            # BART / TextRank work is CPU bound -> keep it off the event loop.
            result = await asyncio.wait_for(asyncio.to_thread(fn, progress), timeout=3600)
            if on_success is not None:
                progress(99, "saving", "Saving results")
                result = await on_success(result)
            job.status = "succeeded"
            job.progress = 100
            job.stage = "completed"
            job.message = "Completed"
            job.result = result
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = "The job timed out after 60 minutes."
            job.stage = "failed"
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "The job was cancelled (server shutdown)."
            job.stage = "failed"
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Job %s (%s) failed", job.id, job.type)
            job.status = "failed"
            job.error = f"{exc}"
            job.stage = "failed"
            job.message = None
        finally:
            job.updated_at = datetime.now(timezone.utc)
            await self._persist(job)
            if job.status == "succeeded":
                logger.info("Job %s (%s) finished", job.id, job.type)

    async def _persist(self, job: Job) -> None:
        try:
            payload = job.as_dict()
            payload.pop("id", None)
            await db[JOBS].update_one(
                {"job_id": job.id},
                {"$set": {**serialize(payload), "job_id": job.id}},
                upsert=True,
            )
        except Exception as exc:  # persistence must never kill the job
            logger.warning("Could not persist job %s: %s", job.id, exc)

    async def get(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            if job.user_id != user_id:
                return None
            return job.as_dict()
        doc = await db[JOBS].find_one({"job_id": job_id, "user_id": user_id})
        if not doc:
            return None
        return serialize(doc)

    async def recent(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        cursor = db[JOBS].find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        return [serialize(doc) async for doc in cursor]

    def prune(self, max_age_seconds: int = 6 * 3600) -> None:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            stale = [
                job_id for job_id, job in self._jobs.items()
                if job.updated_at.timestamp() < cutoff and job.status in ("succeeded", "failed")
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)


job_manager = JobManager()
