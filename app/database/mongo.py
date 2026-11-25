"""Async MongoDB (Atlas) connection and collection access."""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

USERS = "users"
LECTURES = "lectures"
SUMMARIES = "summaries"
KEYWORDS = "keywords"
QUIZZES = "quizzes"
JOBS = "jobs"
ACTIVITY = "activity"


class Database:
    """Thin async wrapper so tests can swap the client easily."""

    def __init__(self) -> None:
        self.client: AsyncMongoClient | None = None
        self.db: Any = None

    async def connect(self) -> None:
        if self.client is not None:
            return
        self.client = AsyncMongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
            tz_aware=True,
            appname="lecturenote-ai",
            retryWrites=True,
        )
        self.db = self.client[settings.mongodb_db]
        await self.client.admin.command("ping")
        await self.ensure_indexes()
        logger.info("Connected to MongoDB database '%s'", settings.mongodb_db)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
            self.db = None

    @property
    def is_connected(self) -> bool:
        return self.db is not None

    def __getitem__(self, name: str):
        if self.db is None:
            raise RuntimeError("Database is not connected yet.")
        return self.db[name]

    async def ping(self) -> bool:
        try:
            if self.client is None:
                return False
            await self.client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    async def ensure_indexes(self) -> None:
        users = self[USERS]
        await users.create_index([("email", ASCENDING)], unique=True, name="uniq_email")
        await users.create_index([("username_lower", ASCENDING)], unique=True, name="uniq_username")

        lectures = self[LECTURES]
        await lectures.create_index([("user_id", ASCENDING), ("upload_date", DESCENDING)])
        await lectures.create_index([("user_id", ASCENDING), ("original_filename", ASCENDING)])

        summaries = self[SUMMARIES]
        await summaries.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        await summaries.create_index([("lecture_id", ASCENDING), ("method", ASCENDING)])

        await self[KEYWORDS].create_index([("lecture_id", ASCENDING), ("score", DESCENDING)])
        await self[QUIZZES].create_index([("lecture_id", ASCENDING), ("created_at", DESCENDING)])
        await self[JOBS].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        await self[JOBS].create_index([("status", ASCENDING)])
        await self[ACTIVITY].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])

    # --- helpers --------------------------------------------------------
    @staticmethod
    def oid(value: str | ObjectId) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid identifier")
        return ObjectId(value)


db = Database()
