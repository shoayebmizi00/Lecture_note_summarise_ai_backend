"""BSON -> JSON friendly serialization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=None)
        return value.isoformat() + "Z"
    return value.isoformat()


def serialize(value: Any) -> Any:
    """Recursively convert Mongo documents into JSON-safe structures."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out["id" if key == "_id" else key] = serialize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value
