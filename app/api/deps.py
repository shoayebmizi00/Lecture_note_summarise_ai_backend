"""Authentication dependencies (httpOnly cookie or Bearer token)."""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from fastapi import Cookie, Depends, Header, HTTPException, status

from app.core.security import decode_access_token
from app.database.mongo import USERS, db


async def _user_from_token(token: str) -> dict[str, Any]:
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if not subject or not ObjectId.is_valid(subject):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject.")
    user = await db[USERS].find_one({"_id": ObjectId(subject)})
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account not found.")
    return user


async def get_current_user(
    lnai_token: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = lnai_token
    if not token and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            token = value.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await _user_from_token(token)


CurrentUser = Depends(get_current_user)
