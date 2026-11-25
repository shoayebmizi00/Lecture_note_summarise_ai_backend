"""Authentication routes: register, login, logout, profile, settings."""

from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pymongo.errors import DuplicateKeyError

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.database.mongo import USERS, db
from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserOut,
    UserSettings,
    UserSettingsUpdate,
)

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)

DEFAULT_SETTINGS = {
    "theme": "light",
    "default_method": "extractive",
    "default_length": "medium",
    "use_ocr": True,
}


def _settings_for(doc: dict) -> UserSettings:
    raw = doc.get("settings") or {}
    try:
        return UserSettings(**raw)
    except Exception:
        return UserSettings()


def user_public(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "username": doc.get("username", ""),
        "email": doc.get("email", ""),
        "created_at": doc.get("created_at"),
        "settings": _settings_for(doc),
    }


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


async def _load_user(user_id: ObjectId) -> dict:
    user = await db[USERS].find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account not found.")
    return user


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, response: Response) -> AuthResponse:
    now = datetime.now(timezone.utc)
    doc = {
        "username": payload.username,
        "username_lower": payload.username.lower(),
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "created_at": now,
        "last_login_at": now,
        "settings": dict(DEFAULT_SETTINGS),
    }
    try:
        result = await db[USERS].insert_one(doc)
    except DuplicateKeyError as exc:
        field = "email" if "email" in str(exc).lower() else "username"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"That {field} is already registered.",
        ) from exc

    doc["_id"] = result.inserted_id
    token = create_access_token(str(doc["_id"]), {"username": doc["username"]})
    _set_auth_cookie(response, token)
    logger.info("New account created: %s", doc["username"])
    return AuthResponse(user=UserOut(**user_public(doc)), access_token=token)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, response: Response) -> AuthResponse:
    identifier = payload.identifier.strip()
    query = {"email": identifier.lower()} if "@" in identifier else {"username_lower": identifier.lower()}
    user = await db[USERS].find_one(query)
    if user is None or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password.",
        )
    await db[USERS].update_one({"_id": user["_id"]}, {"$set": {"last_login_at": datetime.now(timezone.utc)}})
    token = create_access_token(str(user["_id"]), {"username": user.get("username", "")})
    _set_auth_cookie(response, token)
    return AuthResponse(user=UserOut(**user_public(user)), access_token=token)


@router.post("/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(settings.cookie_name, path="/")
    return {"message": "Signed out."}


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)) -> UserOut:
    return UserOut(**user_public(user))


@router.patch("/settings", response_model=UserOut)
async def update_settings(payload: UserSettingsUpdate, user: dict = Depends(get_current_user)) -> UserOut:
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if not updates:
        return UserOut(**user_public(user))
    await db[USERS].update_one(
        {"_id": user["_id"]},
        {"$set": {f"settings.{key}": value for key, value in updates.items()}},
    )
    refreshed = await _load_user(user["_id"])
    return UserOut(**user_public(refreshed))


@router.delete("/account", status_code=status.HTTP_200_OK)
async def delete_account(response: Response, user: dict = Depends(get_current_user)) -> dict:
    """Deletes the account and every lecture/summary owned by it."""
    from app.database.mongo import JOBS, KEYWORDS, LECTURES, QUIZZES, SUMMARIES
    from app.services.storage import delete_stored_file

    user_id = user["_id"]
    async for lecture in db[LECTURES].find({"user_id": user_id}, {"stored_filename": 1}):
        delete_stored_file(lecture.get("stored_filename", ""))
    for collection in (SUMMARIES, KEYWORDS, QUIZZES, JOBS, LECTURES):
        await db[collection].delete_many({"user_id": user_id})
    await db[USERS].delete_one({"_id": user_id})
    response.delete_cookie(settings.cookie_name, path="/")
    return {"message": "Account and all associated data deleted."}
