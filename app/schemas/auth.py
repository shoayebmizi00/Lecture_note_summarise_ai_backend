"""Request/response schemas for authentication and users."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._\-]{3,32}$")


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=32)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        value = value.strip()
        if not _USERNAME_RE.match(value):
            raise ValueError("Username may only contain letters, numbers, dot, dash and underscore.")
        return value

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_RE.match(value):
            raise ValueError("Please enter a valid email address.")
        return value

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
            raise ValueError("Password must contain at least one letter and one number.")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(min_length=3, max_length=254, description="Email or username")
    password: str = Field(min_length=1, max_length=128)


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Literal["light", "dark"] = "light"
    default_method: Literal["extractive", "abstractive"] = "extractive"
    default_length: Literal["short", "medium", "long"] = "medium"
    use_ocr: bool = True


class UserSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Literal["light", "dark"] | None = None
    default_method: Literal["extractive", "abstractive"] | None = None
    default_length: Literal["short", "medium", "long"] | None = None
    use_ocr: bool | None = None


class UserOut(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime | str | None = None
    settings: UserSettings = Field(default_factory=UserSettings)


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str | None = None
    token_type: str = "bearer"
