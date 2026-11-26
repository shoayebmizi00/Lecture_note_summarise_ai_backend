"""Application configuration, loaded from backend/.env (pydantic-settings)."""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app ------------------------------------------------------------
    app_name: str = "LectureNote AI"
    environment: str = "development"
    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = 8000

    # --- database (MongoDB Atlas) ---------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "lecturenote_ai"
    mongodb_timeout_ms: int = 20000

    # --- auth -----------------------------------------------------------
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    cookie_name: str = "lnai_token"
    cookie_secure: bool = False

    # --- http -----------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- storage --------------------------------------------------------
    upload_dir: Path = BACKEND_DIR / "uploads"
    output_dir: Path = BACKEND_DIR / "outputs"
    max_upload_mb: int = 25
    max_pdf_pages: int = 400
    max_extracted_chars: int = 900_000

    # --- pdf / ocr ------------------------------------------------------
    ocr_enabled: bool = True
    ocr_min_chars_per_page: int = 60
    tesseract_cmd: str = ""
    ocr_dpi: int = 200

    # --- models ---------------------------------------------------------
    model_dir: Path = ROOT_DIR / "model" / "cache"
    bart_model: str = "facebook/bart-large-cnn"
    fallback_model: str = "sshleifer/distilbart-cnn-12-6"
    prefer_light_model: bool = False  # force the distilbart fallback
    allow_model_download: bool = True  # False = offline / cache-only
    warmup_models: bool = False
    max_input_tokens: int = 1024
    chunk_token_budget: int = 900
    chunk_overlap_sentences: int = 1
    abstractive_max_chunks: int = 30
    summary_batch_size: int = 4
    bart_num_beams: int = 4  # beam search width; 4 is noticeably better than greedy
    extractive_mmr_lambda: float = 0.65  # relevance vs. redundancy trade-off

    # --- optional semantic keywords -------------------------------------
    enable_keybert: bool = False
    keybert_model: str = "all-MiniLM-L6-v2"

    # --- misc -----------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("jwt_secret")
    @classmethod
    def _default_secret(cls, v: str) -> str:
        # A random secret keeps the app usable out of the box; .env should set one.
        return v or secrets.token_urlsafe(48)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def hf_home(self) -> Path:
        return self.model_dir

    def prepare_dirs(self) -> None:
        for path in (self.upload_dir, self.output_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_dirs()
    # Hugging Face must see these before transformers is imported.
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.hf_home / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if not settings.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if settings.tesseract_cmd:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        except Exception:  # pragma: no cover - optional dependency
            pass
    return settings


settings = get_settings()
