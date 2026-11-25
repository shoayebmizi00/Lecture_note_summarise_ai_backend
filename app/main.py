"""LectureNote AI - FastAPI application entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, auth, downloads, health, lectures, summaries
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.database.mongo import db
from app.services.analysis_service import model_status
from app.services.jobs import job_manager

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v1.0.0 (%s)", settings.app_name, settings.environment)
    try:
        await db.connect()
    except Exception as exc:
        logger.error(
            "MongoDB connection failed: %s. Check MONGODB_URI in backend/.env and "
            "your network / Atlas IP allow-list.", exc,
        )
        raise
    job_manager.prune()
    if settings.warmup_models:
        logger.info("Warming up the abstractive model in the background...")

        def _warm():
            try:
                from app.nlp.abstractive import get_summarizer

                get_summarizer().ensure_loaded()
            except Exception as exc:
                logger.warning("Model warm-up failed: %s", exc)

        asyncio.create_task(asyncio.to_thread(_warm))
    yield
    logger.info("Shutting down; closing MongoDB connection")
    await db.close()


app = FastAPI(
    title="LectureNote AI API",
    description=(
        "Intelligent lecture note summarisation and study assistant: PDF extraction, "
        "TextRank and BART summarisation, keywords, topics, quiz generation and ROUGE evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=None,
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred. Check the backend logs."},
    )


for router in (auth.router, lectures.router, summaries.router, analysis.router,
               downloads.router, health.router):
    app.include_router(router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health",
        "models": model_status(),
    }
