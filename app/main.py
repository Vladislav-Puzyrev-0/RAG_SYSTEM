"""
FastAPI Application — Lifespan, Middleware, DI, Routes, Metrics.

Точка входа Qwen RAG System v3.0 (рефакторинг Phase 1: только структура,
без изменения функциональности retrieval/LLM).
"""
# ── HuggingFace Offline Mode — ДО ВСЕХ импортов transformers ─────────
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import ask, documents, health, quiz, upload
from app.config import settings
from app.logging_config import setup_logging
from app.metrics import DOCUMENTS_COUNT
from app.middleware import RequestIDMiddleware, install_exception_handler
from app.rag.system import QwenRAGSystem

setup_logging()
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Запуск/остановка приложения: инициализация ``QwenRAGSystem``."""
    loop = asyncio.get_running_loop()

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Получен сигнал завершения")
        shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows не поддерживает add_signal_handler для SIGINT.
        pass

    Path(settings.DOCS_DIR).mkdir(exist_ok=True)
    logger.info("Запуск RAG-системы…")
    rag_system = QwenRAGSystem()
    app.state.rag_system = rag_system
    DOCUMENTS_COUNT.set(len(rag_system.hashes))
    logger.info(f"RAG-система готова, документов: {len(rag_system.hashes)}")

    yield

    logger.info("Завершение работы…")
    if getattr(app.state, "rag_system", None) and app.state.rag_system.vectorstore:
        app.state.rag_system._save_vectorstore()
        logger.info("Векторстор сохранён")
    logger.info("RAG-система остановлена")


app = FastAPI(
    title="Qwen RAG System",
    version="3.0.0",
    lifespan=lifespan,
)

# RequestID — ДО CORS, чтобы request_id попадал в логи на всех этапах.
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_exception_handler(app)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    """Отдать одностраничный фронт."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(ask.router, tags=["Ask"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(quiz.router, tags=["Quiz"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
