"""
Централизованная конфигурация RAG-системы из ``.env``.

Использует ``pydantic-settings``. Новые ключи добавляются БЕЗ нарушения
обратной совместимости: для каждого переименованного ключа задаётся
``AliasChoices`` со старым именем.
"""
from __future__ import annotations

import json
from typing import List, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Глобальные настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Server ─────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["console", "json"] = "console"
    CORS_ORIGINS: str = '["http://localhost:3000","http://localhost:8000"]'

    @property
    def cors_list(self) -> List[str]:
        """Распарсить JSON-строку ``CORS_ORIGINS`` в список."""
        try:
            return json.loads(self.CORS_ORIGINS)
        except (json.JSONDecodeError, TypeError):
            return ["*"]

    # ── Document ingestion ─────────────────────────────────
    DOCS_DIR: str = "documents"
    SUPPORTED_EXTENSIONS: str = ".pdf,.docx,.doc,.txt"
    MAX_UPLOAD_MB: int = 50

    @property
    def supported_extensions_list(self) -> List[str]:
        """Список поддерживаемых расширений (с точками, в нижнем регистре)."""
        return [
            ext.strip().lower()
            for ext in self.SUPPORTED_EXTENSIONS.split(",")
            if ext.strip()
        ]

    # ── Chunking ───────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Embeddings ─────────────────────────────────────────
    EMBEDDING_MODEL: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_NORMALIZE: bool = True

    # ── Vector store (FAISS) ───────────────────────────────
    VECTORSTORE_PATH: str = "vectorstore_index"
    FAISS_USE_HNSW: bool = True
    FAISS_HNSW_M: int = 32
    FAISS_HNSW_EF_CONSTRUCTION: int = 200
    FAISS_HNSW_EF_SEARCH: int = 64
    FAISS_USE_GPU: bool = True
    # ``ip`` = inner-product (на нормализованных векторах ≡ косинусу).
    FAISS_METRIC: Literal["ip", "l2"] = "ip"

    # ── Hybrid retrieval (BM25 + dense + RRF) ──────────────
    HYBRID_ENABLED: bool = True
    HYBRID_RRF_K: int = 60
    HYBRID_TOP_K_INTERNAL_FACTOR: int = 4
    BM25_PATH: str = "vectorstore_index/bm25.pkl"

    # ── Cache (SQLite) ─────────────────────────────────────
    RAG_CACHE_ENABLED: bool = True
    RAG_CACHE_PATH: str = "vectorstore_index/cache.sqlite"

    # ── LLM ────────────────────────────────────────────────
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_DEFAULT_TEMPERATURE: float = 0.3
    LLM_DEVICE: str = "auto"

    # ── Retrieval ──────────────────────────────────────────
    RETRIEVAL_DEFAULT_TOP_K: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "RETRIEVAL_DEFAULT_TOP_K", "DEFAULT_TOP_K"
        ),
    )
    RETRIEVAL_MAX_TOP_K: int = Field(
        default=20,
        validation_alias=AliasChoices("RETRIEVAL_MAX_TOP_K", "MAX_TOP_K"),
    )

    # ── Deduplication ──────────────────────────────────────
    DEDUP_ENABLED: bool = True


settings = Settings()
