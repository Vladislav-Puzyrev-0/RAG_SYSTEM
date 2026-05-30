"""Общие модели, не привязанные к конкретному эндпоинту."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Единый формат ошибки, возвращаемый глобальным exception handler."""

    error: bool = True
    code: str = Field(..., description="Машинно-читаемый код ошибки.")
    message: str = Field(..., description="Текст ошибки для человека.")
    request_id: str = Field("-", description="ID запроса для трассировки.")


class StatusResponse(BaseModel):
    """Состояние RAG-системы для UI и health-чеков."""

    status: str
    has_vectorstore: bool
    model_device: str
    loaded: bool
    documents_count: int
    hnsw_enabled: bool
    # Расширенные поля v3 (опциональны — Phase 2/3 их заполнят).
    chunks_count: Optional[int] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    ef_search: Optional[int] = None
    hybrid_enabled: Optional[bool] = None
    embedding_metric: Optional[str] = None
    cache_enabled: Optional[bool] = None
    cache_size_bytes: Optional[int] = None
    bm25_loaded: Optional[bool] = None


class RetrievedChunk(BaseModel):
    """Один найденный фрагмент с метриками гибридного поиска."""

    doc_id: str
    source: str
    text: str
    rank: int
    score_dense: Optional[float] = None
    score_bm25: Optional[float] = None
    score_rrf: Optional[float] = None
