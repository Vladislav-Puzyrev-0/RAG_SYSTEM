"""Модели для эндпоинта /ask и /ask/stream."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.config import settings

#: Режим поиска: только dense, только BM25 или гибрид с RRF.
RetrievalMode = Literal["dense", "bm25", "hybrid"]


class QuestionRequest(BaseModel):
    """Запрос на ответ по корпусу документов."""

    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(
        default=settings.RETRIEVAL_DEFAULT_TOP_K,
        ge=1,
        le=settings.RETRIEVAL_MAX_TOP_K,
    )
    temperature: float = Field(
        default=settings.LLM_DEFAULT_TEMPERATURE, ge=0.0, le=2.0
    )
    # Опциональные тонкие настройки (Phase 2+).
    ef_search: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="HNSW efSearch на конкретный запрос (опционально).",
    )
    expand_context: bool = True
    context_window: int = Field(default=3, ge=0, le=10)
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Режим поиска: dense | bm25 | hybrid (по умолчанию).",
    )


class Source(BaseModel):
    """Атрибуция ответа: один цитированный фрагмент."""

    source: str
    text: Optional[str] = None
    score_rrf: Optional[float] = None
    score_dense: Optional[float] = None
    score_bm25: Optional[float] = None


class AnswerResponse(BaseModel):
    """Ответ LLM с цитатами и диагностикой."""

    answer: str
    # ``sources`` в v2 — это список строк (имён файлов). Оставляем такой формат
    # для обратной совместимости с фронтом v2; расширенные ``Source`` уйдут в
    # отдельное поле ``sources_detailed`` (Phase 3).
    sources: List[str] = Field(default_factory=list)
    sources_detailed: Optional[List[Source]] = None
    context_docs: int = 0
    error: bool = False
    # Диагностика производительности (Phase 3).
    retrieval_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    total_ms: Optional[float] = None
    cache_hit: Optional[bool] = None
