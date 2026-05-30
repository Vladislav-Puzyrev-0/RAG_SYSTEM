"""Зависимости FastAPI (Dependency Injection).

Тонкие хелперы для получения общих ресурсов в роутерах. В тестах их
переопределяют через ``app.dependency_overrides``.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status

from app.logging_config import request_id_var


def get_rag_system(request: Request) -> Any:
    """Вернуть инстанс ``QwenRAGSystem`` из lifespan-state.

    Бросает ``503`` если система не была инициализирована (например,
    приложение поднималось без модели).
    """
    rag_system = getattr(request.app.state, "rag_system", None)
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG-система не инициализирована",
        )
    return rag_system


def get_request_id() -> str:
    """Текущий ``request_id`` для прокидывания в ответы и логи."""
    return request_id_var.get()
