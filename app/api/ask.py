"""
Эндпоинты /ask и /ask/stream.

POST /ask         — синхронный ответ (через гибридный retriever + LLM + кэш).
POST /ask/stream  — Server-Sent Events, события ``token | done | error``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import get_rag_system
from app.models import AnswerResponse, QuestionRequest

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_expand(body: QuestionRequest, query_expand: Optional[bool]) -> bool:
    """Бэк-совместимость: query-параметр ``expand`` имеет приоритет."""
    return body.expand_context if query_expand is None else query_expand


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    body: QuestionRequest,
    rag_system: Any = Depends(get_rag_system),
    expand: Optional[bool] = None,
):
    """Задать вопрос модели (синхронный ответ)."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    temperature = max(0.0, min(2.0, body.temperature))
    result = await rag_system.ask_question(
        body.question,
        body.top_k,
        temperature,
        expand_context=_resolve_expand(body, expand),
        context_window=body.context_window,
        mode=body.mode,
        ef_search=body.ef_search,
    )
    return AnswerResponse(**result)


@router.post("/ask/stream")
async def ask_question_stream(
    body: QuestionRequest,
    rag_system: Any = Depends(get_rag_system),
    expand: Optional[bool] = None,
):
    """Стримить ответ через Server-Sent Events.

    Формат кадров::

        event: token
        data: {"text": "..."}

        event: done
        data: {"sources": [...], "context_docs": N, "llm_ms": ...}

        event: error
        data: {"message": "..."}
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    temperature = max(0.0, min(2.0, body.temperature))
    expand_ctx = _resolve_expand(body, expand)

    async def event_gen():
        async for evt, payload in rag_system.ask_question_stream(
            body.question,
            body.top_k,
            temperature,
            expand_context=expand_ctx,
            context_window=body.context_window,
            mode=body.mode,
            ef_search=body.ef_search,
        ):
            data = json.dumps(payload, ensure_ascii=False)
            yield f"event: {evt}\ndata: {data}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
