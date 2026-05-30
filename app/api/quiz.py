"""
Эндпоинты генерации тестов.

POST /quiz/generate — создаёт тест по теме (topic) и/или конкретному документу
                      (source_filename) через :class:`app.rag.quiz.QuizGenerator`.
GET  /quiz/sources  — список загруженных документов, удобно для UI-выпадашки.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_rag_system
from app.models import QuizRequest, QuizResponse
from app.rag.quiz import QuizGenerator

router = APIRouter()
logger = logging.getLogger("app.api.quiz")


@router.post(
    "/quiz/generate",
    response_model=QuizResponse,
    summary="Генерация теста",
)
async def generate_quiz(
    body: QuizRequest,
    rag_system: Any = Depends(get_rag_system),
) -> QuizResponse:
    """Сгенерировать образовательный тест по загруженным материалам."""
    if not body.topic and not body.source_filename:
        raise HTTPException(
            status_code=400,
            detail="Нужно указать topic или source_filename",
        )

    if body.source_filename and body.source_filename not in rag_system.hashes:
        known = sorted(rag_system.hashes.keys())
        raise HTTPException(
            status_code=404,
            detail=(
                f"Документ '{body.source_filename}' не найден. "
                f"Загружены: {known}"
            ),
        )

    logger.info(
        "Quiz: %d вопросов, topic=%r, source=%r, types=%s, lvl=%s",
        body.num_questions,
        body.topic,
        body.source_filename,
        body.question_types,
        body.difficulty,
    )

    generator = QuizGenerator(rag_system)
    return await generator.generate(body)


@router.get(
    "/quiz/sources",
    summary="Список доступных источников для теста",
)
async def quiz_sources(rag_system: Any = Depends(get_rag_system)):
    """Список документов, доступных для генерации тестов."""
    return {
        "sources": sorted(rag_system.hashes.keys()),
        "total": len(rag_system.hashes),
    }
