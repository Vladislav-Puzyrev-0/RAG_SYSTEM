"""Модели для эндпоинта /quiz/generate (новый формат v3)."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal["mcq", "true_false", "fill_in_blank"]
Difficulty = Literal["easy", "medium", "hard"]
Language = Literal["ru", "en"]


class QuizRequest(BaseModel):
    """Запрос на генерацию теста.

    Должен быть указан хотя бы один из ``topic`` или ``source_filename``.
    """

    topic: Optional[str] = Field(
        None, description="Тема для retrieval (HybridRetriever ищет релевантные чанки)."
    )
    source_filename: Optional[str] = Field(
        None, description="Привязка к конкретному документу из ``documents/``."
    )
    num_questions: int = Field(5, ge=1, le=20)
    question_types: List[QuestionType] = Field(
        default_factory=lambda: ["mcq"]
    )
    difficulty: Difficulty = "medium"
    language: Language = "ru"


class QuizQuestion(BaseModel):
    """Один валидированный вопрос в ответе."""

    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str = ""
    source_chunks: List[str] = Field(default_factory=list)


class QuizResponse(BaseModel):
    """Итог генерации теста: валидные вопросы + предупреждения."""

    questions: List[QuizQuestion]
    total: int
    warnings: List[str] = Field(default_factory=list)
    error: bool = False
