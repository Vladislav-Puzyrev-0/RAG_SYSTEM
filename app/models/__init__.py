"""Пакет Pydantic-моделей.

Реэкспортирует имена в стиле v2 для обратной совместимости.
Старый код вида ``from app.models import QuestionRequest`` продолжит работать.
"""
from app.models.ask import AnswerResponse, QuestionRequest, Source
from app.models.common import ErrorResponse, RetrievedChunk, StatusResponse
from app.models.documents import (
    DeleteResponse,
    DocumentInfo,
    UploadResponse,
)
from app.models.quiz import (
    QuizQuestion,
    QuizRequest,
    QuizResponse,
)

__all__ = [
    # ask
    "AnswerResponse",
    "QuestionRequest",
    "Source",
    # common
    "ErrorResponse",
    "RetrievedChunk",
    "StatusResponse",
    # documents
    "DeleteResponse",
    "DocumentInfo",
    "UploadResponse",
    # quiz
    "QuizQuestion",
    "QuizRequest",
    "QuizResponse",
]
