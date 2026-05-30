"""Модели для эндпоинтов работы с документами."""
from __future__ import annotations

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    """Один загруженный документ."""

    filename: str
    sha256: str
    size_bytes: int


class UploadResponse(BaseModel):
    """Результат пакетной загрузки документов."""

    message: str
    new_files: int
    skipped_duplicates: int
    chunks_created: int
    error: bool = False


class DeleteResponse(BaseModel):
    """Результат удаления документа."""

    message: str
    filename: str
