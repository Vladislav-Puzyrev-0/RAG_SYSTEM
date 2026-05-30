"""
Documents эндпоинты: список и удаление.
"""
import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_rag_system
from app.models import DeleteResponse, DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(rag_system: Any = Depends(get_rag_system)):
    """Список загруженных документов."""
    return rag_system.get_document_list()


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document(
    filename: str,
    rag_system: Any = Depends(get_rag_system),
):
    """Удалить документ из системы."""
    if filename not in rag_system.hashes:
        raise HTTPException(
            status_code=404, detail=f"Документ '{filename}' не найден"
        )

    result = await rag_system.delete_document(filename)
    return DeleteResponse(**result)
