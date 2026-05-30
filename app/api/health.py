"""
Health & Status эндпоинты
"""
import torch
import faiss

from fastapi import APIRouter, Request

from app.models import StatusResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Проверка здоровья (не зависит от модели)."""
    return {"status": "healthy"}


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Статус системы."""
    rag_system = request.app.state.rag_system

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    doc_count = len(rag_system.hashes) if rag_system else 0

    has_hnsw = False
    if rag_system and rag_system.vectorstore:
        has_hnsw = isinstance(
            rag_system.vectorstore.index, faiss.IndexHNSW
        )

    return StatusResponse(
        status="ready",
        has_vectorstore=(
            rag_system.vectorstore is not None if rag_system else False
        ),
        model_device=device_name,
        loaded=rag_system is not None,
        documents_count=doc_count,
        hnsw_enabled=has_hnsw,
    )
