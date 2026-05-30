"""
Upload документов с фоновой обработкой (BackgroundTasks)
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from app.config import settings
from app.deps import get_rag_system
from app.models import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Хранилище статусов фоновых задач ──────────────────────
_upload_statuses: dict[str, dict] = {}


@dataclass
class UploadJob:
    file_paths: List[str]
    status: str = "processing"
    result: Optional[dict] = None
    error: Optional[str] = None


async def _process_upload(job: UploadJob, rag_system):
    """Фоновая обработка загруженных файлов."""
    try:
        result = await rag_system.load_documents(job.file_paths)
        job.result = result
        job.status = "done"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
        logger.error(f"Ошибка фоновой загрузки: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    async_mode: bool = False,
    rag_system: Any = Depends(get_rag_system),
):
    """
    Загрузить документы с дедупликацией.

    ``async_mode=true`` — фоновая обработка, возвращаем сразу 202.
    ``async_mode=false`` (по умолчанию) — ждём завершения.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Нет загруженных файлов")

    file_paths = []
    for file in files:
        file_path = os.path.join(settings.DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(file_path)
        logger.info(f"Файл сохранён: {file.filename}")

    if async_mode:
        # Фоновый режим — сразу возвращаем 202
        job_id = f"{os.getpid()}_{len(_upload_statuses)}"
        job = UploadJob(file_paths=file_paths)
        _upload_statuses[job_id] = job
        background_tasks.add_task(_process_upload, job, rag_system)

        return UploadResponse(
            message=f"Задача {job_id} запущена в фоне. Статус: GET /upload/status/{job_id}",
            new_files=len(file_paths),
            skipped_duplicates=0,
            chunks_created=0,
            error=False,
        )
    else:
        # Синхронный режим — ждём (обратная совместимость)
        try:
            result = await rag_system.load_documents(file_paths)
            msg_parts = [f"✓ Загружено {result['new_files']} файл(ов)"]
            if result["skipped"] > 0:
                msg_parts.append(f"пропущено {result['skipped']} дубликатов")
            msg_parts.append(f"создано {result['total_chunks']} чанков")

            return UploadResponse(
                message=", ".join(msg_parts),
                new_files=result["new_files"],
                skipped_duplicates=result["skipped"],
                chunks_created=result["total_chunks"],
                error=False,
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке: {e}")
            return UploadResponse(
                message=f"Ошибка: {e}",
                new_files=0,
                skipped_duplicates=0,
                chunks_created=0,
                error=True,
            )


@router.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    """Проверить статус фоновой загрузки."""
    if job_id not in _upload_statuses:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    job = _upload_statuses[job_id]
    resp = {"job_id": job_id, "status": job.status}

    if job.result:
        resp["result"] = job.result
    if job.error:
        resp["error"] = job.error

    return resp
