"""
Document ingestion: загрузка, чанкинг, дедупликация
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# ── Deduplication registry ──────────────────────────────
DEDUP_FILE = os.path.join(settings.VECTORSTORE_PATH, "document_hashes.json")


def load_hashes() -> dict:
    """Загрузить словарь {filename: sha256} из файла."""
    if os.path.exists(DEDUP_FILE):
        with open(DEDUP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict):
    """Сохранить словарь хешей."""
    Path(settings.VECTORSTORE_PATH).mkdir(exist_ok=True)
    with open(DEDUP_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


def file_sha256(path: str) -> str:
    """SHA-256 хеш содержимого файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_chunk_documents(file_paths: List[str]) -> List[Document]:
    """
    Загрузить и разбить на чанки документы из списка файлов.
    Возвращает список чанков с метаданными.
    """
    documents = []
    for file_path in file_paths:
        try:
            fname = os.path.basename(file_path)
            logger.info(f"Загрузка: {fname}")
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in (".doc", ".docx"):
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"Неподдерживаемый формат: {fname}")
                continue

            docs = loader.load()
            # Добавляем source в метаданные
            for doc in docs:
                doc.metadata["source"] = fname
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")

    if not documents:
        raise ValueError("Не удалось загрузить документы")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Добавляем порядковый номер чанка и метаданные документа
    total_docs = len(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        chunk.metadata["doc_size"] = total_docs
        chunk.metadata["is_large_doc"] = total_docs > 50  # >50 страниц = большой

    logger.info(f"✓ Создано {len(chunks)} чанков из {len(file_paths)} файл(ов)")

    return chunks
