"""
Сервис эмбеддингов поверх ``sentence-transformers``.

В отличие от обёртки LangChain (``HuggingFaceEmbeddings``), модель здесь
используется напрямую: можно явно задать ``batch_size``, включить L2-нормализацию
и не платить за лишние конверсии numpy↔list.

Класс наследуется от ``langchain_core.embeddings.Embeddings`` — это значит,
что объект можно без изменений передавать в ``langchain_community.FAISS`` как
``embedding_function``.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger("app.rag.embeddings")


class EmbeddingService(Embeddings):
    """Эмбеддинги через SentenceTransformer с явным контролем batch и L2-norm."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        local_path = snapshot_download(model_name, local_files_only=True)
        self._model = SentenceTransformer(local_path, device=device)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "Эмбеддинги: %s (device=%s, dim=%d, normalize=%s, batch=%d)",
            model_name,
            device,
            self.dim,
            normalize,
            batch_size,
        )

    # ── интерфейс langchain Embeddings ─────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Векторизовать список документов батчами."""
        if not texts:
            return []
        vecs = self._encode(list(texts), batch_size=self.batch_size)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Векторизовать один запрос."""
        vec = self._encode([text], batch_size=1)[0]
        return vec.tolist()

    # ── numpy-вариант для прямой работы с FAISS ────────────
    def embed_documents_np(self, texts: List[str]) -> np.ndarray:
        """Тот же ``embed_documents``, но возвращает ``np.ndarray`` (float32)."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self._encode(list(texts), batch_size=self.batch_size)

    def embed_query_np(self, text: str) -> np.ndarray:
        """Один вектор-запрос как numpy."""
        return self._encode([text], batch_size=1)[0]

    # ── внутренние ─────────────────────────────────────────
    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype("float32", copy=False)


def init_embeddings() -> EmbeddingService:
    """Собрать ``EmbeddingService`` из ``settings`` (.env)."""
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        normalize=settings.EMBEDDING_NORMALIZE,
    )
