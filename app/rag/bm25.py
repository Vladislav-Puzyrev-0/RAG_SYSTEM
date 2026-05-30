"""
BM25-индекс на тех же чанках, что и FAISS.

Использует быстрый pure-Python BM25 (``bm25s``). Сохраняется в pickle
рядом с FAISS-индексом. Перестраивается при ingest / delete полностью
(для текущих объёмов корпуса это секунды).

Если ``bm25s`` не установлен, ``init_bm25_from_vectorstore`` вернёт None —
гибридный retriever автоматически деградирует до dense-only.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings
from app.utils.tokenize_ru import tokenize

logger = logging.getLogger("app.rag.bm25")

try:
    import bm25s
    _BM25S_AVAILABLE = True
except ImportError:
    _BM25S_AVAILABLE = False


def bm25_available() -> bool:
    """Установлен ли пакет ``bm25s``."""
    return _BM25S_AVAILABLE


class BM25Store:
    """Хранилище BM25-индекса и сериализуемого состояния."""

    def __init__(
        self,
        doc_ids: List[str],
        tokenized: List[List[str]],
        retriever,
        use_morph: bool = False,
    ) -> None:
        self.doc_ids = list(doc_ids)
        self.tokenized = [list(t) for t in tokenized]
        self.retriever = retriever
        self.use_morph = use_morph

    @property
    def ntotal(self) -> int:
        return len(self.doc_ids)

    # ── фабрики ───────────────────────────────────────────
    @classmethod
    def from_corpus(
        cls,
        doc_ids: List[str],
        texts: List[str],
        use_morph: bool = False,
    ) -> "BM25Store":
        if not _BM25S_AVAILABLE:
            raise RuntimeError("bm25s не установлен")
        tokenized = [tokenize(t, use_morph=use_morph) for t in texts]
        retriever = cls._build_retriever(tokenized)
        return cls(doc_ids, tokenized, retriever, use_morph=use_morph)

    @classmethod
    def load(cls, path: str) -> Optional["BM25Store"]:
        if not _BM25S_AVAILABLE:
            return None
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            retriever = cls._build_retriever(data["tokenized"])
            return cls(
                data["doc_ids"],
                data["tokenized"],
                retriever,
                use_morph=bool(data.get("use_morph", False)),
            )
        except Exception as e:
            logger.warning("Не удалось загрузить BM25 из %s: %s", path, e)
            return None

    # ── инкрементальные модификации (rebuild целиком) ─────
    def add(self, doc_ids: List[str], texts: List[str]) -> None:
        new_tok = [tokenize(t, use_morph=self.use_morph) for t in texts]
        self.doc_ids.extend(doc_ids)
        self.tokenized.extend(new_tok)
        self.retriever = self._build_retriever(self.tokenized)

    def remove(self, doc_ids_to_remove: List[str]) -> None:
        bad = set(doc_ids_to_remove)
        keep = [
            (did, tok)
            for did, tok in zip(self.doc_ids, self.tokenized)
            if did not in bad
        ]
        self.doc_ids = [d for d, _ in keep]
        self.tokenized = [t for _, t in keep]
        if self.tokenized:
            self.retriever = self._build_retriever(self.tokenized)
        else:
            self.retriever = None

    # ── поиск ─────────────────────────────────────────────
    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Вернуть ``[(doc_id, score), ...]``, отсортированный по убыванию."""
        if self.retriever is None or not self.doc_ids:
            return []
        q_tokens = tokenize(query, use_morph=self.use_morph)
        if not q_tokens:
            return []
        k_eff = min(k, len(self.doc_ids))
        indices, scores = self.retriever.retrieve([q_tokens], k=k_eff)
        out: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            i = int(idx)
            if 0 <= i < len(self.doc_ids):
                out.append((self.doc_ids[i], float(score)))
        return out

    # ── сохранение ────────────────────────────────────────
    def save(self, path: str) -> None:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_ids": self.doc_ids,
            "tokenized": self.tokenized,
            "use_morph": self.use_morph,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 сохранён в %s (%d чанков)", path, len(self.doc_ids))

    # ── внутреннее ────────────────────────────────────────
    @staticmethod
    def _build_retriever(tokenized: List[List[str]]):
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        return retriever


def init_bm25_from_vectorstore(vectorstore) -> Optional[BM25Store]:
    """Собрать BM25 из текущего docstore FAISS-индекса.

    Возвращает None если ``bm25s`` не установлен или ``vectorstore`` пуст.
    Результат НЕ сохраняется — это делает вызывающий через ``store.save``.
    """
    if not _BM25S_AVAILABLE:
        logger.warning(
            "bm25s не установлен — hybrid-поиск деградирует до dense-only"
        )
        return None
    if vectorstore is None or vectorstore.index.ntotal == 0:
        return None
    doc_ids: List[str] = []
    texts: List[str] = []
    for _idx, doc_id in vectorstore.index_to_docstore_id.items():
        doc = vectorstore.docstore.search(doc_id)
        if doc is None:
            continue
        doc_ids.append(doc_id)
        texts.append(doc.page_content)
    if not doc_ids:
        return None
    logger.info("Сборка BM25 по %d чанкам…", len(doc_ids))
    return BM25Store.from_corpus(doc_ids, texts)


def ensure_bm25(
    vectorstore, path: Optional[str] = None
) -> Optional[BM25Store]:
    """Гарантировать свежий BM25-индекс рядом с FAISS.

    Логика:

    1. Если ``bm25s`` не установлен → None.
    2. Попытаться загрузить ``path`` (по умолчанию ``settings.BM25_PATH``).
    3. Если файла нет или ``ntotal`` не совпадает с FAISS — пересобрать.
    4. Сохранить на диск.
    """
    if not _BM25S_AVAILABLE:
        return None
    path = path or settings.BM25_PATH
    expected = vectorstore.index.ntotal if vectorstore else 0
    loaded = BM25Store.load(path)
    if loaded is not None and loaded.ntotal == expected:
        logger.info("BM25 загружен из %s (%d чанков)", path, loaded.ntotal)
        return loaded
    if loaded is not None:
        logger.warning(
            "BM25 рассинхрон: %d ≠ FAISS %d — полная пересборка",
            loaded.ntotal, expected,
        )
    store = init_bm25_from_vectorstore(vectorstore)
    if store is not None:
        store.save(path)
    return store
