"""
QwenRAGSystem — фасад над всеми RAG-компонентами (embeddings, FAISS,
BM25, кэш, LLM). Обеспечивает RW-pattern для конкуррентных запросов:
писатели (ingest/delete) сериализованы общим ``_write_lock``, читатели
(``ask_question`` / стрим) ходят без локов и видят текущую ссылку на
``self.vectorstore``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import AsyncIterator, List, Optional, Tuple

from app.config import settings
from app.metrics import DOCUMENTS_COUNT, INGESTION_CHUNKS
from app.rag.bm25 import BM25Store, bm25_available, ensure_bm25
from app.rag.cache import RAGCache, answer_cache_key, init_cache
from app.rag.embeddings import init_embeddings
from app.rag.ingestion import (
    file_sha256,
    load_and_chunk_documents,
    load_hashes,
    save_hashes,
)
from app.rag.llm import init_llm
from app.rag.prompts import ask_prompt
from app.rag.retrieval import _build_context, _expand_context, retrieve_and_generate
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    save_vectorstore,
)

logger = logging.getLogger("app.rag.system")


class QwenRAGSystem:
    """Координатор всех RAG-компонентов (embeddings, FAISS, BM25, cache, LLM)."""

    def __init__(self) -> None:
        logger.info("Инициализация QwenRAGSystem v3.0…")

        # Эмбеддинги и LLM — самые тяжёлые шаги.
        self.embeddings = init_embeddings()
        self.llm, self.device_human = init_llm()

        # Векторстор + RW-lock.
        self.vectorstore = None
        self._write_lock = asyncio.Lock()

        # Дедупликация.
        self.hashes = load_hashes()
        logger.info("Дедупликация: %d известных файлов", len(self.hashes))

        # FAISS.
        self._load_vectorstore()

        # BM25 — собирается из docstore текущего FAISS (None если bm25s нет).
        self.bm25: Optional[BM25Store] = ensure_bm25(self.vectorstore)

        # SQLite-кэш ответов / эмбеддингов.
        self.cache: Optional[RAGCache] = init_cache()

        # Hybrid-ретривер поверх FAISS + BM25.
        self.retriever = HybridRetriever(
            self.vectorstore, self.embeddings, self.bm25
        )

        logger.info(
            "QwenRAGSystem v3.0 готова (bm25=%s, cache=%s, metric=%s)",
            "yes" if self.bm25 is not None else "no",
            "yes" if self.cache is not None else "no",
            settings.FAISS_METRIC,
        )

    # ── helpers ───────────────────────────────────────────
    def _load_vectorstore(self) -> None:
        self.vectorstore = load_vectorstore(self.embeddings)

    def _save_vectorstore(self) -> None:
        save_vectorstore(self.vectorstore)

    def _rebuild_bm25_and_retriever(self) -> None:
        """Полная пересборка BM25 + обновление ретривера.

        Используется после ingest/delete. На текущих корпусах (~тысячи чанков)
        полный rebuild занимает доли секунды.
        """
        self.bm25 = ensure_bm25(self.vectorstore)
        self.retriever = HybridRetriever(
            self.vectorstore, self.embeddings, self.bm25
        )

    # ── Document ingestion ───────────────────────────────
    async def load_documents(self, file_paths: List[str]) -> dict:
        """Загрузка документов с дедупликацией."""
        async with self._write_lock:
            return await self._load_documents_unsafe(file_paths)

    async def _load_documents_unsafe(self, file_paths: List[str]) -> dict:
        new_files: List[str] = []
        skipped: List[str] = []

        for file_path in file_paths:
            fname = os.path.basename(file_path)
            fhash = file_sha256(file_path)

            if settings.DEDUP_ENABLED and fname in self.hashes:
                if self.hashes[fname] == fhash:
                    logger.info("Пропуск дубликата: %s", fname)
                    skipped.append(fname)
                    os.remove(file_path)
                    continue
                logger.info("Файл изменён, перезагрузка: %s", fname)
                self._remove_document_from_store(fname)
                del self.hashes[fname]

            new_files.append(file_path)
            self.hashes[fname] = fhash

        if not new_files and not skipped:
            raise ValueError("Нет документов для обработки")

        chunks = await asyncio.to_thread(load_and_chunk_documents, new_files)

        if self.vectorstore is None:
            self.vectorstore = await asyncio.to_thread(
                create_vectorstore, chunks, self.embeddings
            )
        else:
            await asyncio.to_thread(self.vectorstore.add_documents, chunks)

        save_hashes(self.hashes)
        await asyncio.to_thread(self._save_vectorstore)

        # BM25 строится из обновлённого docstore.
        await asyncio.to_thread(self._rebuild_bm25_and_retriever)

        DOCUMENTS_COUNT.set(len(self.hashes))
        INGESTION_CHUNKS.labels(status="success").inc(len(chunks))

        logger.info(
            "Документы загружены: %d файл(ов), %d пропущено, %d чанков",
            len(new_files), len(skipped), len(chunks),
        )
        return {
            "new_files": len(new_files),
            "skipped": len(skipped),
            "total_chunks": len(chunks),
        }

    def _remove_document_from_store(self, filename: str) -> None:
        """Удалить все чанки документа из FAISS (через rebuild индекса)."""
        if self.vectorstore is None:
            return
        docstore = self.vectorstore.docstore
        index_to_docstore_id = self.vectorstore.index_to_docstore_id

        to_remove: List[str] = []
        to_keep: List[Tuple[int, str]] = []
        for idx, doc_id in index_to_docstore_id.items():
            doc = docstore.search(doc_id)
            if doc and doc.metadata.get("source") == filename:
                to_remove.append(doc_id)
            else:
                to_keep.append((idx, doc_id))

        if not to_remove:
            return
        logger.info(
            "Удаление %d чанков файла %s из векторстора",
            len(to_remove), filename,
        )

        remaining_docs = []
        for _idx, doc_id in to_keep:
            doc = docstore.search(doc_id)
            if doc:
                remaining_docs.append(doc)

        if remaining_docs:
            self.vectorstore = create_vectorstore(
                remaining_docs, self.embeddings
            )
        else:
            self.vectorstore = None

    # ── Delete document ─────────────────────────────────
    async def delete_document(self, filename: str) -> dict:
        async with self._write_lock:
            return await self._delete_document_unsafe(filename)

    async def _delete_document_unsafe(self, filename: str) -> dict:
        file_path = os.path.join(settings.DOCS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Файл удалён: %s", filename)

        await asyncio.to_thread(self._remove_document_from_store, filename)
        self.hashes.pop(filename, None)
        save_hashes(self.hashes)
        await asyncio.to_thread(self._save_vectorstore)
        await asyncio.to_thread(self._rebuild_bm25_and_retriever)

        DOCUMENTS_COUNT.set(len(self.hashes))
        return {
            "message": f"Документ '{filename}' удалён",
            "filename": filename,
        }

    # ── Query (sync) ─────────────────────────────────────
    async def ask_question(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.3,
        expand_context: bool = True,
        context_window: int = 3,
        mode: str = "hybrid",
        ef_search: Optional[int] = None,
    ) -> dict:
        """Запрос с одним LLM-ответом (используется /ask)."""
        return await retrieve_and_generate(
            question,
            self.retriever,
            self.llm,
            cache=self.cache,
            top_k=top_k,
            temperature=temperature,
            mode=mode,
            ef_search=ef_search,
            expand_context=expand_context,
            context_window=context_window,
            llm_model=settings.LLM_MODEL,
            emb_model=settings.EMBEDDING_MODEL,
        )

    # ── Query (stream) ───────────────────────────────────
    async def ask_question_stream(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.3,
        expand_context: bool = True,
        context_window: int = 3,
        mode: str = "hybrid",
        ef_search: Optional[int] = None,
    ) -> AsyncIterator[Tuple[str, dict]]:
        """Стриминг ответа LLM.

        Yields кортежи ``(event_type, payload)`` где ``event_type`` —
        одно из ``token`` / ``done`` / ``error``. Кэш ответов
        НЕ используется для стрима (текстовый delta нельзя ретропроиграть
        смыслом одной строки).
        """
        if self.retriever is None or self.vectorstore is None:
            yield (
                "error",
                {"message": "Нет документов. Загрузите через /upload"},
            )
            return

        start = time.time()
        try:
            top_k_eff = max(top_k, 8) if expand_context else top_k
            top_k_eff = max(
                1, min(top_k_eff, settings.RETRIEVAL_MAX_TOP_K)
            )
            docs = await self.retriever.search_documents(
                question,
                top_k=top_k_eff,
                mode=mode,
                ef_search=ef_search,
            )
            if not docs:
                yield (
                    "done",
                    {
                        "sources": [],
                        "context_docs": 0,
                        "total_ms": (time.time() - start) * 1000,
                    },
                )
                return

            if expand_context:
                docs = await asyncio.to_thread(
                    _expand_context,
                    self.vectorstore,
                    docs,
                    context_window,
                    question,
                )

            context = _build_context(docs)
            prompt = ask_prompt(question, context)

            llm_t0 = time.time()
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def _producer() -> None:
                try:
                    for piece in self.llm.stream(
                        prompt,
                        settings.LLM_MAX_NEW_TOKENS,
                        temperature,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, piece)
                except Exception as e:  # noqa: BLE001
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("__err__", str(e))
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                piece = await queue.get()
                if piece is sentinel:
                    break
                if isinstance(piece, tuple) and piece and piece[0] == "__err__":
                    yield ("error", {"message": piece[1]})
                    return
                yield ("token", {"text": piece})

            sources = sorted(
                {d.metadata.get("source", "?") for d in docs}
            )
            yield (
                "done",
                {
                    "sources": sources,
                    "context_docs": len(docs),
                    "llm_ms": (time.time() - llm_t0) * 1000,
                    "total_ms": (time.time() - start) * 1000,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("ask_question_stream failed")
            yield ("error", {"message": str(e)})

    # ── Status helpers ───────────────────────────────────
    def status_extras(self) -> dict:
        """Расширенные поля для /status (Phase 2/3)."""
        ntotal = (
            self.vectorstore.index.ntotal
            if self.vectorstore is not None
            else 0
        )
        cache_size = 0
        if (
            self.cache is not None
            and os.path.exists(self.cache.path)
        ):
            try:
                cache_size = os.path.getsize(self.cache.path)
            except OSError:
                cache_size = 0
        ef_search = None
        try:
            ef_search = int(self.vectorstore.index.hnsw.efSearch)
        except Exception:  # noqa: BLE001
            ef_search = None
        return {
            "chunks_count": ntotal,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "ef_search": ef_search,
            "hybrid_enabled": bool(self.retriever and self.retriever.hybrid_enabled),
            "embedding_metric": settings.FAISS_METRIC,
            "cache_enabled": self.cache is not None,
            "cache_size_bytes": cache_size,
            "bm25_loaded": self.bm25 is not None and bm25_available(),
            "device_human": self.device_human,
        }

    # ── Document list ───────────────────────────────────
    def get_document_list(self) -> List[dict]:
        docs = []
        for fname, fhash in self.hashes.items():
            fpath = os.path.join(settings.DOCS_DIR, fname)
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            docs.append(
                {"filename": fname, "sha256": fhash, "size_bytes": size}
            )
        return docs
