"""
Retrieval + LLM-генерация ответа с адаптивным расширением контекста.

В v3 поиск выполняет :class:`app.rag.retriever.HybridRetriever`
(BM25 + dense + RRF). Здесь остаются:

* функции расширения контекста (``_expand_context``, ``_limit_expanded_context``)
  и построения контекстной строки для LLM;
* основной асинхронный пайплайн ``retrieve_and_generate``, который дополнительно
  ходит в SQLite-кэш ответов.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import List, Optional

from app.config import settings
from app.metrics import CACHE_HITS, CACHE_MISSES, LLM_QUERY_DURATION
from app.rag.cache import RAGCache, answer_cache_key
from app.rag.prompts import AST_TAG, END_TAG, ask_prompt, is_no_answer

logger = logging.getLogger("app.rag.retrieval")


def _get_chunk_index(doc) -> int:
    """Получить индекс чанка из метаданных (0 для старых индексов)."""
    return doc.metadata.get("chunk_index", 0)


def _expand_context(vectorstore, source_docs, window_size=3, question=""):
    """Расширить контекст соседними чанками каждого источника.

    Для старых индексов без ``chunk_index`` делает вторичный
    similarity-search как fallback.
    """
    sources = set()
    for doc in source_docs:
        sources.add(doc.metadata.get("source", "unknown"))

    docstore = vectorstore.docstore
    index_to_docstore_id = vectorstore.index_to_docstore_id

    all_docs = []
    for _idx, doc_id in index_to_docstore_id.items():
        d = docstore.search(doc_id)
        if d:
            all_docs.append(d)

    expanded = []
    for source in sources:
        source_chunks = [
            d for d in all_docs if d.metadata.get("source") == source
        ]
        if not source_chunks:
            continue
        source_chunks.sort(key=_get_chunk_index)
        min_idx = _get_chunk_index(source_chunks[0])
        max_idx = _get_chunk_index(source_chunks[-1])

        # Fallback: для старого индекса (нет chunk_index) — повторный поиск.
        has_chunk_idx = any(
            d.metadata.get("chunk_index") is not None for d in source_chunks
        )
        if not has_chunk_idx:
            search_query = question or source_chunks[0].page_content[:200]
            extended = vectorstore.similarity_search(
                search_query, k=min(len(source_chunks) * 3, 50)
            )
            for doc in extended:
                if (
                    doc.metadata.get("source") == source
                    and doc not in expanded
                ):
                    expanded.append(doc)
            continue

        expand_start = max(0, min_idx - window_size)
        expand_end = max_idx + window_size

        is_large = source_chunks[0].metadata.get("is_large_doc", False)
        if is_large:
            expand_start = max(0, min_idx - window_size * 2)
            expand_end = max_idx + window_size * 2

        for doc in source_chunks:
            idx = _get_chunk_index(doc)
            if expand_start <= idx <= expand_end and doc not in expanded:
                expanded.append(doc)

    grouped = defaultdict(list)
    for doc in expanded:
        grouped[doc.metadata.get("source", "unknown")].append(doc)

    result = []
    for _src, docs in grouped.items():
        docs.sort(key=_get_chunk_index)
        result.extend(docs)

    MAX_CHUNKS = 12
    if len(result) > MAX_CHUNKS:
        step = len(result) / MAX_CHUNKS
        result = [result[int(i * step)] for i in range(MAX_CHUNKS)]
    return result


def _build_context(docs) -> str:
    """Собрать связную контекстную строку из чанков."""
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "?")
        idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[{src} #чанк {idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _limit_expanded_context(docs, max_total: int = 12, max_per_source: int = 4):
    """Постпроцессор: ограничить число чанков общим и на источник."""
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source", "unknown")].append(doc)
    trimmed = []
    for _src, src_docs in grouped.items():
        src_docs.sort(key=_get_chunk_index)
        trimmed.extend(src_docs[:max_per_source])
    if len(trimmed) <= max_total:
        return trimmed
    step = len(trimmed) / max_total
    return [trimmed[int(i * step)] for i in range(max_total)]


def _strip_llm_tags(text: str) -> str:
    """Отрезать ChatML-теги от ответа модели."""
    if AST_TAG + "\n" in text:
        text = text.split(AST_TAG + "\n")[-1]
    return text.replace(END_TAG, "").strip()


async def retrieve_and_generate(
    question: str,
    retriever,
    llm,
    cache: Optional[RAGCache] = None,
    *,
    top_k: int = 3,
    temperature: float = 0.3,
    mode: str = "hybrid",
    ef_search: Optional[int] = None,
    expand_context: bool = True,
    context_window: int = 3,
    llm_model: str = "",
    emb_model: str = "",
) -> dict:
    """Поиск (HybridRetriever) → контекст → LLM-генерация (+ SQLite-кэш)."""
    if retriever is None or retriever.vectorstore is None:
        return {
            "answer": "Нет документов. Загрузите через /upload",
            "sources": [],
            "context_docs": 0,
            "error": True,
        }

    # ── кэш ответов ──────────────────────────────────────
    cache_key: Optional[str] = None
    if cache is not None:
        cache_key = answer_cache_key(
            llm_model, emb_model, question,
            top_k, temperature, mode, ef_search,
        )
        cached = cache.get_answer(cache_key)
        if cached is not None:
            CACHE_HITS.labels(kind="answer").inc()
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "context_docs": len(cached["sources"]),
                "error": False,
                "cache_hit": True,
            }
        CACHE_MISSES.labels(kind="answer").inc()

    start = time.time()
    try:
        top_k_eff = max(top_k, 8) if expand_context else top_k
        top_k_eff = max(1, min(top_k_eff, settings.RETRIEVAL_MAX_TOP_K))

        retr_t0 = time.time()
        docs = await retriever.search_documents(
            question, top_k=top_k_eff, mode=mode, ef_search=ef_search
        )
        retrieval_ms = (time.time() - retr_t0) * 1000

        if not docs:
            return {
                "answer": "В документах нет информации по этому вопросу.",
                "sources": [],
                "context_docs": 0,
                "error": False,
                "retrieval_ms": retrieval_ms,
                "total_ms": (time.time() - start) * 1000,
                "cache_hit": False,
            }

        if expand_context:
            docs = await asyncio.to_thread(
                _expand_context,
                retriever.vectorstore,
                docs,
                context_window,
                question,
            )

        context = _build_context(docs)
        prompt = ask_prompt(question, context)

        llm_t0 = time.time()
        raw_answer = await asyncio.to_thread(
            llm.generate, prompt, settings.LLM_MAX_NEW_TOKENS, temperature
        )
        llm_ms = (time.time() - llm_t0) * 1000
        answer = _strip_llm_tags(raw_answer)

        dur = time.time() - start
        LLM_QUERY_DURATION.labels(status="success").observe(dur)
        if is_no_answer(answer):
            logger.info("LLM сообщила «нет ответа в контексте»")

        sources = sorted({d.metadata.get("source", "?") for d in docs})
        if cache is not None and cache_key:
            cache.put_answer(
                cache_key, top_k, temperature, answer, sources
            )

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": len(docs),
            "error": False,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": dur * 1000,
            "cache_hit": False,
        }
    except Exception as e:  # noqa: BLE001
        dur = time.time() - start
        LLM_QUERY_DURATION.labels(status="error").observe(dur)
        logger.error("LLM query error: %s (%.2fs)", e, dur)
        return {
            "answer": f"Ошибка: {e}",
            "sources": [],
            "context_docs": 0,
            "error": True,
        }
