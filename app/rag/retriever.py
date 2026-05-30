"""
Гибридный retrieval: dense (FAISS-HNSW) + BM25 + Reciprocal Rank Fusion.

Возвращает список :class:`app.models.common.RetrievedChunk` со всеми
рассчитанными скорами, что позволяет UI и eval-пайплайну отдельно
анализировать вклад dense / bm25 каналов.

Режимы:

* ``dense`` — только FAISS.
* ``bm25``  — только BM25.
* ``hybrid`` — оба канала параллельно (``asyncio.gather``),
  фьюжн через RRF (по умолчанию) либо линейная комбинация по
  ``alpha`` на min-max нормированных скорах.

При ``mode='hybrid'``, если BM25 не доступен, retriever автоматически
деградирует до dense-only.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from app.config import settings
from app.metrics import RETRIEVAL_DURATION
from app.models.common import RetrievedChunk
from app.rag.bm25 import BM25Store
from app.rag.vectorstore import set_ef_search

logger = logging.getLogger("app.rag.retriever")


class HybridRetriever:
    """Объединяющий dense + BM25 поиск через RRF (или weighted fusion)."""

    def __init__(
        self,
        vectorstore,
        embeddings,
        bm25: Optional[BM25Store],
    ) -> None:
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.bm25 = bm25

    @property
    def hybrid_enabled(self) -> bool:
        return self.bm25 is not None and self.bm25.ntotal > 0

    # ── публичный API ─────────────────────────────────────
    async def search(
        self,
        query: str,
        top_k: int,
        mode: str = "hybrid",
        alpha: Optional[float] = None,
        ef_search: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Найти top_k наиболее релевантных чанков по выбранному режиму."""
        if self.vectorstore is None or self.vectorstore.index.ntotal == 0:
            return []

        effective_mode = mode
        if effective_mode == "hybrid" and not self.hybrid_enabled:
            effective_mode = "dense"

        if ef_search:
            set_ef_search(self.vectorstore, ef_search)

        internal_k = max(
            top_k * settings.HYBRID_TOP_K_INTERNAL_FACTOR, 20
        )
        start = time.time()
        try:
            if effective_mode == "dense":
                dense = await asyncio.to_thread(
                    self._dense_search, query, internal_k
                )
                fused = self._dense_only_chunks(dense, top_k)
            elif effective_mode == "bm25":
                bm = await asyncio.to_thread(self._bm25_search, query, internal_k)
                fused = self._bm25_only_chunks(bm, top_k)
            else:
                dense_task = asyncio.to_thread(self._dense_search, query, internal_k)
                bm25_task = asyncio.to_thread(self._bm25_search, query, internal_k)
                dense, bm = await asyncio.gather(dense_task, bm25_task)
                fused = self._fuse(dense, bm, top_k, alpha)
        finally:
            RETRIEVAL_DURATION.labels(mode=effective_mode).observe(
                time.time() - start
            )

        if logger.isEnabledFor(logging.DEBUG):
            for c in fused:
                logger.debug(
                    "rank=%d rrf=%s dense=%s bm25=%s source=%s",
                    c.rank,
                    c.score_rrf,
                    c.score_dense,
                    c.score_bm25,
                    c.source,
                )
        return fused

    async def search_documents(
        self,
        query: str,
        top_k: int,
        mode: str = "hybrid",
        ef_search: Optional[int] = None,
    ) -> List[Document]:
        """Совместимая выдача в виде langchain ``Document``-ов из docstore.

        Сохраняет ВСЕ оригинальные метаданные (``chunk_index``, ``source``,
        ``is_large_doc``, ...), что нужно ``_expand_context``.
        """
        chunks = await self.search(
            query, top_k=top_k, mode=mode, ef_search=ef_search
        )
        docs: List[Document] = []
        for c in chunks:
            d = self.vectorstore.docstore.search(c.doc_id)
            if d is not None:
                docs.append(d)
        return docs

    # ── нижний слой ───────────────────────────────────────
    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        """Прямой вызов faiss.search с маппингом в docstore."""
        q = np.asarray(
            self.embeddings.embed_query_np(query), dtype="float32"
        ).reshape(1, -1)
        D, I = self.vectorstore.index.search(q, top_k)
        metric_is_l2 = (
            self.vectorstore.index.metric_type == faiss.METRIC_L2
        )
        results: List[Dict] = []
        for rank, (faiss_idx, raw_score) in enumerate(
            zip(I[0], D[0]), start=1
        ):
            i = int(faiss_idx)
            if i < 0:
                continue
            doc_id = self.vectorstore.index_to_docstore_id.get(i)
            if doc_id is None:
                continue
            doc = self.vectorstore.docstore.search(doc_id)
            if doc is None:
                continue
            # «Больше = лучше» унифицировано для UI.
            score = (
                -float(raw_score) if metric_is_l2 else float(raw_score)
            )
            results.append(
                {
                    "doc_id": doc_id,
                    "doc": doc,
                    "rank": rank,
                    "score": score,
                }
            )
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if self.bm25 is None:
            return []
        pairs = self.bm25.search(query, k=top_k)
        out: List[Dict] = []
        for rank, (doc_id, score) in enumerate(pairs, start=1):
            doc = self.vectorstore.docstore.search(doc_id)
            if doc is None:
                continue
            out.append(
                {
                    "doc_id": doc_id,
                    "doc": doc,
                    "rank": rank,
                    "score": float(score),
                }
            )
        return out

    # ── фьюжн ─────────────────────────────────────────────
    def _fuse(
        self,
        dense: List[Dict],
        bm25: List[Dict],
        top_k: int,
        alpha: Optional[float],
    ) -> List[RetrievedChunk]:
        by_id: Dict[str, Dict] = {}
        for r in dense:
            entry = by_id.setdefault(r["doc_id"], {"doc": r["doc"]})
            entry["dense_rank"] = r["rank"]
            entry["dense_score"] = r["score"]
        for r in bm25:
            entry = by_id.setdefault(r["doc_id"], {"doc": r["doc"]})
            entry["bm25_rank"] = r["rank"]
            entry["bm25_score"] = r["score"]

        if alpha is None:
            k = settings.HYBRID_RRF_K
            for entry in by_id.values():
                rrf = 0.0
                if "dense_rank" in entry:
                    rrf += 1.0 / (k + entry["dense_rank"])
                if "bm25_rank" in entry:
                    rrf += 1.0 / (k + entry["bm25_rank"])
                entry["score_fused"] = rrf
        else:
            d_norm = _min_max(
                {did: e.get("dense_score") for did, e in by_id.items()}
            )
            b_norm = _min_max(
                {did: e.get("bm25_score") for did, e in by_id.items()}
            )
            for did, entry in by_id.items():
                d = d_norm.get(did, 0.0)
                b = b_norm.get(did, 0.0)
                entry["score_fused"] = (
                    alpha * d + (1.0 - alpha) * b
                )

        ordered = sorted(
            by_id.items(), key=lambda kv: -kv[1].get("score_fused", 0.0)
        )[:top_k]
        chunks: List[RetrievedChunk] = []
        for new_rank, (doc_id, entry) in enumerate(ordered, start=1):
            doc = entry["doc"]
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    source=doc.metadata.get("source", "?"),
                    text=doc.page_content,
                    rank=new_rank,
                    score_dense=entry.get("dense_score"),
                    score_bm25=entry.get("bm25_score"),
                    score_rrf=entry.get("score_fused"),
                )
            )
        return chunks

    def _dense_only_chunks(
        self, results: List[Dict], top_k: int
    ) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for new_rank, r in enumerate(results[:top_k], start=1):
            doc = r["doc"]
            out.append(
                RetrievedChunk(
                    doc_id=r["doc_id"],
                    source=doc.metadata.get("source", "?"),
                    text=doc.page_content,
                    rank=new_rank,
                    score_dense=r["score"],
                )
            )
        return out

    def _bm25_only_chunks(
        self, results: List[Dict], top_k: int
    ) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for new_rank, r in enumerate(results[:top_k], start=1):
            doc = r["doc"]
            out.append(
                RetrievedChunk(
                    doc_id=r["doc_id"],
                    source=doc.metadata.get("source", "?"),
                    text=doc.page_content,
                    rank=new_rank,
                    score_bm25=r["score"],
                )
            )
        return out


def _min_max(scores: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Min-max нормализация словаря скоров (None-значения отбрасываются)."""
    vals = [v for v in scores.values() if v is not None]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k, v in scores.items() if v is not None}
    return {
        k: (v - lo) / (hi - lo)
        for k, v in scores.items()
        if v is not None
    }
