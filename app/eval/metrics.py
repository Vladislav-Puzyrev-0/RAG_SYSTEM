"""
Метрики качества retrieval (offline).

Все функции работают с одной парой ``(retrieved, golden_item)`` и
возвращают скаляр ``float``. Агрегация по датасету выполняется
вызывающим кодом (среднее по запросам).

Релевантность определяется на уровне ``source`` ИЛИ ``doc_id``:
если чанк имеет ``source`` из ``relevant_sources`` или ``doc_id`` из
``relevant_chunk_ids`` — он считается релевантным.
"""
from __future__ import annotations

import math
from typing import Sequence

from app.eval.dataset import GoldenItem
from app.models.common import RetrievedChunk


def _is_relevant(chunk: RetrievedChunk, item: GoldenItem) -> bool:
    if chunk.source and chunk.source in item.relevant_sources:
        return True
    if chunk.doc_id and chunk.doc_id in item.relevant_chunk_ids:
        return True
    return False


def _relevance_binary(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> list[int]:
    """0/1 для каждого чанка из топ-k (порядок сохраняется)."""
    return [1 if _is_relevant(c, item) else 0 for c in retrieved[:k]]


def precision_at_k(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if _is_relevant(c, item))
    return hits / len(top_k)


def recall_at_k(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> float:
    """Доля релевантных источников / чанков, попавших в top-k.

    Считаем уникальные совпадения по ``source`` и ``doc_id``, чтобы дубликаты
    чанков из одного документа не давали искусственного буста.
    """
    relevant_universe = set(item.relevant_sources) | set(item.relevant_chunk_ids)
    if not relevant_universe:
        return 0.0
    matched: set[str] = set()
    for c in retrieved[:k]:
        if c.source in item.relevant_sources:
            matched.add(c.source)
        if c.doc_id in item.relevant_chunk_ids:
            matched.add(c.doc_id)
    return len(matched) / len(relevant_universe)


def reciprocal_rank(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem
) -> float:
    """RR для одного запроса. MRR = mean RR по датасету."""
    for i, c in enumerate(retrieved, start=1):
        if _is_relevant(c, item):
            return 1.0 / i
    return 0.0


def hit_rate_at_k(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> float:
    return 1.0 if any(_is_relevant(c, item) for c in retrieved[:k]) else 0.0


def ndcg_at_k(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> float:
    """Нормированный DCG@k (бинарная релевантность)."""
    rels = _relevance_binary(retrieved, item, k)
    if not rels or sum(rels) == 0:
        return 0.0
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# Список метрик, поддерживаемый CLI-агрегатором.
METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate_at_k",
)


def compute_all(
    retrieved: Sequence[RetrievedChunk], item: GoldenItem, k: int
) -> dict:
    """Вернуть словарь со всеми метриками для одной пары."""
    return {
        "precision_at_k": precision_at_k(retrieved, item, k),
        "recall_at_k": recall_at_k(retrieved, item, k),
        "mrr": reciprocal_rank(retrieved, item),
        "ndcg_at_k": ndcg_at_k(retrieved, item, k),
        "hit_rate_at_k": hit_rate_at_k(retrieved, item, k),
    }
