"""
Загрузка golden-датасета для оффлайн-оценки retrieval.

Формат JSONL — по одному JSON-объекту на строку::

    {"query": "...", "relevant_sources": ["doc.pdf"], "relevant_chunk_ids": []}

Поля:

* ``query`` (str, обязательно) — запрос к RAG.
* ``relevant_sources`` (list[str], опционально) — имена файлов-источников,
  которые считаются релевантными.
* ``relevant_chunk_ids`` (list[str], опционально) — конкретные docstore-id
  чанков, если хочется судить с точностью до чанка.

Пустые строки и строки, начинающиеся с ``#``, пропускаются.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger("app.eval.dataset")


@dataclass
class GoldenItem:
    """Одна golden-запись датасета."""

    query: str
    relevant_sources: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)


def load_dataset(path: Path) -> List[GoldenItem]:
    """Прочитать JSONL-файл, валидируя каждую строку."""
    items: List[GoldenItem] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.warning("Строка %d: невалидный JSON (%s)", line_no, e)
            continue
        if not isinstance(obj, dict):
            logger.warning("Строка %d: ожидался объект, получен %s",
                           line_no, type(obj).__name__)
            continue
        query = obj.get("query")
        if not isinstance(query, str) or not query.strip():
            logger.warning("Строка %d: пустой или невалидный 'query'", line_no)
            continue
        items.append(
            GoldenItem(
                query=query.strip(),
                relevant_sources=list(obj.get("relevant_sources") or []),
                relevant_chunk_ids=list(obj.get("relevant_chunk_ids") or []),
            )
        )
    return items
