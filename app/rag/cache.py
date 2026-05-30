"""
SQLite-кэш для RAG-системы.

Содержит две таблицы:

* ``embeddings(text_sha256 PRIMARY KEY, vector BLOB, model, created_at)`` —
  кэш векторов запросов.
* ``answers(query_sha256 PRIMARY KEY, top_k, temperature, answer,
  sources_json, created_at)`` — кэш ответов LLM.

Ключ ответа :func:`answer_cache_key` включает обе модели (LLM и эмбеддингов),
нормализованный текст запроса, ``top_k``, температуру (округлённую до 2-х
знаков), режим retrieval (``dense``/``bm25``/``hybrid``) и ``ef_search``,
что необходимо для воспроизводимости eval-сравнений.

Включение/отключение — переменная ``RAG_CACHE_ENABLED``.

CLI::

    python -m app.rag.cache clear
    python -m app.rag.cache stats
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("app.rag.cache")


def answer_cache_key(
    llm_model: str,
    emb_model: str,
    query: str,
    top_k: int,
    temperature: float,
    mode: str,
    ef_search: Optional[int],
) -> str:
    """Стабильный ключ кэша ответа. См. формат в module docstring."""
    raw = "|".join(
        [
            llm_model or "",
            emb_model or "",
            (query or "").strip().lower(),
            str(int(top_k)),
            f"{round(float(temperature), 2):.2f}",
            mode or "",
            "" if ef_search is None else str(int(ef_search)),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def text_cache_key(text: str) -> str:
    """Ключ для кэша эмбеддинга текста."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class RAGCache:
    """Тонкая sqlite3-обёртка, потокобезопасная (check_same_thread=False)."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    # ── схема ─────────────────────────────────────────────
    def _init_schema(self) -> None:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_sha256 TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    query_sha256 TEXT PRIMARY KEY,
                    top_k INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    # ── embeddings ────────────────────────────────────────
    def get_embedding(self, key: str, model: str) -> Optional[np.ndarray]:
        with closing(self._conn.cursor()) as cur:
            row = cur.execute(
                "SELECT vector FROM embeddings WHERE text_sha256=? AND model=?",
                (key, model),
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype="float32")

    def put_embedding(
        self, key: str, vector: np.ndarray, model: str
    ) -> None:
        blob = np.asarray(vector, dtype="float32").tobytes()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                (key, blob, model, time.time()),
            )

    # ── answers ───────────────────────────────────────────
    def get_answer(self, key: str) -> Optional[dict]:
        with closing(self._conn.cursor()) as cur:
            row = cur.execute(
                "SELECT top_k, temperature, answer, sources_json "
                "FROM answers WHERE query_sha256=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        top_k, temperature, answer, sources_json = row
        return {
            "top_k": int(top_k),
            "temperature": float(temperature),
            "answer": answer,
            "sources": json.loads(sources_json),
        }

    def put_answer(
        self,
        key: str,
        top_k: int,
        temperature: float,
        answer: str,
        sources: List[str],
    ) -> None:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?)",
                (
                    key,
                    int(top_k),
                    float(temperature),
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    time.time(),
                ),
            )

    # ── обслуживание ─────────────────────────────────────
    def clear(self) -> None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM embeddings;")
            cur.execute("DELETE FROM answers;")
        # VACUUM — отдельной транзакцией.
        self._conn.execute("VACUUM;")

    def stats(self) -> dict:
        with closing(self._conn.cursor()) as cur:
            n_emb = cur.execute(
                "SELECT COUNT(*) FROM embeddings;"
            ).fetchone()[0]
            n_ans = cur.execute(
                "SELECT COUNT(*) FROM answers;"
            ).fetchone()[0]
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return {
            "path": self.path,
            "embeddings": int(n_emb),
            "answers": int(n_ans),
            "size_bytes": int(size),
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def init_cache() -> Optional[RAGCache]:
    """Создать ``RAGCache`` если включён в настройках, иначе None."""
    if not settings.RAG_CACHE_ENABLED:
        logger.info("Кэш отключён (RAG_CACHE_ENABLED=false)")
        return None
    return RAGCache(settings.RAG_CACHE_PATH)


def _cli_main(argv) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )
    if len(argv) < 2:
        print(
            "Использование: python -m app.rag.cache clear|stats",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = argv[1]
    cache = RAGCache(settings.RAG_CACHE_PATH)
    if cmd == "clear":
        cache.clear()
        print("OK: кэш очищен")
    elif cmd == "stats":
        for k, v in cache.stats().items():
            print(f"{k}: {v}")
    else:
        print(f"Неизвестная команда: {cmd}", file=sys.stderr)
        sys.exit(2)
    cache.close()


if __name__ == "__main__":
    _cli_main(sys.argv)
