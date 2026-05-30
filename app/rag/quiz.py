"""
Генерация образовательных тестов по корпусу.

Поток:

1. Сбор чанков:
   * если задан ``source_filename`` — берём первые ``N`` чанков этого документа
     напрямую из docstore (без LLM);
   * иначе если задан ``topic`` — гибридный поиск через
     :class:`app.rag.retriever.HybridRetriever`;
   * если задано и то, и другое — поиск ограничивается одним документом
     (фильтр после retrieval).

2. Жёсткий JSON-промпт (см. :func:`app.rag.prompts.quiz_prompt`).
3. ``json.loads`` + pydantic-валидация каждого вопроса.
4. При полностью невалидном ответе — один retry с
   :func:`app.rag.prompts.quiz_repair_prompt`.
5. Невалидные элементы складываются в ``warnings``, валидные — в ответ.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Iterable, List, Optional, Tuple

from app.config import settings
from app.metrics import QUIZ_GENERATION_DURATION
from app.models.quiz import QuizQuestion, QuizRequest, QuizResponse
from app.rag.prompts import quiz_prompt, quiz_repair_prompt

logger = logging.getLogger("app.rag.quiz")


class QuizGenerator:
    """Высокоуровневый фасад над LLM + retriever для генерации тестов."""

    def __init__(self, rag_system) -> None:
        # rag_system — экземпляр QwenRAGSystem (имеет .retriever, .vectorstore,
        # .llm). Импорт через duck-typing, чтобы избежать кольцевого импорта.
        self.rag = rag_system

    async def generate(self, req: QuizRequest) -> QuizResponse:
        if not req.topic and not req.source_filename:
            return QuizResponse(
                questions=[], total=0,
                warnings=["Не указано ни topic, ни source_filename"],
                error=True,
            )

        start = time.time()
        status_label = "success"
        try:
            chunks = await self._collect_chunks(req)
            if not chunks:
                QUIZ_GENERATION_DURATION.labels(status="empty").observe(
                    time.time() - start
                )
                return QuizResponse(
                    questions=[], total=0,
                    warnings=["Не найдено релевантных чанков"],
                    error=False,
                )

            # Шаг 1: основной проход.
            prompt = quiz_prompt(
                chunks,
                req.num_questions,
                list(req.question_types),
                req.difficulty,
                req.language,
            )
            raw_answer = await asyncio.to_thread(
                self.rag.llm.generate,
                prompt,
                max(settings.LLM_MAX_NEW_TOKENS, 1024),
                0.5,
            )
            questions, warnings = self._parse_questions(raw_answer)

            # Шаг 2: один repair-retry если совсем плохо.
            if not questions:
                logger.info("quiz: первый ответ невалиден, repair-retry")
                warnings.append("repair-retry применён")
                repair = quiz_repair_prompt(
                    chunks,
                    req.num_questions,
                    list(req.question_types),
                    req.difficulty,
                    req.language,
                    raw_answer,
                )
                raw_answer2 = await asyncio.to_thread(
                    self.rag.llm.generate,
                    repair,
                    max(settings.LLM_MAX_NEW_TOKENS, 1024),
                    0.3,
                )
                q2, w2 = self._parse_questions(raw_answer2)
                warnings.extend(w2)
                questions = q2

            # Подстраховка: подкладываем source_chunks из retrieved, если LLM забыла.
            preview_sources = [c[:240] for c in chunks[:3]]
            for q in questions:
                if not q.source_chunks:
                    q.source_chunks = preview_sources

            limited = questions[: req.num_questions]
            if len(limited) < req.num_questions:
                warnings.append(
                    f"запрошено {req.num_questions}, валидных получено {len(limited)}"
                )
            return QuizResponse(
                questions=limited,
                total=len(limited),
                warnings=warnings,
                error=False,
            )
        except Exception as e:  # noqa: BLE001
            status_label = "error"
            logger.exception("quiz generation failed")
            return QuizResponse(
                questions=[], total=0,
                warnings=[f"ошибка: {e}"], error=True,
            )
        finally:
            QUIZ_GENERATION_DURATION.labels(status=status_label).observe(
                time.time() - start
            )

    # ── retrieval ─────────────────────────────────────────
    async def _collect_chunks(self, req: QuizRequest) -> List[str]:
        """Собрать текстовые фрагменты под генерацию."""
        target_count = max(req.num_questions * 3, 6)

        # Только source — берём первые N чанков из docstore.
        if req.source_filename and not req.topic:
            return self._chunks_by_source(req.source_filename, target_count)

        # Topic (опционально с фильтром по source).
        retriever = getattr(self.rag, "retriever", None)
        if retriever is None or retriever.vectorstore is None:
            return []
        retrieved = await retriever.search(
            req.topic or "", top_k=target_count, mode="hybrid"
        )
        if req.source_filename:
            retrieved = [
                c for c in retrieved if c.source == req.source_filename
            ]
            if not retrieved:
                # Фолбэк — взять чанки документа без topic-фильтра.
                return self._chunks_by_source(
                    req.source_filename, target_count
                )
        return [c.text for c in retrieved]

    def _chunks_by_source(self, source: str, k: int) -> List[str]:
        vs = getattr(self.rag, "vectorstore", None)
        if vs is None:
            return []
        out: List[Tuple[int, str]] = []
        for _idx, doc_id in vs.index_to_docstore_id.items():
            doc = vs.docstore.search(doc_id)
            if doc and doc.metadata.get("source") == source:
                out.append(
                    (
                        int(doc.metadata.get("chunk_index", len(out))),
                        doc.page_content,
                    )
                )
        out.sort(key=lambda x: x[0])
        return [text for _ci, text in out[:k]]

    # ── parsing ───────────────────────────────────────────
    @staticmethod
    def _parse_questions(
        raw: str,
    ) -> Tuple[List[QuizQuestion], List[str]]:
        """Разобрать ответ LLM в список ``QuizQuestion`` + warnings."""
        warnings: List[str] = []
        text = QuizGenerator._extract_json_array(raw)
        if text is None:
            return [], ["LLM не вернула JSON-массив"]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return [], [f"json.loads ошибка: {e}"]
        if not isinstance(data, list):
            return [], [
                f"ожидался JSON-массив, получен {type(data).__name__}"
            ]
        items: List[QuizQuestion] = []
        for i, raw_item in enumerate(data):
            if not isinstance(raw_item, dict):
                warnings.append(f"вопрос #{i}: не объект")
                continue
            try:
                items.append(QuizQuestion(**raw_item))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"вопрос #{i}: невалидный ({e})")
        return items, warnings

    @staticmethod
    def _extract_json_array(text: str) -> Optional[str]:
        """Толерантно достать JSON-массив из ответа LLM."""
        if not text:
            return None
        cleaned = text.strip()
        # Убрать markdown-fences ```json ... ``` (если LLM их вставила).
        cleaned = re.sub(
            r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        return cleaned[start : end + 1]
