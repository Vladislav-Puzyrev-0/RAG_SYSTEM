"""
app/rag/quiz_generator.py
Модуль генерации образовательных тестов по загруженным документам.
Поддерживает:
  - выбор источников (конкретных документов)
  - настройку количества вопросов
  - опциональную генерацию ответов
  - три типа вопросов: открытый, MCQ (4 варианта), правда/ложь
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("app.rag.quiz_generator")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic-модели (ответ API)
# ─────────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel


class QuizItem(BaseModel):
    question: str
    question_type: str        # "open" | "mcq" | "truefalse"
    options: Optional[List[str]] = None   # для MCQ — 4 варианта
    answer: Optional[str] = None          # None если generate_answers=False
    source: str                           # имя документа-источника
    chunk_preview: str                    # первые ~120 символов чанка


class QuizResponse(BaseModel):
    items: List[QuizItem]
    total: int
    sources_used: List[str]
    error: bool = False
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Промпты
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_OPEN = """\
<|im_start|>system
Ты — преподаватель, составляющий учебные тесты. Отвечай СТРОГО на русском языке.
<|im_end|>
<|im_start|>user
По следующему тексту сформулируй РОВНО 1 учебный вопрос открытого типа.
Требования:
- Вопрос должен проверять понимание, а не просто воспроизведение фактов.
- Вопрос должен быть самодостаточным (без указания «по тексту»).
- Выведи ТОЛЬКО текст вопроса, без нумерации и пояснений.

Текст:
{chunk}
<|im_end|>
<|im_start|>assistant
"""

PROMPT_OPEN_WITH_ANSWER = """\
<|im_start|>system
Ты — преподаватель, составляющий учебные тесты. Отвечай СТРОГО на русском языке.
<|im_end|>
<|im_start|>user
По следующему тексту сформулируй РОВНО 1 учебный вопрос открытого типа И дай краткий эталонный ответ.
Формат (строго JSON, без markdown-обёртки):
{{"question": "...", "answer": "..."}}

Текст:
{chunk}
<|im_end|>
<|im_start|>assistant
"""

PROMPT_MCQ = """\
<|im_start|>system
Ты — преподаватель, составляющий тесты с выбором ответа. Отвечай СТРОГО на русском языке.
<|im_end|>
<|im_start|>user
По следующему тексту создай 1 вопрос с 4 вариантами ответа (один правильный).
Формат (строго JSON, без markdown-обёртки):
{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A"}}

Текст:
{chunk}
<|im_end|>
<|im_start|>assistant
"""

PROMPT_TF = """\
<|im_start|>system
Ты — преподаватель, составляющий тесты «Правда/Ложь». Отвечай СТРОГО на русском языке.
<|im_end|>
<|im_start|>user
По следующему тексту придумай 1 утверждение (правда или ложь) для проверки знаний.
Формат (строго JSON, без markdown-обёртки):
{{"question": "...", "answer": "Правда"}}  или  {{"question": "...", "answer": "Ложь"}}

Текст:
{chunk}
<|im_end|>
<|im_start|>assistant
"""

QUESTION_TYPES = ["open", "mcq", "truefalse"]


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Вытащить первый JSON-объект из строки (устойчиво к мусору LLM)."""
    text = text.strip()
    # Убираем Markdown-обёртку ```json ... ```
    text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    # Находим первый {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"JSON не найден в ответе LLM: {text[:200]}")


def _parse_llm_output(raw: str, q_type: str, generate_answers: bool) -> dict:
    """Преобразовать сырой вывод LLM в структурированный dict."""
    # Вырезаем assistant-часть
    if "<|im_start|>assistant" in raw:
        raw = raw.split("<|im_start|>assistant")[-1]
    raw = raw.replace("<|im_end|>", "").strip()

    if q_type == "open" and not generate_answers:
        # LLM возвращает просто текст вопроса
        return {"question": raw.split("\n")[0].strip(), "answer": None, "options": None}

    parsed = _extract_json(raw)

    if q_type == "open":
        return {
            "question": parsed.get("question", "").strip(),
            "answer": parsed.get("answer") if generate_answers else None,
            "options": None,
        }
    elif q_type == "mcq":
        return {
            "question": parsed.get("question", "").strip(),
            "options": parsed.get("options", []),
            "answer": parsed.get("answer") if generate_answers else None,
        }
    elif q_type == "truefalse":
        return {
            "question": parsed.get("question", "").strip(),
            "options": ["Правда", "Ложь"],
            "answer": parsed.get("answer") if generate_answers else None,
        }
    return {"question": raw[:200], "answer": None, "options": None}


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────

class QuizGenerator:
    """
    Генератор тестовых вопросов поверх существующего QwenRAGSystem.

    Принимает ссылку на уже инициализированный RAG-объект, чтобы не
    дублировать загрузку модели.
    """

    def __init__(self, rag_system):
        self.rag = rag_system   # QwenRAGSystem instance

    # ------------------------------------------------------------------ #
    # Публичный метод                                                      #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        num_questions: int = 5,
        sources: Optional[List[str]] = None,
        generate_answers: bool = True,
        question_types: Optional[List[str]] = None,
        temperature: float = 0.7,
        chunks_per_question: int = 1,
    ) -> QuizResponse:
        """
        Сгенерировать тест.

        Args:
            num_questions:      Количество вопросов (1–50).
            sources:            Список имён файлов-источников; None = все.
            generate_answers:   Включить эталонные ответы в вывод.
            question_types:     Подмножество ["open", "mcq", "truefalse"].
            temperature:        Температура LLM (выше → разнообразнее).
            chunks_per_question: Сколько чанков объединять для одного вопроса.
        """
        if not self.rag.vectorstore:
            return QuizResponse(
                items=[], total=0, sources_used=[],
                error=True, message="Векторстор пуст — загрузите документы."
            )

        num_questions = max(1, min(num_questions, 50))
        q_types = question_types or QUESTION_TYPES

        # 1. Получаем чанки, отфильтрованные по источникам
        chunks = self._fetch_chunks(sources, num_questions * chunks_per_question)
        if not chunks:
            return QuizResponse(
                items=[], total=0, sources_used=[],
                error=True, message="Нет чанков по выбранным источникам."
            )

        # Перемешаем для разнообразия
        random.shuffle(chunks)

        items: List[QuizItem] = []
        sources_used: set = set()

        for i in range(num_questions):
            # Берём chunks_per_question чанков на вопрос (циклически)
            selected = [chunks[(i * chunks_per_question + j) % len(chunks)]
                        for j in range(chunks_per_question)]
            combined_text = "\n\n".join(c.page_content for c in selected)
            chunk_source = selected[0].metadata.get("source", "unknown")

            q_type = q_types[i % len(q_types)]

            try:
                item = self._generate_one(
                    combined_text, q_type, generate_answers, temperature, chunk_source
                )
                items.append(item)
                sources_used.add(chunk_source)
            except Exception as e:
                logger.warning(f"Вопрос {i+1} пропущен из-за ошибки: {e}")

        return QuizResponse(
            items=items,
            total=len(items),
            sources_used=sorted(sources_used),
            error=False,
            message=f"Сгенерировано {len(items)} вопросов."
        )

    # ------------------------------------------------------------------ #
    # Внутренние методы                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_chunks(
        self, sources: Optional[List[str]], n: int
    ) -> List[Document]:
        """Получить чанки из векторстора, опционально фильтруя по источникам."""
        vs = self.rag.vectorstore
        docstore = vs.docstore
        id_map = vs.index_to_docstore_id

        all_docs: List[Document] = []
        for doc_id in id_map.values():
            doc = docstore.search(doc_id)
            if doc is None:
                continue
            src = doc.metadata.get("source", "")
            if sources and src not in sources:
                continue
            all_docs.append(doc)

        # Если запрошено больше чанков, чем есть — возвращаем все
        if len(all_docs) <= n:
            return all_docs

        # Стратифицированная выборка по источникам
        by_source: dict[str, List[Document]] = {}
        for d in all_docs:
            s = d.metadata.get("source", "unknown")
            by_source.setdefault(s, []).append(d)

        result: List[Document] = []
        per_src = max(1, n // len(by_source))
        for docs in by_source.values():
            result.extend(random.sample(docs, min(per_src, len(docs))))

        # Дополняем до n если нужно
        remaining = [d for d in all_docs if d not in result]
        random.shuffle(remaining)
        result.extend(remaining[:n - len(result)])
        return result[:n]

    def _generate_one(
        self,
        text: str,
        q_type: str,
        generate_answers: bool,
        temperature: float,
        source: str,
    ) -> QuizItem:
        """Сформировать один вопрос через LLM."""
        # Ограничиваем длину чанка (избегаем переполнения контекста)
        text = text[:1500]

        if q_type == "open":
            template = PROMPT_OPEN_WITH_ANSWER if generate_answers else PROMPT_OPEN
        elif q_type == "mcq":
            template = PROMPT_MCQ
        elif q_type == "truefalse":
            template = PROMPT_TF
        else:
            template = PROMPT_OPEN

        prompt = template.format(chunk=text)

        response = self.rag.llm_pipeline(
            prompt,
            max_new_tokens=300,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.rag.tokenizer.eos_token_id,
        )
        raw = response[0]["generated_text"]

        parsed = _parse_llm_output(raw, q_type, generate_answers)

        return QuizItem(
            question=parsed["question"],
            question_type=q_type,
            options=parsed.get("options"),
            answer=parsed.get("answer"),
            source=source,
            chunk_preview=text[:120].replace("\n", " ") + "…",
        )
