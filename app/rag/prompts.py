"""
Шаблоны промптов и константы парсинга ответов LLM (Qwen2.5, ChatML).

Содержит:

* :func:`ask_prompt` — для /ask и /ask/stream;
* :func:`quiz_prompt` — для /quiz/generate (жёсткий JSON-формат);
* :func:`quiz_repair_prompt` — однократный retry, если LLM выдала битый JSON.
"""
from __future__ import annotations

from typing import List, Sequence

# ChatML-теги собираем частями, чтобы не путать парсер исходника при редактировании.
SYS_TAG = "<|" + "system|>"
USR_TAG = "<|" + "user|>"
AST_TAG = "<|" + "assistant|>"
END_TAG = "<|" + "endoftext|>"


def ask_prompt(question: str, context: str) -> str:
    """Промпт для эндпоинта /ask и /ask/stream."""
    return (
        f"{SYS_TAG}\n"
        "Ты — AI-ассистент. Отвечай строго на русском. "
        "Если ответа нет в контексте — скажи "
        "«Не могу найти ответ в документах»."
        f"{END_TAG}\n"
        f"{USR_TAG}\nКонтекст:\n{context}\n\n"
        f"Вопрос: {question}{END_TAG}\n"
        f"{AST_TAG}\n"
    )


# Маркеры «нет ответа» в выводе LLM — используются ретривером для подсветки
# случаев, когда модель не нашла ответа в контексте.
NO_ANSWER_MARKERS = (
    "не могу найти", "нет в контексте", "нет информации",
    "не упоминается", "не встречается", "не содержится",
    "нет данных", "не предоставлено", "не дано",
    "в данном фрагменте", "в данном контексте",
    "не могу сказать", "нет сведений",
)


def is_no_answer(answer: str) -> bool:
    """Эвристика: LLM ответила «не знаю»."""
    if not answer:
        return True
    low = answer.lower()
    return any(marker in low for marker in NO_ANSWER_MARKERS)


# ──────────────────────────────────────────────────────────
# Quiz prompts (Phase 4)
# ──────────────────────────────────────────────────────────

_QUIZ_TYPE_DESCRIPTIONS = {
    "mcq": (
        "MCQ — вопрос с 4 вариантами (A,B,C,D), один правильный. "
        "Поле options обязательно: список из 4 строк без префикса буквы. "
        "correct_answer — буква 'A'|'B'|'C'|'D'."
    ),
    "true_false": (
        "True/False — утверждение, которое нужно оценить. options=null. "
        "correct_answer — 'true' или 'false' (строкой, в нижнем регистре)."
    ),
    "fill_in_blank": (
        "Fill-in-the-blank — вопрос с пропуском, обозначенным как '___'. "
        "options=null. correct_answer — строка, которой надо заполнить пропуск."
    ),
}


def _format_quiz_context(chunks: Sequence[str]) -> str:
    """Слепить чанки в нумерованный контекст для квиз-промпта."""
    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[Фрагмент {i}]\n{ch}")
    return "\n\n---\n\n".join(parts)


def quiz_prompt(
    chunks: Sequence[str],
    num_questions: int,
    question_types: List[str],
    difficulty: str,
    language: str,
) -> str:
    """Жёсткий JSON-промпт для /quiz/generate.

    Требования к LLM:
    * Вернуть ТОЛЬКО валидный JSON-массив, без markdown-обёртки, без пояснений
      ДО или ПОСЛЕ массива.
    * Каждый элемент массива — объект со строго заданными полями.
    """
    lang_hint = "русском" if language == "ru" else "английском"
    types_block = "\n".join(
        f"- {t}: {_QUIZ_TYPE_DESCRIPTIONS[t]}"
        for t in question_types
        if t in _QUIZ_TYPE_DESCRIPTIONS
    )
    context = _format_quiz_context(chunks)
    return (
        f"{SYS_TAG}\n"
        "Ты — преподаватель, который составляет тестовые задания по предоставленным материалам. "
        "Отвечай ИСКЛЮЧИТЕЛЬНО валидным JSON-массивом. "
        "Без markdown-обёртки (```), без пояснений до или после массива."
        f"{END_TAG}\n"
        f"{USR_TAG}\n"
        f"Составь ровно {num_questions} вопросов на {lang_hint} языке. "
        f"Сложность: {difficulty}.\n\n"
        f"Разрешённые типы вопросов:\n{types_block}\n\n"
        f"Контекст:\n{context}\n\n"
        "Формат ответа — массив JSON, каждый элемент строго такого вида:\n"
        "{\n"
        '  "type": "mcq" | "true_false" | "fill_in_blank",\n'
        '  "question": "текст вопроса",\n'
        '  "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"] | null,\n'
        '  "correct_answer": "строка",\n'
        '  "explanation": "пояснение почему этот ответ правильный",\n'
        '  "source_chunks": ["цитата 1", "цитата 2"]\n'
        "}\n"
        "Никаких ключей сверх этих шести. Никакого текста вне массива."
        f"{END_TAG}\n"
        f"{AST_TAG}\n"
    )


def quiz_repair_prompt(
    chunks: Sequence[str],
    num_questions: int,
    question_types: List[str],
    difficulty: str,
    language: str,
    previous_bad_answer: str,
) -> str:
    """Repair-промпт: однократный retry если первый ответ невалиден.

    Включает фрагмент битого ответа и явное напоминание о формате.
    """
    base = quiz_prompt(
        chunks, num_questions, question_types, difficulty, language
    )
    # Заменим финальный assistant-tag на «исправь предыдущий ответ»-вставку.
    base_without_tail = base.rsplit(f"{AST_TAG}\n", 1)[0]
    bad_preview = previous_bad_answer.strip()[:600]
    return (
        base_without_tail
        + f"{USR_TAG}\nПредыдущая попытка вернула НЕвалидный JSON. "
        + "Перевыпусти ответ строго в требуемом формате (только JSON-массив, "
        + "без markdown-обёрток, без пояснений). Пример невалидного ответа "
        + "(не повторяй эти ошибки):\n"
        + bad_preview
        + f"{END_TAG}\n"
        + f"{AST_TAG}\n"
    )
