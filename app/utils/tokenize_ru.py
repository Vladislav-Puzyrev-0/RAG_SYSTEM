"""
Простой ru-токенизатор для BM25.

Базовый режим: lowercase + regex (буквы и цифры). Дополнительно поддерживается
лемматизация через ``pymorphy3`` если пакет установлен и явно запрошена через
``use_morph=True``. ``pymorphy3`` НЕ входит в обязательные зависимости.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

# Буквы (rus + lat) + цифры подряд — стандарт для русского BM25.
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


@lru_cache(maxsize=1)
def _maybe_morph() -> Optional[object]:
    """Лениво создать ``pymorphy3.MorphAnalyzer`` или вернуть None."""
    try:
        import pymorphy3
        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


def tokenize(text: str, use_morph: bool = False) -> List[str]:
    """Разбить текст на BM25-токены.

    :param text: исходная строка
    :param use_morph: попытаться нормализовать токены через pymorphy3
        (если установлен); по умолчанию выключено
    :return: список токенов в нижнем регистре
    """
    if not text:
        return []
    raw = _TOKEN_RE.findall(text)
    tokens = [t.lower() for t in raw]
    if not use_morph:
        return tokens
    morph = _maybe_morph()
    if morph is None:
        return tokens
    return [morph.parse(t)[0].normal_form for t in tokens]


def morph_available() -> bool:
    """Установлен ли pymorphy3 (true только если import успешен)."""
    return _maybe_morph() is not None
