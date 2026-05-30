"""
Конфигурация логирования.

Поддерживает два формата (выбор через ``LOG_FORMAT`` в ``.env``):

* ``console`` — человекочитаемый формат для разработки.
* ``json`` — однострочный JSON для прод-окружения / ELK-стека.

К каждому сообщению автоматически прикрепляется ``request_id`` из
``contextvars`` (см. :mod:`app.middleware`), что позволяет связывать
несколько лог-записей одного HTTP-запроса.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from app.config import settings


# ──────────────────────────────────────────────────────────
# Контекст запроса
# ──────────────────────────────────────────────────────────
#: ID текущего HTTP-запроса (заполняется ``RequestIDMiddleware``).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIDFilter(logging.Filter):
    """Прокидывает ``request_id`` из contextvars в ``LogRecord``."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Сериализует ``LogRecord`` в одну JSON-строку."""

    #: Стандартные поля ``LogRecord``, которые мы НЕ копируем в payload.
    _RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # Дополнительные поля, переданные через ``extra={...}``.
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONSOLE_FMT = "%(asctime)s [%(levelname)s] %(name)s rid=%(request_id)s - %(message)s"


def setup_logging() -> None:
    """Настроить корневой логгер согласно ``settings``."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(_RequestIDFilter())

    if settings.LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_CONSOLE_FMT))

    root = logging.getLogger()
    # На повторных вызовах (например, в тестах) убираем старые handlers,
    # чтобы не дублировать вывод.
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn-логгеры — пусть тоже идут через наш handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        lg.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Хелпер для роутеров и сервисов.

    Возвращает стандартный ``logging.Logger`` — request_id прокидывается
    автоматически фильтром, заданным в :func:`setup_logging`.
    """
    return logging.getLogger(name)
