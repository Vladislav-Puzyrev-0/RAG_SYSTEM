"""Middleware и глобальный exception handler.

Содержит:

* :class:`RequestIDMiddleware` — генерирует или принимает заголовок
  ``X-Request-ID``, кладёт его в ``contextvars`` и в response-заголовки.
* :func:`install_exception_handler` — единый формат ошибок
  :class:`app.models.common.ErrorResponse`.
"""
from __future__ import annotations

import logging
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.logging_config import request_id_var
from app.models.common import ErrorResponse

logger = logging.getLogger("app.middleware")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Добавляет ``X-Request-ID`` к каждому запросу."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


def install_exception_handler(app: FastAPI) -> None:
    """Регистрирует обработчики ошибок c единым форматом ответа."""

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):  # noqa: D401
        rid = request_id_var.get()
        logger.info(
            "HTTPException %s on %s: %s",
            exc.status_code,
            request.url.path,
            exc.detail,
        )
        payload = ErrorResponse(
            code=f"http_{exc.status_code}",
            message=str(exc.detail),
            request_id=rid,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
            headers={REQUEST_ID_HEADER: rid},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(
        request: Request, exc: RequestValidationError
    ):  # noqa: D401
        rid = request_id_var.get()
        # Краткое сообщение: первая ошибка + общее количество.
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", ()))
        msg = first.get("msg", "validation error")
        payload = ErrorResponse(
            code="validation_error",
            message=f"{loc}: {msg}" if loc else msg,
            request_id=rid,
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(),
            headers={REQUEST_ID_HEADER: rid},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: D401
        rid = request_id_var.get()
        logger.exception(
            "Unhandled exception on %s: %s", request.url.path, exc
        )
        payload = ErrorResponse(
            code="internal_error",
            message="Внутренняя ошибка сервера",
            request_id=rid,
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(),
            headers={REQUEST_ID_HEADER: rid},
        )
