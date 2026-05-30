"""
Обёртка над Qwen2.5 с двумя режимами генерации:

* :meth:`QwenLLM.generate` — синхронная генерация полного ответа (используется
  эндпоинтом ``/ask``).
* :meth:`QwenLLM.stream` — потоковая генерация через
  ``transformers.TextIteratorStreamer`` в отдельном потоке. Используется
  эндпоинтом ``/ask/stream``.

Загрузка весов выполняется через ``huggingface_hub.snapshot_download(...,
local_files_only=True)`` — модель должна быть заранее скачана в локальный
HuggingFace-кэш.
"""
from __future__ import annotations

import logging
from threading import Thread
from typing import Iterator, Optional, Tuple

import torch

from app.config import settings

logger = logging.getLogger("app.rag.llm")


class QwenLLM:
    """Тонкий фасад над Qwen-моделью."""

    def __init__(self, model, tokenizer, device_human: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device_human = device_human
        self.eos_token_id = tokenizer.eos_token_id

    # ── sync (используется /ask) ──────────────────────────
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        temperature = (
            settings.LLM_DEFAULT_TEMPERATURE
            if temperature is None
            else float(temperature)
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.eos_token_id,
            )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── stream (используется /ask/stream) ─────────────────
    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Генератор: отдаёт куски сгенерированного текста по мере появления."""
        from transformers import TextIteratorStreamer

        max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        temperature = (
            settings.LLM_DEFAULT_TEMPERATURE
            if temperature is None
            else float(temperature)
        )
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.eos_token_id,
        )
        thread = Thread(
            target=self._safe_generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            thread.join(timeout=1.0)

    # ── внутреннее ────────────────────────────────────────
    def _safe_generate(self, **kwargs) -> None:
        try:
            with torch.inference_mode():
                self.model.generate(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.exception("Stream-generation thread failed: %s", e)


def init_llm() -> Tuple[QwenLLM, str]:
    """Инициализировать модель из локального HF-кэша.

    :return: ``(llm, device_human_name)``, где второе значение полезно
        для ``/status`` и логов.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    device_human = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    logger.info("Устройство для LLM: %s", device_human)

    model_path = snapshot_download(settings.LLM_MODEL, local_files_only=True)
    dtype = torch.float16 if has_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=settings.LLM_DEVICE,
        local_files_only=True,
    )
    logger.info("LLM: %s готова", settings.LLM_MODEL)
    return QwenLLM(model, tokenizer, device_human), device_human
