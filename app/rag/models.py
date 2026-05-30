"""
Инициализация моделей: эмбеддинги и LLM
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from app.config import settings

logger = logging.getLogger(__name__)


def init_embeddings():
    """Инициализировать эмбеддинги из локального кэша."""
    from huggingface_hub import snapshot_download

    # Получаем путь к локальному кэшу (без сети)
    model_path = snapshot_download(
        settings.EMBEDDING_MODEL,
        local_files_only=True,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
    )
    logger.info(f"✓ Эмбеддинги: {settings.EMBEDDING_MODEL} (local)")
    return embeddings


def init_llm():
    """Инициализировать LLM (Qwen2.5-7B-Instruct) из локального кэша."""
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Устройство: CPU")

    # Получаем путь к локальному кэшу (без сети)
    model_path = snapshot_download(
        settings.LLM_MODEL,
        local_files_only=True,
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=settings.LLM_DEVICE,
        local_files_only=True,
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_DEFAULT_TEMPERATURE,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    logger.info(f"✓ LLM: {settings.LLM_MODEL}")

    return llm, llm_pipeline, tokenizer
