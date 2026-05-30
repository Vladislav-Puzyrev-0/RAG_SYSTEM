"""
Prometheus метрики для RAG-системы
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# ── Метрики ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
)

DOCUMENTS_COUNT = Gauge(
    "rag_documents_count",
    "Number of loaded documents",
)

LLM_QUERY_DURATION = Histogram(
    "rag_llm_query_duration_seconds",
    "LLM query duration",
    ["status"],
)

INGESTION_CHUNKS = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks created",
    ["status"],
)

# ── Phase 3: hybrid retrieval + cache ────────────────────
RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval duration in seconds (без LLM)",
    ["mode"],  # dense | bm25 | hybrid
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "RAG cache hits",
    ["kind"],  # answer | embedding
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "RAG cache misses",
    ["kind"],  # answer | embedding
)

# ── Phase 4 (placeholder) ────────────────────────────────
QUIZ_GENERATION_DURATION = Histogram(
    "rag_quiz_generation_duration_seconds",
    "Quiz generation duration in seconds",
    ["status"],
)


def metric_endpoint(endpoint_name):
    """Декоратор для метрик эндпоинтов."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                status = "success" if getattr(result, "error", False) is False else "error"
                REQUEST_COUNT.labels(endpoint=endpoint_name, status=status).inc()
                return result
            except Exception:
                REQUEST_COUNT.labels(endpoint=endpoint_name, status="error").inc()
                raise
            finally:
                duration = time.time() - start
                REQUEST_DURATION.labels(endpoint=endpoint_name).observe(duration)
        return wrapper
    return decorator


def metric_llm_query(func):
    """Декоратор для метрик LLM-запросов."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            status = "success" if not result.get("error") else "error"
            LLM_QUERY_DURATION.labels(status=status).observe(time.time() - start)
            return result
        except Exception:
            LLM_QUERY_DURATION.labels(status="error").observe(time.time() - start)
            raise
    return wrapper
