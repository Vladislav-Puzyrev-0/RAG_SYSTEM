"""
FAISS-helpers: создание, сохранение, загрузка индекса, миграции.

Поддерживает обе метрики (``ip`` — inner-product и ``l2``) и динамический
``efSearch`` для HNSW. Содержит CLI для миграции существующего L2-индекса
на inner-product без пересчёта эмбеддингов::

    python -m app.rag.vectorstore migrate-to-ip
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Optional, Tuple

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from app.config import settings

logger = logging.getLogger("app.rag.vectorstore")

_METRIC_BY_NAME = {
    "ip": faiss.METRIC_INNER_PRODUCT,
    "l2": faiss.METRIC_L2,
}
_NAME_BY_METRIC = {v: k for k, v in _METRIC_BY_NAME.items()}


def _metric_name(index: faiss.Index) -> str:
    """Имя метрики индекса (``ip`` | ``l2`` | ``?``)."""
    return _NAME_BY_METRIC.get(getattr(index, "metric_type", -1), "?")


def detect_faiss_gpu() -> Tuple[bool, Optional[object]]:
    """Определить доступность GPU-сборки FAISS."""
    if not settings.FAISS_USE_GPU:
        return False, None
    try:
        res = faiss.StandardGpuResources()
        ngpu = faiss.get_num_gpus()
        if ngpu > 0:
            logger.info("FAISS GPU доступен: %d GPU(s)", ngpu)
            return True, res
        logger.warning(
            "FAISS_USE_GPU=true, но GPU не найден — fallback на CPU"
        )
        return False, None
    except Exception as e:
        logger.warning("FAISS GPU недоступен (%s), fallback на CPU", e)
        return False, None


def build_hnsw_index(
    embedding_dim: int, metric: Optional[str] = None
) -> faiss.Index:
    """Создать пустой HNSW-индекс с метрикой из ``settings`` или явной."""
    metric = (metric or settings.FAISS_METRIC).lower()
    if metric not in _METRIC_BY_NAME:
        raise ValueError(f"Неизвестная FAISS_METRIC={metric!r}")
    index = faiss.IndexHNSWFlat(
        embedding_dim,
        settings.FAISS_HNSW_M,
        _METRIC_BY_NAME[metric],
    )
    index.hnsw.efConstruction = settings.FAISS_HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = settings.FAISS_HNSW_EF_SEARCH
    return index


def make_faiss_langchain_store(index, embeddings, docstore=None):
    """Обернуть FAISS-индекс в LangChain-совместимый ``FAISS``-VectorStore."""
    if docstore is None:
        docstore = InMemoryDocstore()
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={},
    )


def create_vectorstore(chunks, embeddings):
    """Создать векторстор (HNSW если включён, иначе Flat)."""
    if settings.FAISS_USE_HNSW:
        dim = len(embeddings.embed_query("test"))
        hnsw = build_hnsw_index(dim)
        store = make_faiss_langchain_store(hnsw, embeddings)
        store.add_documents(chunks)
        logger.info(
            "Создан HNSW-индекс (M=%d, metric=%s)",
            settings.FAISS_HNSW_M,
            settings.FAISS_METRIC,
        )
        return store
    return FAISS.from_documents(chunks, embeddings)


def migrate_to_hnsw(vectorstore):
    """Перестроить Flat-индекс в HNSW (сохранено для обратной совместимости)."""
    old_index = vectorstore.index
    n = old_index.ntotal
    if n == 0:
        return
    dim = old_index.d
    vectors = old_index.reconstruct_n(0, n)
    hnsw = build_hnsw_index(dim)
    hnsw.add(vectors)
    vectorstore.index = hnsw
    logger.info(
        "Flat(%d) → HNSW(M=%d, ef=%d, metric=%s)",
        n,
        settings.FAISS_HNSW_M,
        settings.FAISS_HNSW_EF_SEARCH,
        settings.FAISS_METRIC,
    )


def set_ef_search(vectorstore, ef: int) -> None:
    """Изменить ``efSearch`` HNSW-индекса в рантайме.

    Замечание: операция мутирует общий индекс — между параллельными
    поисковыми запросами действует последнее заданное значение.
    """
    if vectorstore is None or vectorstore.index is None:
        return
    idx = vectorstore.index
    if isinstance(idx, faiss.IndexHNSW):
        idx.hnsw.efSearch = int(ef)


def save_vectorstore(vectorstore):
    """Сохранить векторстор (с миграцией Flat→HNSW при необходимости)."""
    if vectorstore is None:
        return

    if settings.FAISS_USE_HNSW:
        current = vectorstore.index
        if not isinstance(current, faiss.IndexHNSW):
            logger.info("Миграция Flat → HNSW…")
            migrate_to_hnsw(vectorstore)

    has_gpu, _ = detect_faiss_gpu()
    if has_gpu:
        try:
            vectorstore.index = faiss.index_gpu_to_cpu(vectorstore.index)
            logger.info("Индекс временно перенесён на CPU для сохранения")
        except Exception:
            pass

    vectorstore.save_local(settings.VECTORSTORE_PATH)
    logger.info("Векторстор сохранён в %s", settings.VECTORSTORE_PATH)

    if has_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
            vectorstore.index = faiss.index_cpu_to_gpu(
                gpu_res, 0, vectorstore.index
            )
        except Exception:
            pass


def load_vectorstore(embeddings):
    """Загрузить сохранённый векторстор; предупредить о mismatch метрики."""
    index_file = os.path.join(settings.VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        logger.info("Сохранённый векторстор не найден — будет создан заново")
        return None
    try:
        has_gpu, gpu_res = detect_faiss_gpu()

        raw_index = faiss.read_index(index_file)
        is_hnsw = isinstance(raw_index, faiss.IndexHNSW)
        m_name = _metric_name(raw_index)
        ef = raw_index.hnsw.efSearch if is_hnsw else -1
        logger.info(
            "Индекс: %s (%d векторов, metric=%s, efSearch=%d)",
            "HNSW" if is_hnsw else "Flat",
            raw_index.ntotal,
            m_name,
            ef,
        )
        if m_name != settings.FAISS_METRIC:
            logger.warning(
                "FAISS_METRIC=%s, но сохранённый индекс использует %s. "
                "Запусти `python -m app.rag.vectorstore migrate-to-ip` "
                "для миграции на inner-product без пересчёта эмбеддингов.",
                settings.FAISS_METRIC,
                m_name,
            )

        vectorstore = FAISS.load_local(
            settings.VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        if has_gpu and not is_hnsw:
            try:
                vectorstore.index = faiss.index_cpu_to_gpu(
                    gpu_res, 0, vectorstore.index
                )
                logger.info("Векторстор мигрирован на GPU")
            except Exception as e:
                logger.warning("Не удалось мигрировать на GPU: %s", e)

        logger.info(
            "Векторстор загружен из %s (%d векторов)",
            settings.VECTORSTORE_PATH,
            vectorstore.index.ntotal,
        )
        return vectorstore
    except Exception as e:
        logger.warning("Не удалось загрузить векторстор: %s", e)
        return None


def migrate_l2_to_ip(vectorstore_path: Optional[str] = None) -> None:
    """Миграция существующего L2-индекса на inner-product.

    Не пересчитывает эмбеддинги: реконструирует векторы из FAISS,
    нормализует L2 и собирает новый HNSW с ``METRIC_INNER_PRODUCT``.
    Исходный ``index.faiss`` сохраняется как ``index.faiss.l2.bak``.
    Docstore (``index.pkl``) НЕ затрагивается — ID чанков сохраняются.
    """
    path = vectorstore_path or settings.VECTORSTORE_PATH
    index_file = os.path.join(path, "index.faiss")
    if not os.path.exists(index_file):
        logger.error("Файл %s не найден — нечего мигрировать", index_file)
        return

    raw = faiss.read_index(index_file)
    if raw.metric_type == faiss.METRIC_INNER_PRODUCT:
        logger.info("Индекс уже использует inner-product — миграция не нужна")
        return

    n, d = raw.ntotal, raw.d
    logger.info("Реконструкция %d векторов (dim=%d)…", n, d)
    vecs = raw.reconstruct_n(0, n).astype("float32", copy=False)
    faiss.normalize_L2(vecs)

    new_index = faiss.IndexHNSWFlat(
        d, settings.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT
    )
    new_index.hnsw.efConstruction = settings.FAISS_HNSW_EF_CONSTRUCTION
    new_index.hnsw.efSearch = settings.FAISS_HNSW_EF_SEARCH
    new_index.add(vecs)
    logger.info("Новый HNSW(IP) индекс собран, ntotal=%d", new_index.ntotal)

    backup = index_file + ".l2.bak"
    if not os.path.exists(backup):
        shutil.copy2(index_file, backup)
        logger.info("Бэкап исходного индекса: %s", backup)
    else:
        logger.warning("Бэкап %s уже существует — перезапись пропущена", backup)

    faiss.write_index(new_index, index_file)
    logger.info("%s переписан (METRIC_INNER_PRODUCT)", index_file)
    logger.info("docstore (index.pkl) не затронут — id чанков сохранены")


def _cli_main(argv) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )
    if len(argv) < 2:
        print(
            "Использование: python -m app.rag.vectorstore migrate-to-ip",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = argv[1]
    if cmd == "migrate-to-ip":
        migrate_l2_to_ip()
        return
    print(f"Неизвестная команда: {cmd}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    _cli_main(sys.argv)
