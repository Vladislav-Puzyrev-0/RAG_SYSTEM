"""
CLI оффлайн-оценки retrieval (и опционально end-to-end с LLM).

Запуск::

    python -m app.eval.run \\
        --dataset eval_data/golden.example.jsonl \\
        --mode hybrid \\
        --top-k 5 \\
        --output results/eval_$(date +%s).json
        [--with-llm]

LLM загружается ТОЛЬКО при ``--with-llm`` (в обычном retrieval-режиме грузим
лишь embeddings + FAISS + BM25 — быстро и без Qwen-весов).

На выходе: JSON-файл со всеми деталями + markdown-отчёт (рядом, с тем же
именем и расширением .md) для прямой вставки в диссертацию.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.eval.dataset import GoldenItem, load_dataset
from app.eval.latency import summary as latency_summary
from app.eval.metrics import METRIC_NAMES, compute_all
from app.rag.bm25 import ensure_bm25
from app.rag.embeddings import init_embeddings
from app.rag.prompts import ask_prompt
from app.rag.retrieval import _build_context
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import load_vectorstore

logger = logging.getLogger("app.eval.run")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m app.eval.run",
        description="Оффлайн-оценка retrieval (P@k, R@k, MRR, nDCG@k, Hit@k).",
    )
    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Путь к JSONL-датасету (см. app/eval/dataset.py).",
    )
    p.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--output", required=True, type=Path,
        help="Куда сохранить JSON-отчёт. Markdown-отчёт ляжет рядом с .md.",
    )
    p.add_argument(
        "--with-llm", action="store_true",
        help="Замерить также end-to-end latency (загрузка Qwen ~30–60s на CPU).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Ограничить число запросов для быстрых проверок.",
    )
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────
async def _evaluate(args: argparse.Namespace) -> dict:
    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("Пустой датасет — нечего оценивать.")

    logger.info("Инициализация retriever…")
    embeddings = init_embeddings()
    vectorstore = load_vectorstore(embeddings)
    if vectorstore is None or vectorstore.index.ntotal == 0:
        raise SystemExit(
            f"Пустой векторстор в {settings.VECTORSTORE_PATH}. "
            "Загрузи документы через /upload перед оценкой."
        )
    bm25 = ensure_bm25(vectorstore)
    retriever = HybridRetriever(vectorstore, embeddings, bm25)

    llm = None
    if args.with_llm:
        logger.info("Загрузка LLM (--with-llm)…")
        from app.rag.llm import init_llm
        llm, _device = init_llm()

    per_query: list[dict] = []
    retrieval_ms: list[float] = []
    e2e_ms: list[float] = []

    for i, item in enumerate(items, start=1):
        t0 = time.time()
        chunks = await retriever.search(
            item.query, top_k=args.top_k, mode=args.mode
        )
        ret_dur = (time.time() - t0) * 1000
        retrieval_ms.append(ret_dur)

        metrics = compute_all(chunks, item, args.top_k)
        record: dict = {
            "query": item.query,
            "relevant_sources": item.relevant_sources,
            "top_k_sources": [c.source for c in chunks],
            "metrics": metrics,
            "retrieval_ms": ret_dur,
        }

        if llm is not None:
            docs = []
            for c in chunks:
                d = vectorstore.docstore.search(c.doc_id)
                if d is not None:
                    docs.append(d)
            prompt = ask_prompt(item.query, _build_context(docs))
            llm_t0 = time.time()
            answer = await asyncio.to_thread(
                llm.generate, prompt, settings.LLM_MAX_NEW_TOKENS, 0.3
            )
            total_dur = (time.time() - llm_t0) * 1000 + ret_dur
            record["llm_ms"] = (time.time() - llm_t0) * 1000
            record["end_to_end_ms"] = total_dur
            record["answer"] = answer
            e2e_ms.append(total_dur)

        per_query.append(record)
        logger.info(
            "[%d/%d] %.0fms hit@k=%.0f mrr=%.2f — %s",
            i, len(items), ret_dur,
            metrics["hit_rate_at_k"], metrics["mrr"],
            item.query[:60],
        )

    # Агрегация: среднее по запросам.
    aggregated = {
        name: float(statistics.fmean(q["metrics"][name] for q in per_query))
        for name in METRIC_NAMES
    }

    return {
        "dataset": str(args.dataset),
        "mode": args.mode,
        "top_k": args.top_k,
        "with_llm": bool(llm is not None),
        "num_queries": len(per_query),
        "metrics": aggregated,
        "latency_retrieval_ms": latency_summary(retrieval_ms),
        "latency_end_to_end_ms": latency_summary(e2e_ms) if e2e_ms else None,
        "settings": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "faiss_metric": settings.FAISS_METRIC,
            "hnsw_m": settings.FAISS_HNSW_M,
            "hnsw_ef_search": settings.FAISS_HNSW_EF_SEARCH,
            "hybrid_rrf_k": settings.HYBRID_RRF_K,
        },
        "per_query": per_query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────
def _render_markdown(report: dict) -> str:
    m = report["metrics"]
    lat_r = report["latency_retrieval_ms"]
    lat_e = report["latency_end_to_end_ms"]

    lines: list[str] = []
    lines.append("# RAG offline evaluation\n")
    lines.append(f"- Датасет: `{report['dataset']}`")
    lines.append(f"- Режим: **{report['mode']}**")
    lines.append(f"- top_k: **{report['top_k']}**")
    lines.append(f"- Запросов: **{report['num_queries']}**")
    lines.append(f"- LLM end-to-end: **{report['with_llm']}**")
    lines.append(f"- Когда: {report['timestamp']}\n")

    lines.append("## Качество retrieval\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    lines.append(f"| Precision@{report['top_k']} | {m['precision_at_k']:.4f} |")
    lines.append(f"| Recall@{report['top_k']} | {m['recall_at_k']:.4f} |")
    lines.append(f"| MRR | {m['mrr']:.4f} |")
    lines.append(f"| nDCG@{report['top_k']} | {m['ndcg_at_k']:.4f} |")
    lines.append(f"| Hit-rate@{report['top_k']} | {m['hit_rate_at_k']:.4f} |")
    lines.append("")

    lines.append("## Латентность retrieval, ms\n")
    lines.append("| p50 | p95 | p99 | mean | n |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {lat_r['p50']:.1f} | {lat_r['p95']:.1f} | {lat_r['p99']:.1f} | "
        f"{lat_r['mean']:.1f} | {lat_r['n']} |"
    )
    lines.append("")

    if lat_e:
        lines.append("## Латентность end-to-end (retrieval + LLM), ms\n")
        lines.append("| p50 | p95 | p99 | mean | n |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| {lat_e['p50']:.1f} | {lat_e['p95']:.1f} | {lat_e['p99']:.1f} | "
            f"{lat_e['mean']:.1f} | {lat_e['n']} |"
        )
        lines.append("")

    s = report["settings"]
    lines.append("## Конфигурация\n")
    lines.append("| Параметр | Значение |")
    lines.append("|---|---|")
    for k, v in s.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## По запросам\n")
    lines.append("| # | query | hit@k | mrr | nDCG | ret_ms |")
    lines.append("|---|---|---|---|---|---|")
    for i, q in enumerate(report["per_query"], start=1):
        qm = q["metrics"]
        qtext = q["query"].replace("|", "\\|")
        if len(qtext) > 60:
            qtext = qtext[:57] + "…"
        lines.append(
            f"| {i} | {qtext} | {qm['hit_rate_at_k']:.0f} | "
            f"{qm['mrr']:.2f} | {qm['ndcg_at_k']:.2f} | "
            f"{q['retrieval_ms']:.0f} |"
        )

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────
def _write_outputs(report: dict, out_json: Path) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    return out_md


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = asyncio.run(_evaluate(args))
    out_md = _write_outputs(report, args.output)
    print(f"JSON:     {args.output}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
