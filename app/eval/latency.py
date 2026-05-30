"""
Расчёт латентностных перцентилей через ``statistics.quantiles(n=100)``.

Используется для retrieval-only и end-to-end замеров в eval-CLI.
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, Sequence


def percentiles(
    values_ms: Sequence[float],
    ps: Iterable[int] = (50, 95, 99),
) -> Dict[str, float]:
    """Вернуть ``{"p50": ..., "p95": ..., "p99": ...}`` в миллисекундах.

    * 0 значений → нули.
    * 1 значение → дублируем как все перцентили (квартили требуют ≥ 2 точек).
    * ≥ 2 значений → ``statistics.quantiles(values, n=100, method="inclusive")``.
    """
    ps_list = list(ps)
    if not values_ms:
        return {f"p{p}": 0.0 for p in ps_list}
    if len(values_ms) == 1:
        v = float(values_ms[0])
        return {f"p{p}": v for p in ps_list}
    qs = statistics.quantiles(values_ms, n=100, method="inclusive")
    # qs[i] — это (i+1)-й перцентиль из 99 промежуточных точек.
    return {f"p{p}": float(qs[p - 1]) for p in ps_list}


def summary(values_ms: Sequence[float]) -> Dict[str, float]:
    """Расширенная сводка: min, mean, max + перцентили."""
    base = percentiles(values_ms)
    if not values_ms:
        base.update({"min": 0.0, "mean": 0.0, "max": 0.0, "n": 0})
        return base
    base.update(
        {
            "min": float(min(values_ms)),
            "mean": float(statistics.fmean(values_ms)),
            "max": float(max(values_ms)),
            "n": len(values_ms),
        }
    )
    return base
