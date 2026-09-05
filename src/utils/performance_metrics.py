from __future__ import annotations

import math

import numpy as np
import pandas as pd


def newey_west_t_stat(values, *, lag: int = 0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n_obs = int(len(series.index))
    if n_obs < 2:
        return float("nan")
    demeaned = series.to_numpy(dtype=float) - float(series.mean())
    gamma0 = float(np.dot(demeaned, demeaned) / n_obs)
    variance = gamma0
    max_lag = min(max(int(lag), 0), n_obs - 1)
    for lag_index in range(1, max_lag + 1):
        cov = float(np.dot(demeaned[lag_index:], demeaned[:-lag_index]) / n_obs)
        weight = 1.0 - (lag_index / (max_lag + 1.0))
        variance += 2.0 * weight * cov
    if variance <= 0.0 or not math.isfinite(variance):
        return float("nan")
    standard_error = math.sqrt(variance / n_obs)
    if standard_error <= 0.0:
        return float("nan")
    return float(series.mean()) / standard_error


def annualized_sharpe(values, *, periods_per_year: int = 252) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(series.index) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return float("nan")
    return float(series.mean()) / std * math.sqrt(float(periods_per_year))


def years_required_for_tstat(sharpe: float, *, target_t_stat: float = 1.96) -> float:
    try:
        numeric = abs(float(sharpe))
    except (TypeError, ValueError):
        return float("nan")
    if numeric <= 0.0 or not math.isfinite(numeric):
        return float("nan")
    return (float(target_t_stat) / numeric) ** 2


def beta_to_benchmark(strategy_values, benchmark_values) -> float:
    strategy = pd.to_numeric(pd.Series(strategy_values), errors="coerce")
    benchmark = pd.to_numeric(pd.Series(benchmark_values), errors="coerce")
    valid = strategy.notna() & benchmark.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    bench = benchmark[valid].to_numpy(dtype=float)
    strat = strategy[valid].to_numpy(dtype=float)
    variance = float(np.var(bench, ddof=1))
    if variance <= 0.0 or not math.isfinite(variance):
        return float("nan")
    covariance = float(np.cov(strat, bench, ddof=1)[0, 1])
    return covariance / variance
