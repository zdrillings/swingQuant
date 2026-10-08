from __future__ import annotations

import math

import pandas as pd

from src.research.orthogonal_ensemble import acceptance_window_summary


BASKET_SIZE_FLOORS = (
    ("last_fold_hit_rate_excess", ">=", 0.02),
    ("last_fold_beat_universe_rate", ">=", 0.50),
    ("last_fold_mean_target_excess", ">=", 0.0),
    ("last_fold_spearman", ">=", 0.0),
    ("last_fold_top_ticker_date_rate", "<=", 0.40),
    ("trailing_3fold_hit_rate_excess", ">=", 0.02),
    ("trailing_3fold_beat_universe_rate", ">=", 0.50),
    ("trailing_3fold_mean_target_excess", ">=", 0.0),
    ("trailing_3fold_spearman", ">=", 0.0),
    ("trailing_3fold_top_ticker_date_rate", "<=", 0.40),
)


def summarize_basket_sizes(
    predictions: pd.DataFrame,
    *,
    basket_sizes: tuple[int, ...] = (2, 3, 4, 5, 6),
    target_column: str = "alpha_vs_sector_60d",
    fold_size: int = 20,
    trailing_folds: int = 3,
) -> pd.DataFrame:
    """Return gate-style metrics for several top-N basket sizes."""
    rows: list[dict[str, object]] = []
    for basket_size in basket_sizes:
        top_n = max(int(basket_size), 1)
        summary = acceptance_window_summary(
            predictions,
            target_column=target_column,
            top_n=top_n,
            fold_size=fold_size,
            trailing_folds=trailing_folds,
        )
        rows.append(
            {
                "basket_size": top_n,
                "clears_both_beat_windows": clears_both_beat_windows(summary),
                "clears_window_floors": clears_window_floors(summary),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def clears_both_beat_windows(summary: dict[str, object]) -> bool:
    return _finite_at_least(summary.get("last_fold_beat_universe_rate"), 0.50) and _finite_at_least(
        summary.get("trailing_3fold_beat_universe_rate"), 0.50
    )


def clears_window_floors(summary: dict[str, object]) -> bool:
    for column, operator, threshold in BASKET_SIZE_FLOORS:
        value = _finite_float(summary.get(column))
        if operator == ">=" and not (math.isfinite(value) and value >= threshold):
            return False
        if operator == "<=" and not (math.isfinite(value) and value <= threshold):
            return False
    return True


def _finite_at_least(value: object, threshold: float) -> bool:
    number = _finite_float(value)
    return math.isfinite(number) and number >= float(threshold)


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")
