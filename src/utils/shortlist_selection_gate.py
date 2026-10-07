from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShortlistSelectionGate:
    enabled: bool = False
    method: str = "score_quantile"
    quantile: float = 0.95
    lookback_sessions: int = 126

    @classmethod
    def from_config(cls, payload: dict | None) -> "ShortlistSelectionGate":
        raw = payload if isinstance(payload, dict) else {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            method=str(raw.get("method", "score_quantile") or "score_quantile"),
            quantile=float(raw.get("quantile", 0.95)),
            lookback_sessions=int(raw.get("lookback_sessions", 126)),
        )

    @property
    def active(self) -> bool:
        return self.enabled and self.method == "score_quantile"


def rolling_score_quantile_thresholds(
    frame: pd.DataFrame,
    *,
    score_column: str = "predicted_alpha",
    date_column: str = "snapshot_date",
    quantile: float = 0.95,
    lookback_sessions: int = 126,
) -> pd.Series:
    if frame.empty or score_column not in frame.columns or date_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    working = frame[[date_column, score_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column]).dt.normalize()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    thresholds_by_date: dict[pd.Timestamp, float] = {}
    unique_dates = sorted(working[date_column].dropna().unique().tolist())
    lookback = max(int(lookback_sessions), 1)
    q = min(max(float(quantile), 0.0), 1.0)
    for offset, current_date in enumerate(unique_dates):
        prior_dates = unique_dates[max(0, offset - lookback) : offset]
        if not prior_dates:
            thresholds_by_date[pd.Timestamp(current_date)] = float("nan")
            continue
        prior_scores = working.loc[working[date_column].isin(prior_dates), score_column].dropna()
        thresholds_by_date[pd.Timestamp(current_date)] = (
            float(prior_scores.quantile(q)) if not prior_scores.empty else float("nan")
        )
    return working[date_column].map(lambda value: thresholds_by_date.get(pd.Timestamp(value), float("nan")))


def latest_score_quantile_threshold(
    frame: pd.DataFrame,
    *,
    score_column: str = "predicted_alpha",
    date_column: str = "snapshot_date",
    quantile: float = 0.95,
    lookback_sessions: int = 126,
) -> float | None:
    if frame.empty or score_column not in frame.columns or date_column not in frame.columns:
        return None
    working = frame[[date_column, score_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column]).dt.normalize()
    dates = sorted(working[date_column].dropna().unique().tolist())
    if not dates:
        return None
    lookback = max(int(lookback_sessions), 1)
    prior_dates = dates[-lookback:]
    prior_scores = pd.to_numeric(
        working.loc[working[date_column].isin(prior_dates), score_column],
        errors="coerce",
    ).dropna()
    if prior_scores.empty:
        return None
    q = min(max(float(quantile), 0.0), 1.0)
    return float(prior_scores.quantile(q))


def apply_score_quantile_gate(
    frame: pd.DataFrame,
    *,
    gate: ShortlistSelectionGate,
    threshold: float | None = None,
    score_column: str = "predicted_alpha",
) -> pd.DataFrame:
    if not gate.active:
        return frame.copy()
    if threshold is None or not np.isfinite(float(threshold)):
        return frame.iloc[0:0].copy()
    if frame.empty or score_column not in frame.columns:
        return frame.iloc[0:0].copy()
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    return frame.loc[scores >= float(threshold)].copy()
