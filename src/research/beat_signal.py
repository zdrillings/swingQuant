from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_BEAT_THRESHOLD = 0.0


def add_forward_beat_label(
    frame: pd.DataFrame,
    *,
    source_column: str = "alpha_vs_sector_60d",
    target_column: str = "forward_beat_sector_60d",
    threshold: float = DEFAULT_BEAT_THRESHOLD,
) -> pd.DataFrame:
    working = frame.copy()
    source = pd.to_numeric(working[source_column], errors="coerce")
    working[target_column] = np.where(source.notna(), source.gt(float(threshold)).astype(float), np.nan)
    return working


def build_rank_hybrid_predictions(
    beat_predictions: pd.DataFrame,
    alpha_predictions: pd.DataFrame,
    *,
    beat_weight: float,
    alpha_weight: float | None = None,
    beat_score_column: str = "predicted_alpha",
    alpha_score_column: str = "predicted_alpha",
    target_column: str = "alpha_vs_sector_60d",
    model_name: str = "beat_alpha_rank_hybrid",
) -> pd.DataFrame:
    if alpha_weight is None:
        alpha_weight = 1.0 - float(beat_weight)
    total = float(beat_weight) + float(alpha_weight)
    if not math.isfinite(total) or total <= 0.0:
        return pd.DataFrame()

    beat = _rank_frame(
        beat_predictions,
        score_column=beat_score_column,
        rank_column="_beat_rank",
        target_column=target_column,
    )
    alpha = _rank_frame(
        alpha_predictions,
        score_column=alpha_score_column,
        rank_column="_alpha_rank",
        target_column=target_column,
    )
    if beat.empty or alpha.empty:
        return pd.DataFrame()

    merged = beat.merge(
        alpha[["snapshot_date", "ticker", "_alpha_rank"]],
        on=["snapshot_date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["predicted_alpha"] = (
        (float(beat_weight) / total) * pd.to_numeric(merged["_beat_rank"], errors="coerce")
        + (float(alpha_weight) / total) * pd.to_numeric(merged["_alpha_rank"], errors="coerce")
    )
    merged["model_name"] = str(model_name)
    return merged.drop(columns=["_beat_rank", "_alpha_rank"], errors="ignore").copy()


def _rank_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    rank_column: str,
    target_column: str,
) -> pd.DataFrame:
    required = {"snapshot_date", "ticker", score_column}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.strip()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    keep_columns = ["snapshot_date", "ticker", score_column]
    if target_column in working.columns:
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        keep_columns.append(target_column)
    working = working.dropna(subset=["snapshot_date", "ticker", score_column])
    if working.empty:
        return pd.DataFrame()
    working[rank_column] = working[score_column].groupby(working["snapshot_date"]).rank(method="average", pct=True)
    keep_columns.append(rank_column)
    return working[keep_columns].drop_duplicates(["snapshot_date", "ticker"]).copy()
