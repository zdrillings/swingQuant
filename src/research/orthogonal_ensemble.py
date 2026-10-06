from __future__ import annotations

import math

import pandas as pd


DEFAULT_WEIGHT_FLOOR = 0.001


def inverse_absolute_spearman_weights(
    spearman_by_model: dict[str, float],
    *,
    floor: float = DEFAULT_WEIGHT_FLOOR,
) -> dict[str, float]:
    """Normalize inverse-absolute-Spearman weights for ensemble rank averaging."""
    raw: dict[str, float] = {}
    minimum = max(float(floor), 1e-12)
    for model_name, value in spearman_by_model.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        raw[str(model_name)] = 1.0 / max(abs(numeric), minimum)
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {model_name: weight / total for model_name, weight in raw.items()}


def pooled_spearman(
    frame: pd.DataFrame,
    *,
    score_column: str = "predicted_alpha",
    target_column: str = "alpha_vs_sector_60d",
) -> float:
    if frame.empty or score_column not in frame.columns or target_column not in frame.columns:
        return float("nan")
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid = scores.notna() & target.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    corr = scores[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


def build_rank_ensemble_predictions(
    predictions: pd.DataFrame,
    *,
    members: tuple[str, ...],
    target_column: str = "alpha_vs_sector_60d",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    required = {"snapshot_date", "ticker", "model_name", "predicted_alpha", target_column}
    if predictions.empty or not required.issubset(predictions.columns) or len(members) < 2:
        return pd.DataFrame()
    scoped = predictions[predictions["model_name"].astype(str).isin(members)].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["snapshot_date"] = pd.to_datetime(scoped["snapshot_date"], errors="coerce").dt.normalize()
    scoped["ticker"] = scoped["ticker"].astype(str).str.strip()
    scoped["model_name"] = scoped["model_name"].astype(str)
    scoped["predicted_alpha"] = pd.to_numeric(scoped["predicted_alpha"], errors="coerce")
    scoped[target_column] = pd.to_numeric(scoped[target_column], errors="coerce")
    scoped = scoped.dropna(subset=["snapshot_date", "ticker", "model_name", "predicted_alpha", target_column])
    if scoped.empty:
        return pd.DataFrame()

    score_frames: list[pd.DataFrame] = []
    for model_name, model_frame in scoped.groupby("model_name", sort=True):
        scored = model_frame[["snapshot_date", "ticker", "predicted_alpha"]].copy()
        scored = scored.rename(columns={"predicted_alpha": str(model_name)})
        score_frames.append(scored)
    if not score_frames:
        return pd.DataFrame()

    target = scoped[["snapshot_date", "ticker", target_column]].drop_duplicates(["snapshot_date", "ticker"])
    merged = target.copy()
    for score_frame in score_frames:
        merged = merged.merge(score_frame, on=["snapshot_date", "ticker"], how="inner")
    available_members = [model for model in members if model in merged.columns]
    if len(available_members) < 2:
        return pd.DataFrame()
    for model_name in available_members:
        merged[model_name] = pd.to_numeric(merged[model_name], errors="coerce").groupby(
            merged["snapshot_date"]
        ).rank(method="average", pct=True)

    if weights is None:
        normalized = {model: 1.0 / len(available_members) for model in available_members}
    else:
        selected = {model: float(weights.get(model, 0.0)) for model in available_members}
        total = sum(weight for weight in selected.values() if math.isfinite(weight) and weight > 0.0)
        if total <= 0.0:
            return pd.DataFrame()
        normalized = {
            model: (weight / total if math.isfinite(weight) and weight > 0.0 else 0.0)
            for model, weight in selected.items()
        }
    merged["predicted_alpha"] = 0.0
    for model_name, weight in normalized.items():
        merged["predicted_alpha"] += pd.to_numeric(merged[model_name], errors="coerce") * weight
    return merged[["snapshot_date", "ticker", target_column, "predicted_alpha"]].copy()


def window_spearman_summary(
    frame: pd.DataFrame,
    *,
    target_column: str = "alpha_vs_sector_60d",
    last_fold_dates: int = 20,
    trailing_fold_dates: int = 60,
) -> dict[str, float | int]:
    if frame.empty:
        return {
            "rows": 0,
            "dates": 0,
            "full_oos_spearman": float("nan"),
            "last_fold_spearman": float("nan"),
            "trailing_3fold_spearman": float("nan"),
        }
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    dates = sorted(working["snapshot_date"].dropna().drop_duplicates().tolist())
    last_dates = set(dates[-max(int(last_fold_dates), 1) :])
    trailing_dates = set(dates[-max(int(trailing_fold_dates), 1) :])
    return {
        "rows": int(len(working.index)),
        "dates": int(len(dates)),
        "full_oos_spearman": pooled_spearman(working, target_column=target_column),
        "last_fold_spearman": pooled_spearman(
            working[working["snapshot_date"].isin(last_dates)],
            target_column=target_column,
        ),
        "trailing_3fold_spearman": pooled_spearman(
            working[working["snapshot_date"].isin(trailing_dates)],
            target_column=target_column,
        ),
    }
