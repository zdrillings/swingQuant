from __future__ import annotations

import math

import pandas as pd


DEFAULT_WEIGHT_FLOOR = 0.001
DEFAULT_ROUND_TRIP_COST = 0.001


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


def build_two_member_rank_blend(
    predictions: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    left_weight: float,
    right_weight: float,
    target_column: str = "alpha_vs_sector_60d",
) -> pd.DataFrame:
    required = {"snapshot_date", "ticker", "model_name", "predicted_alpha", target_column}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    left = _model_rank_frame(
        predictions,
        model_name=left_model,
        rank_column="_left_rank",
        target_column=target_column,
    )
    right = _model_rank_frame(
        predictions,
        model_name=right_model,
        rank_column="_right_rank",
        target_column=target_column,
    )
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(
        right[["snapshot_date", "ticker", "_right_rank"]],
        on=["snapshot_date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()
    total = float(left_weight) + float(right_weight)
    if not math.isfinite(total) or total <= 0.0:
        return pd.DataFrame()
    merged["predicted_alpha"] = (
        (float(left_weight) / total) * pd.to_numeric(merged["_left_rank"], errors="coerce")
        + (float(right_weight) / total) * pd.to_numeric(merged["_right_rank"], errors="coerce")
    )
    merged["model_name"] = f"{left_model}_{right_model}_rank_blend_{left_weight:.2f}_{right_weight:.2f}"
    return merged.drop(columns=["_left_rank", "_right_rank"], errors="ignore").copy()


def acceptance_window_summary(
    frame: pd.DataFrame,
    *,
    target_column: str = "alpha_vs_sector_60d",
    top_n: int = 2,
    fold_size: int = 20,
    trailing_folds: int = 3,
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
) -> dict[str, float | int]:
    return {
        **_prefixed_summary(
            _top_n_basket_summary(
                frame,
                target_column=target_column,
                top_n=top_n,
                round_trip_cost=round_trip_cost,
            ),
            "full_oos",
        ),
        **_prefixed_summary(
            _top_n_basket_summary(
                _last_n_dates(frame, max(int(fold_size), 1)),
                target_column=target_column,
                top_n=top_n,
                round_trip_cost=round_trip_cost,
            ),
            "last_fold",
        ),
        **_prefixed_summary(
            _top_n_basket_summary(
                _last_n_dates(frame, max(int(fold_size), 1) * max(int(trailing_folds), 1)),
                target_column=target_column,
                top_n=top_n,
                round_trip_cost=round_trip_cost,
            ),
            "trailing_3fold",
        ),
    }


def passes_acceptance(summary: dict[str, float | int]) -> bool:
    return (
        _finite_at_least(summary.get("full_oos_spearman"), 0.0)
        and _finite_at_least(summary.get("last_fold_hit_rate_excess"), 0.02)
        and _finite_at_least(summary.get("last_fold_beat_universe_rate"), 0.50)
        and _finite_at_least(summary.get("last_fold_mean_target_excess"), 0.0)
        and _finite_at_least(summary.get("last_fold_spearman"), 0.0)
        and _finite_at_least(summary.get("trailing_3fold_hit_rate_excess"), 0.02)
        and _finite_at_least(summary.get("trailing_3fold_beat_universe_rate"), 0.50)
        and _finite_at_least(summary.get("trailing_3fold_mean_target_excess"), 0.0)
        and _finite_at_least(summary.get("trailing_3fold_spearman"), 0.0)
    )


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


def _model_rank_frame(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    rank_column: str,
    target_column: str,
) -> pd.DataFrame:
    scoped = predictions[predictions["model_name"].astype(str).eq(str(model_name))].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["snapshot_date"] = pd.to_datetime(scoped["snapshot_date"], errors="coerce").dt.normalize()
    scoped["ticker"] = scoped["ticker"].astype(str).str.strip()
    scoped["predicted_alpha"] = pd.to_numeric(scoped["predicted_alpha"], errors="coerce")
    scoped = scoped.dropna(subset=["snapshot_date", "ticker", "predicted_alpha"])
    if scoped.empty:
        return pd.DataFrame()
    scoped[rank_column] = scoped["predicted_alpha"].groupby(scoped["snapshot_date"]).rank(method="average", pct=True)
    columns = ["snapshot_date", "ticker", "model_name", "predicted_alpha", rank_column]
    if target_column in scoped.columns:
        columns.append(target_column)
    return scoped[columns].copy()


def _last_n_dates(frame: pd.DataFrame, date_count: int) -> pd.DataFrame:
    if frame.empty or "snapshot_date" not in frame.columns:
        return frame.iloc[0:0].copy()
    dates = sorted(pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().drop_duplicates().tolist())
    selected = set(dates[-min(max(int(date_count), 1), len(dates)) :])
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    return working[working["snapshot_date"].isin(selected)].copy()


def _top_n_basket_summary(
    frame: pd.DataFrame,
    *,
    target_column: str,
    top_n: int,
    round_trip_cost: float,
) -> dict[str, float | int]:
    if frame.empty or target_column not in frame.columns:
        return _empty_basket_summary()
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.strip()
    working["predicted_alpha"] = pd.to_numeric(working["predicted_alpha"], errors="coerce")
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=["snapshot_date", "ticker", "predicted_alpha", target_column])
    if working.empty:
        return _empty_basket_summary()

    rows: list[dict[str, float]] = []
    for _, day_frame in working.groupby("snapshot_date", sort=True):
        ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
        picks = ordered.head(max(int(top_n), 1)).copy()
        target = pd.to_numeric(picks[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        universe = pd.to_numeric(day_frame[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        if target.empty or universe.empty:
            continue
        score = pd.to_numeric(ordered["predicted_alpha"], errors="coerce")
        full_target = pd.to_numeric(ordered[target_column], errors="coerce")
        valid = score.notna() & full_target.notna()
        spearman = float("nan")
        if int(valid.sum()) >= 3 and score[valid].nunique(dropna=True) > 1 and full_target[valid].nunique(dropna=True) > 1:
            corr = score[valid].corr(full_target[valid], method="spearman")
            if pd.notna(corr) and math.isfinite(float(corr)):
                spearman = float(corr)
        net_target = target - float(round_trip_cost)
        net_universe = universe - float(round_trip_cost)
        rows.append(
            {
                "pick_count": float(len(picks.index)),
                "mean_target": float(net_target.mean()),
                "hit_rate": float((net_target > 0.0).mean()),
                "universe_mean_target": float(net_universe.mean()),
                "universe_hit_rate": float((net_universe > 0.0).mean()),
                "spearman": spearman,
            }
        )
    if not rows:
        return _empty_basket_summary()
    daily = pd.DataFrame(rows)
    return {
        "dates": int(len(daily.index)),
        "avg_pick_count": float(daily["pick_count"].mean()),
        "mean_target_excess": float((daily["mean_target"] - daily["universe_mean_target"]).mean()),
        "hit_rate_excess": float((daily["hit_rate"] - daily["universe_hit_rate"]).mean()),
        "beat_universe_rate": float((daily["mean_target"] > daily["universe_mean_target"]).mean()),
        "spearman": float(daily["spearman"].dropna().mean()) if daily["spearman"].notna().any() else float("nan"),
    }


def _empty_basket_summary() -> dict[str, float | int]:
    return {
        "dates": 0,
        "avg_pick_count": float("nan"),
        "mean_target_excess": float("nan"),
        "hit_rate_excess": float("nan"),
        "beat_universe_rate": float("nan"),
        "spearman": float("nan"),
    }


def _prefixed_summary(summary: dict[str, float | int], prefix: str) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _finite_at_least(value: object, floor: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= float(floor)
