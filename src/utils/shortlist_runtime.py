from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService
from src.settings import load_feature_config

CONFIDENCE_BASKET_SIZE = 2


@dataclass(frozen=True)
class LiveShortlistModelContext:
    generated_at: str
    champion_model: str
    live_snapshot_date: str | None
    live_predictions: pd.DataFrame
    top_n: int
    recent_20d_beat_rate: float | None = None
    recent_20d_mean_target: float | None = None
    recent_20d_hit_rate: float | None = None
    recent_60d_beat_rate: float | None = None
    recent_60d_mean_target: float | None = None
    recent_60d_hit_rate: float | None = None
    recent_1fold_beat_rate: float | None = None
    recent_1fold_mean_target: float | None = None
    recent_1fold_hit_rate: float | None = None
    recent_3fold_beat_rate: float | None = None
    recent_3fold_mean_target: float | None = None
    recent_3fold_hit_rate: float | None = None


def load_live_shortlist_model_context(
    db_manager,
    *,
    horizon_days: int = 20,
    top_n: int = 10,
    min_train_dates: int = 252,
    test_window_dates: int = 20,
    recent_dates: int = 60,
    refresh_if_stale: bool = True,
    allow_refresh: bool = False,
    preferred_model_name: str | None = None,
    eligible_universe_mode: str = "passed_only",
    model_scope: str = "global",
    xgboost_config: str = "baseline",
) -> LiveShortlistModelContext | None:
    required_methods = (
        "load_shortlist_model_runs",
        "load_shortlist_model_predictions",
        "list_universe_daily_snapshot_dates",
    )
    if not all(hasattr(db_manager, name) for name in required_methods):
        return None

    runs = _load_shortlist_model_runs(
        db_manager,
        horizon_days=int(horizon_days),
        eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
        model_scope=str(model_scope or "global"),
        xgboost_config=str(xgboost_config or "baseline"),
        limit=1,
    )
    latest_snapshot_dates = db_manager.list_universe_daily_snapshot_dates()
    latest_snapshot_date = latest_snapshot_dates[-1] if latest_snapshot_dates else None
    needs_refresh = runs.empty
    if not needs_refresh and refresh_if_stale and latest_snapshot_date is not None:
        run_snapshot_date = runs.iloc[0]["live_snapshot_date"]
        needs_refresh = str(run_snapshot_date or "") != str(latest_snapshot_date)
    if needs_refresh:
        if not allow_refresh:
            return None
        ShortlistModelService(db_manager).run(
            top_n=int(top_n),
            horizon_days=int(horizon_days),
            min_train_dates=int(min_train_dates),
            test_window_dates=int(test_window_dates),
            recent_dates=int(recent_dates),
            eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
            model_scope=str(model_scope or "global"),
            xgboost_config=str(xgboost_config or "baseline"),
        )
        runs = _load_shortlist_model_runs(
            db_manager,
            horizon_days=int(horizon_days),
            eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
            model_scope=str(model_scope or "global"),
            xgboost_config=str(xgboost_config or "baseline"),
            limit=1,
        )
        if runs.empty:
            return None

    latest_run = runs.iloc[0]
    generated_at = str(latest_run["generated_at"])
    champion_model = str(latest_run["champion_model"])
    selected_model = str(preferred_model_name).strip() if preferred_model_name not in (None, "") else champion_model
    if not selected_model:
        selected_model = champion_model
    live_predictions = db_manager.load_shortlist_model_predictions(
        generated_at=generated_at,
        horizon_days=int(horizon_days),
        eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
        model_scope=str(model_scope or "global"),
        dataset_split="live",
        model_name=selected_model,
    )
    if live_predictions.empty:
        return None
    live_predictions = live_predictions.copy()
    live_predictions["snapshot_date"] = pd.to_datetime(live_predictions["snapshot_date"]).dt.normalize()
    live_predictions["predicted_alpha"] = pd.to_numeric(live_predictions["predicted_alpha"], errors="coerce")
    live_predictions["md_volume_30d"] = pd.to_numeric(live_predictions["md_volume_30d"], errors="coerce")
    if "details_json" in live_predictions.columns:
        details = live_predictions["details_json"].apply(_parse_prediction_details)
        live_predictions["model_top_reasons"] = details.apply(lambda payload: payload.get("model_top_reasons", []))
        live_predictions["model_reason_summary"] = details.apply(lambda payload: payload.get("model_reason_summary"))
        live_predictions["calibrated_p_beat_sector"] = pd.to_numeric(
            details.apply(lambda payload: payload.get("calibrated_p_beat_sector")),
            errors="coerce",
        )
    if "calibrated_p_beat_sector" not in live_predictions.columns:
        live_predictions["calibrated_p_beat_sector"] = np.nan
    live_predictions = live_predictions.sort_values(
        ["predicted_alpha", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)
    live_predictions["model_rank"] = range(1, len(live_predictions.index) + 1)
    live_predictions = _annotate_live_prediction_comparisons(live_predictions, top_n=int(top_n))
    recent_metrics = {
        20: {"beat_rate": None, "mean_target": None, "hit_rate": None},
        60: {"beat_rate": None, "mean_target": None, "hit_rate": None},
        1: {"beat_rate": None, "mean_target": None, "hit_rate": None},
        3: {"beat_rate": None, "mean_target": None, "hit_rate": None},
    }
    try:
        oos_predictions = db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
            model_scope=str(model_scope or "global"),
            dataset_split="oos",
            model_name=selected_model,
        )
        if not oos_predictions.empty:
            oos_predictions["snapshot_date"] = pd.to_datetime(oos_predictions["snapshot_date"]).dt.normalize()
            if "details_json" in oos_predictions.columns:
                oos_details = oos_predictions["details_json"].apply(_parse_prediction_details)
                oos_predictions["calibrated_p_beat_sector"] = pd.to_numeric(
                    oos_details.apply(lambda payload: payload.get("calibrated_p_beat_sector")),
                    errors="coerce",
                )
            oos_dates = sorted(oos_predictions["snapshot_date"].drop_duplicates().tolist())
            for window in (20, 60, 1, 3):
                recent_oos_dates = oos_dates[-window:] if len(oos_dates) >= window else oos_dates
                recent = oos_predictions[oos_predictions["snapshot_date"].isin(recent_oos_dates)].copy()
                recent_metrics[window] = _score_recent_oos_basket(recent)
    except Exception:
        pass
    if not _passes_runtime_promotion_gate(
        recent_metrics=recent_metrics,
    ):
        return None

    return LiveShortlistModelContext(
        generated_at=generated_at,
        champion_model=selected_model,
        live_snapshot_date=str(latest_run["live_snapshot_date"]) if latest_run["live_snapshot_date"] is not None else None,
        live_predictions=live_predictions,
        top_n=int(top_n),
        recent_20d_beat_rate=recent_metrics[20]["beat_rate"],
        recent_20d_mean_target=recent_metrics[20]["mean_target"],
        recent_20d_hit_rate=recent_metrics[20]["hit_rate"],
        recent_60d_beat_rate=recent_metrics[60]["beat_rate"],
        recent_60d_mean_target=recent_metrics[60]["mean_target"],
        recent_60d_hit_rate=recent_metrics[60]["hit_rate"],
        recent_1fold_beat_rate=recent_metrics[1]["beat_rate"],
        recent_1fold_mean_target=recent_metrics[1]["mean_target"],
        recent_1fold_hit_rate=recent_metrics[1]["hit_rate"],
        recent_3fold_beat_rate=recent_metrics[3]["beat_rate"],
        recent_3fold_mean_target=recent_metrics[3]["mean_target"],
        recent_3fold_hit_rate=recent_metrics[3]["hit_rate"],
    )


def _score_recent_oos_basket(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {"beat_rate": None, "mean_target": None, "hit_rate": None}
    daily_means = []
    daily_universe = []
    pick_actuals = []
    for _snap_date, day_frame in frame.groupby("snapshot_date", sort=True):
        sort_columns = ["predicted_alpha"]
        ascending = [False]
        if "calibrated_p_beat_sector" in day_frame.columns and day_frame["calibrated_p_beat_sector"].notna().any():
            sort_columns = ["calibrated_p_beat_sector", "predicted_alpha"]
            ascending = [False, False]
        ordered = day_frame.sort_values(sort_columns, ascending=ascending)
        picks = ordered.head(CONFIDENCE_BASKET_SIZE)
        actual = pd.to_numeric(picks["actual_alpha_vs_sector"], errors="coerce").dropna()
        universe = pd.to_numeric(day_frame["actual_alpha_vs_sector"], errors="coerce").dropna()
        if not actual.empty and not universe.empty:
            daily_means.append(float(actual.mean()))
            daily_universe.append(float(universe.mean()))
            pick_actuals.extend(float(value) for value in actual.tolist())
    beat_rate = None
    mean_target = None
    hit_rate = None
    if daily_means:
        mean_target = float(np.mean(daily_means))
        beats = sum(1 for picks_mean, universe_mean in zip(daily_means, daily_universe) if picks_mean > universe_mean)
        beat_rate = beats / len(daily_means)
    if pick_actuals:
        hit_rate = sum(1 for value in pick_actuals if value > 0) / len(pick_actuals)
    return {"beat_rate": beat_rate, "mean_target": mean_target, "hit_rate": hit_rate}


def _runtime_promotion_gate() -> dict[str, float | bool]:
    config = load_feature_config()
    payload = (
        config.get("scan_policy", {})
        .get("shortlist_model", {})
        .get("promotion_gate", {})
        if isinstance(config, dict)
        else {}
    )
    return {
        "enabled": bool(payload.get("enabled", True)),
        "min_recent_20d_hit_rate": float(payload.get("min_recent_20d_hit_rate", 0.50)),
        "min_recent_20d_beat_universe_rate": float(payload.get("min_recent_20d_beat_universe_rate", 0.50)),
        "min_recent_20d_mean_target": float(payload.get("min_recent_20d_mean_target", 0.0)),
        "min_recent_60d_hit_rate": float(payload.get("min_recent_60d_hit_rate", 0.50)),
        "min_recent_60d_beat_universe_rate": float(payload.get("min_recent_60d_beat_universe_rate", 0.50)),
        "min_recent_60d_mean_target": float(payload.get("min_recent_60d_mean_target", 0.0)),
        "min_recent_1fold_hit_rate": float(payload.get("min_recent_1fold_hit_rate", 0.50)),
        "min_recent_1fold_beat_universe_rate": float(payload.get("min_recent_1fold_beat_universe_rate", 0.50)),
        "min_recent_1fold_mean_target": float(payload.get("min_recent_1fold_mean_target", 0.0)),
        "min_recent_3fold_hit_rate": float(payload.get("min_recent_3fold_hit_rate", 0.50)),
        "min_recent_3fold_beat_universe_rate": float(payload.get("min_recent_3fold_beat_universe_rate", 0.50)),
        "min_recent_3fold_mean_target": float(payload.get("min_recent_3fold_mean_target", 0.0)),
    }


def _passes_runtime_promotion_gate(
    *,
    recent_metrics: dict[int, dict[str, float | None]],
) -> bool:
    gate = _runtime_promotion_gate()
    if not bool(gate.get("enabled", True)):
        return True
    for window in (20, 60):
        metrics = recent_metrics.get(window, {})
        checks = (
            (metrics.get("hit_rate"), gate[f"min_recent_{window}d_hit_rate"]),
            (metrics.get("beat_rate"), gate[f"min_recent_{window}d_beat_universe_rate"]),
            (metrics.get("mean_target"), gate[f"min_recent_{window}d_mean_target"]),
        )
        for value, threshold in checks:
            try:
                numeric = float(value)
                required = float(threshold)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(numeric) or numeric < required:
                return False
    for folds in (1, 3):
        metrics = recent_metrics.get(folds, {})
        checks = (
            (metrics.get("hit_rate"), gate[f"min_recent_{folds}fold_hit_rate"]),
            (metrics.get("beat_rate"), gate[f"min_recent_{folds}fold_beat_universe_rate"]),
            (metrics.get("mean_target"), gate[f"min_recent_{folds}fold_mean_target"]),
        )
        for value, threshold in checks:
            try:
                numeric = float(value)
                required = float(threshold)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(numeric) or numeric < required:
                return False
    return True


def _load_shortlist_model_runs(
    db_manager,
    *,
    horizon_days: int,
    eligible_universe_mode: str,
    model_scope: str,
    xgboost_config: str,
    limit: int,
) -> pd.DataFrame:
    try:
        return db_manager.load_shortlist_model_runs(
            horizon_days=horizon_days,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            xgboost_config=xgboost_config,
            limit=limit,
        )
    except TypeError:
        return db_manager.load_shortlist_model_runs(
            horizon_days=horizon_days,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            limit=limit,
        )


def _parse_prediction_details(value) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _annotate_live_prediction_comparisons(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    if "model_top_reasons" not in working.columns:
        working["model_top_reasons"] = [[] for _ in range(len(working.index))]
    working["model_comparison_summary"] = None
    selected = working.loc[working["model_rank"] <= int(top_n)].copy()
    excluded = working.loc[working["model_rank"] > int(top_n)].copy()
    if excluded.empty:
        return working
    global_cutoff = excluded.iloc[0]
    for selected_index, selected_row in selected.iterrows():
        sector_matches = excluded.loc[excluded["sector"] == selected_row["sector"]]
        comparator = sector_matches.iloc[0] if not sector_matches.empty else global_cutoff
        working.at[selected_index, "model_comparison_summary"] = _build_comparison_summary(
            selected_row=selected_row,
            comparator_row=comparator,
            same_sector=not sector_matches.empty,
        )
    return working


def _build_comparison_summary(*, selected_row, comparator_row, same_sector: bool) -> str | None:
    selected_reasons = _ensure_reason_list(selected_row.get("model_top_reasons"))
    comparator_reasons = _ensure_reason_list(comparator_row.get("model_top_reasons"))
    differentiators = [reason for reason in selected_reasons if reason not in comparator_reasons]
    comparator_ticker = str(comparator_row.get("ticker"))
    comparator_sector = str(comparator_row.get("sector"))
    if same_sector:
        prefix = f"{comparator_ticker} in {comparator_sector}"
    else:
        prefix = f"next-ranked {comparator_ticker}"
    if differentiators:
        return f"{prefix} on {' and '.join(differentiators[:2])}"
    return prefix


def _ensure_reason_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    return []
