from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from src.research.shortlist_model_service import ShortlistModelService


@dataclass(frozen=True)
class LiveShortlistModelContext:
    generated_at: str
    champion_model: str
    live_snapshot_date: str | None
    live_predictions: pd.DataFrame
    top_n: int


def load_live_shortlist_model_context(
    db_manager,
    *,
    horizon_days: int = 20,
    top_n: int = 10,
    min_train_dates: int = 252,
    test_window_dates: int = 20,
    recent_dates: int = 60,
    refresh_if_stale: bool = True,
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

    runs = db_manager.load_shortlist_model_runs(
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
        runs = db_manager.load_shortlist_model_runs(
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
    selected_model = (
        str(preferred_model_name).strip()
        if preferred_model_name not in (None, "")
        else champion_model
    )
    live_predictions = db_manager.load_shortlist_model_predictions(
        generated_at=generated_at,
        horizon_days=int(horizon_days),
        eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
        model_scope=str(model_scope or "global"),
        dataset_split="live",
        model_name=selected_model,
    )
    if live_predictions.empty and selected_model != champion_model:
        live_predictions = db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=str(eligible_universe_mode or "passed_only"),
            model_scope=str(model_scope or "global"),
            dataset_split="live",
            model_name=champion_model,
        )
        selected_model = champion_model
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
    live_predictions = live_predictions.sort_values(
        ["predicted_alpha", "md_volume_30d", "ticker"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    live_predictions["model_rank"] = range(1, len(live_predictions.index) + 1)
    live_predictions = _annotate_live_prediction_comparisons(live_predictions, top_n=int(top_n))
    return LiveShortlistModelContext(
        generated_at=generated_at,
        champion_model=selected_model,
        live_snapshot_date=str(latest_run["live_snapshot_date"]) if latest_run["live_snapshot_date"] is not None else None,
        live_predictions=live_predictions,
        top_n=int(top_n),
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
