from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, build_rank_augmented_feature_frame, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


TARGET_COLUMN = "alpha_vs_sector_60d"
D2_BASE_FEATURES = [
    "analyst_eps_revision_breadth_change_14d",
    "analyst_eps_estimate_dispersion_change_14d",
]
CANDIDATE_MODELS = ("ridge_model", "lasso_model", "xgboost_model")


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _shortlist_config() -> dict[str, object]:
    payload = load_feature_config().get("scan_policy", {}).get("shortlist_model", {})
    return {
        "eligible_universe_mode": str(payload.get("production_eligible_universe_mode", "passed_or_trend")),
        "label_horizon_dates": int(payload.get("horizon_days", 60)),
        "min_train_dates": int(payload.get("min_train_dates", 252)),
        "max_train_dates": int(payload.get("max_train_dates", payload.get("min_train_dates", 252))),
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", payload.get("test_window_dates", 20))),
        "model_scope": str(payload.get("production_model_scope", "global")),
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
    }


def _read_table_columns(*, duckdb_path: Path, table: str) -> set[str]:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {str(row[1]) for row in rows}


def _read_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
    import duckdb

    safe_columns = ", ".join(columns)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT {safe_columns}
            FROM universe_daily_snapshots
            WHERE {TARGET_COLUMN} IS NOT NULL
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _read_revision_snapshots(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT snapshot_date, ticker, earnings_estimate_json, eps_revisions_json
            FROM analyst_revision_snapshots
            ORDER BY ticker, snapshot_date
            """
        ).fetchdf()


def _read_calendar_dates(*, duckdb_path: Path) -> list[pd.Timestamp]:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
        ).fetchall()
    return sorted(pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna().dt.normalize().tolist())


def _snapshot_columns(available: set[str]) -> list[str]:
    required = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        "regime_green",
        TARGET_COLUMN,
    ]
    optional = [column for column in MODEL_FEATURE_COLUMNS if column in available and column not in D2_BASE_FEATURES]
    return list(dict.fromkeys([*required, *optional]))


def _json_records(value: object) -> list[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [record for record in parsed if isinstance(record, dict)] if isinstance(parsed, list) else []


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _eps_revision_breadth(records: list[dict]) -> float | None:
    up_total = 0.0
    down_total = 0.0
    for record in records:
        for key, value in record.items():
            normalized = str(key).lower()
            numeric = _float_or_none(value)
            if numeric is None:
                continue
            if "up" in normalized:
                up_total += numeric
            elif "down" in normalized:
                down_total += numeric
    total = up_total + down_total
    return (up_total - down_total) / total if total > 0 else None


def _eps_estimate_dispersion(records: list[dict]) -> float | None:
    period_priority = {"0q": 0, "+1q": 1, "0y": 2, "+1y": 3}
    candidates = sorted(
        records,
        key=lambda record: period_priority.get(str(record.get("period", "")).strip().lower(), 99),
    )
    for record in candidates:
        estimate_avg = _float_or_none(record.get("avg"))
        estimate_high = _float_or_none(record.get("high"))
        estimate_low = _float_or_none(record.get("low"))
        if estimate_avg is None or estimate_high is None or estimate_low is None:
            continue
        denominator = abs(estimate_avg)
        if denominator <= 0:
            continue
        return (estimate_high - estimate_low) / denominator
    return None


def _revision_metric_frame(revisions: pd.DataFrame) -> pd.DataFrame:
    if revisions.empty:
        return pd.DataFrame(columns=["snapshot_date", "ticker", "revision_breadth", "estimate_dispersion"])
    working = revisions.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["revision_breadth"] = working["eps_revisions_json"].map(lambda value: _eps_revision_breadth(_json_records(value)))
    working["estimate_dispersion"] = working["earnings_estimate_json"].map(
        lambda value: _eps_estimate_dispersion(_json_records(value))
    )
    return working[["snapshot_date", "ticker", "revision_breadth", "estimate_dispersion"]].dropna(
        subset=["snapshot_date", "ticker"],
    )


def _merge_d2_features(snapshots: pd.DataFrame, revisions: pd.DataFrame) -> pd.DataFrame:
    working = snapshots.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["lag_cutoff_date"] = working["snapshot_date"] - pd.tseries.offsets.BDay(14)
    metrics = _revision_metric_frame(revisions)
    frames: list[pd.DataFrame] = []
    for ticker, snapshot_group in working.groupby("ticker", sort=False):
        left = snapshot_group.sort_values("snapshot_date").copy()
        right = metrics.loc[metrics["ticker"].eq(ticker)].sort_values("snapshot_date").copy()
        if right.empty:
            left["analyst_eps_revision_breadth_change_14d"] = pd.NA
            left["analyst_eps_estimate_dispersion_change_14d"] = pd.NA
            frames.append(left.drop(columns=["lag_cutoff_date"]))
            continue
        current = pd.merge_asof(
            left,
            right.rename(
                columns={
                    "snapshot_date": "revision_snapshot_date",
                    "revision_breadth": "current_revision_breadth",
                    "estimate_dispersion": "current_estimate_dispersion",
                }
            ).drop(columns=["ticker"]),
            left_on="snapshot_date",
            right_on="revision_snapshot_date",
            direction="backward",
        )
        lagged = pd.merge_asof(
            current.sort_values("lag_cutoff_date"),
            right.rename(
                columns={
                    "snapshot_date": "lagged_revision_snapshot_date",
                    "revision_breadth": "lagged_revision_breadth",
                    "estimate_dispersion": "lagged_estimate_dispersion",
                }
            ).drop(columns=["ticker"]),
            left_on="lag_cutoff_date",
            right_on="lagged_revision_snapshot_date",
            direction="backward",
        ).sort_values("snapshot_date")
        lagged["analyst_eps_revision_breadth_change_14d"] = (
            pd.to_numeric(lagged["current_revision_breadth"], errors="coerce")
            - pd.to_numeric(lagged["lagged_revision_breadth"], errors="coerce")
        )
        lagged["analyst_eps_estimate_dispersion_change_14d"] = (
            pd.to_numeric(lagged["current_estimate_dispersion"], errors="coerce")
            - pd.to_numeric(lagged["lagged_estimate_dispersion"], errors="coerce")
        )
        frames.append(
            lagged.drop(
                columns=[
                    "lag_cutoff_date",
                    "revision_snapshot_date",
                    "lagged_revision_snapshot_date",
                    "current_revision_breadth",
                    "current_estimate_dispersion",
                    "lagged_revision_breadth",
                    "lagged_estimate_dispersion",
                ]
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else working.drop(columns=["lag_cutoff_date"])


def _feature_candidates() -> list[str]:
    return [feature for feature in expand_model_feature_columns(D2_BASE_FEATURES) if feature.split("__")[0] in D2_BASE_FEATURES]


def _fold_ic_rows(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    config: dict[str, object],
    min_feature_ic: float,
    observation_fraction: float,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.normalize().drop_duplicates().tolist())
    start_index = int(config["min_train_dates"])
    stride = max(int(config["evaluation_stride_dates"]), 1)
    horizon = int(config["label_horizon_dates"])
    rows: list[dict[str, object]] = []
    while start_index < len(dates):
        test_dates = dates[start_index : start_index + int(config["test_window_dates"])]
        train_pool_dates = dates[: max(0, start_index - horizon)]
        if len(train_pool_dates) < int(config["min_train_dates"]):
            start_index += stride
            continue
        train_frame = frame[frame["snapshot_date"].isin(set(train_pool_dates))].copy()
        train_frame = service._stride_training_labels(
            train_frame,
            dates=dates,
            anchor_index=start_index,
            label_horizon_dates=horizon,
        )
        if train_frame.empty:
            start_index += stride
            continue
        feature_frame = train_frame[["snapshot_date", "sector", *D2_BASE_FEATURES]].copy()
        feature_frame, _ = build_rank_augmented_feature_frame(feature_frame)
        target = pd.to_numeric(train_frame[TARGET_COLUMN], errors="coerce")
        min_observations = service._feature_ic_min_observations(
            len(train_frame.index),
            min_observation_fraction=observation_fraction,
        )
        for feature_name in _feature_candidates():
            values = pd.to_numeric(feature_frame[feature_name], errors="coerce")
            valid = values.notna() & target.notna()
            observations = int(valid.sum())
            ic = float("nan")
            cleared = False
            if observations >= min_observations and values[valid].nunique(dropna=True) >= 2 and target[valid].nunique(dropna=True) >= 2:
                corr = values[valid].corr(target[valid], method="spearman")
                if pd.notna(corr) and math.isfinite(float(corr)):
                    ic = float(corr)
                    cleared = abs(ic) >= float(min_feature_ic)
            rows.append(
                {
                    "fold_start": pd.Timestamp(test_dates[0]).date().isoformat() if test_dates else "n/a",
                    "feature": feature_name,
                    "train_rows": len(train_frame.index),
                    "observations": observations,
                    "min_observations": min_observations,
                    "ic": ic,
                    "cleared": bool(cleared),
                }
            )
        start_index += stride
    return pd.DataFrame(rows)


def _ic_summary(fold_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_rows.empty:
        return pd.DataFrame(columns=["feature", "folds", "mean_ic", "mean_abs_ic", "max_abs_ic", "clear_folds", "clear_rate"])
    working = fold_rows.copy()
    working["abs_ic"] = pd.to_numeric(working["ic"], errors="coerce").abs()
    summary = (
        working.groupby("feature", sort=False)
        .agg(
            folds=("fold_start", "nunique"),
            mean_ic=("ic", "mean"),
            mean_abs_ic=("abs_ic", "mean"),
            max_abs_ic=("abs_ic", "max"),
            clear_folds=("cleared", "sum"),
            clear_rate=("cleared", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["mean_abs_ic", "max_abs_ic", "feature"], ascending=[False, False, True]).reset_index(drop=True)


def _pooled_spearman(frame: pd.DataFrame) -> float:
    score = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = score.notna() & target.notna()
    if int(valid.sum()) < 3 or score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
        return float("nan")
    corr = score[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _run_model(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    config: dict[str, object],
) -> pd.DataFrame:
    xgboost_params = None
    if model_name == "xgboost_model":
        xgboost_params = {**service._xgboost_params_for_config(str(config["xgboost_config"])), "n_jobs": 1}
    predictions = service._walk_forward_predictions(
        eligible,
        target_column=TARGET_COLUMN,
        evaluation_target_column=TARGET_COLUMN,
        model_name=model_name,
        min_train_dates=int(config["min_train_dates"]),
        max_train_dates=None if config["max_train_dates"] in (None, "") else int(config["max_train_dates"]),
        test_window_dates=int(config["test_window_dates"]),
        evaluation_stride_dates=int(config["evaluation_stride_dates"]),
        label_horizon_dates=int(config["label_horizon_dates"]),
        model_scope=str(config["model_scope"]),
        xgboost_params=xgboost_params,
        feature_columns_override=feature_columns,
        min_feature_ic=service._load_min_feature_ic(),
        min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        regime_matching_mode="off",
        min_regime_train_dates=service._load_min_regime_train_dates(),
        regime_transition_purge_mode="off",
    )
    return predictions if predictions is not None else pd.DataFrame()


def _bakeoff_rows(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    baseline_features = service._filter_model_feature_columns(
        [feature for feature in expand_model_feature_columns(MODEL_FEATURE_COLUMNS) if feature.split("__")[0] not in D2_BASE_FEATURES]
    )
    with_d2_features = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_MODELS:
        dense_without = _run_model(service, eligible, model_name=model_name, feature_columns=baseline_features, config=config)
        dense_with = _run_model(service, eligible, model_name=model_name, feature_columns=with_d2_features, config=config)
        honest_without = service._non_overlapping_oos_predictions(
            dense_without,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        honest_with = service._non_overlapping_oos_predictions(
            dense_with,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        without_s = _pooled_spearman(honest_without)
        with_s = _pooled_spearman(honest_with)
        rows.append(
            {
                "model": model_name,
                "without_d2_honest_rows": len(honest_without.index),
                "with_d2_honest_rows": len(honest_with.index),
                "without_d2_full_oos_spearman": without_s,
                "with_d2_full_oos_spearman": with_s,
                "delta_with_minus_without": with_s - without_s if math.isfinite(with_s) and math.isfinite(without_s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _render_report(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    fold_rows: pd.DataFrame,
    bakeoff: pd.DataFrame | None,
    eligible_rows: int,
    eligible_dates: int,
    feature_coverage_rows: int,
    first_feature_date: str,
    analyst_history_start: str,
    first_possible_lagged_date: str,
    latest_eligible_date: str,
    min_feature_ic: float,
    observation_fraction: float,
) -> None:
    surviving = summary[summary["mean_abs_ic"].ge(float(min_feature_ic))]["feature"].astype(str).tolist() if not summary.empty else []
    finite_summary = summary["mean_abs_ic"].dropna() if "mean_abs_ic" in summary.columns else pd.Series(dtype=float)
    if surviving:
        screen_verdict = "survived"
    elif finite_summary.empty:
        screen_verdict = "insufficient_matured_coverage"
    else:
        screen_verdict = "failed"
    lines = [
        "# D2 Revision Acceleration Audit",
        "",
        "- idea: 14-session change in analyst EPS revision breadth and EPS estimate dispersion",
        "- data_access: DuckDB read_only=True; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        "- lag_policy: latest analyst revision snapshot at or before snapshot_date minus 14 business sessions",
        "- dispersion_source: earnings_estimate_json high/low/avg, priority period 0q then +1q then 0y then +1y",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- eligible_rows_with_any_d2_feature: {feature_coverage_rows}",
        f"- first_eligible_d2_feature_date: {first_feature_date}",
        f"- analyst_revision_history_start: {analyst_history_start}",
        f"- first_possible_14_session_delta_date: {first_possible_lagged_date}",
        f"- latest_matured_eligible_date: {latest_eligible_date}",
        f"- min_feature_ic: {float(min_feature_ic):.4f}",
        f"- min_feature_ic_observation_fraction: {float(observation_fraction):.4f}",
        f"- screen_verdict: {screen_verdict}",
        f"- surviving_features_by_mean_abs_ic: {', '.join(surviving) if surviving else 'none'}",
        "",
        "## Fold-Local IC Summary",
        "",
        "| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {int(row.folds)} | {_fmt(row.mean_ic)} | {_fmt(row.mean_abs_ic)} | "
            f"{_fmt(row.max_abs_ic)} | {int(row.clear_folds)} | {_fmt(row.clear_rate)} |"
        )
    lines.extend(["", "## Fold Details", "", "| fold_start | feature | observations | min_observations | ic | cleared |", "|---|---|---:|---:|---:|---|"])
    for row in fold_rows.itertuples(index=False):
        lines.append(
            f"| {row.fold_start} | {row.feature} | {int(row.observations)} | {int(row.min_observations)} | "
            f"{_fmt(row.ic)} | {bool(row.cleared)} |"
        )
    lines.append("")
    lines.append("## Honest-Grid With/Without Bakeoff")
    lines.append("")
    if bakeoff is None:
        if screen_verdict == "insufficient_matured_coverage":
            lines.append(
                "Skipped: the current fold-local training windows have no D2 observations above the configured "
                "20% observation floor. Re-run after enough post-2026-06-23 analyst history has matured into "
                "60d labels."
            )
        else:
            lines.append("Skipped: no D2 feature reached mean_abs_ic >= min_feature_ic, so D2 is falsified at the IC screen.")
    elif bakeoff.empty:
        lines.append("No bakeoff rows produced.")
    else:
        lines.extend(
            [
                "| model | without_d2_rows | with_d2_rows | without_d2_full_oos_spearman | with_d2_full_oos_spearman | delta_with_minus_without |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bakeoff.itertuples(index=False):
            lines.append(
                f"| {row.model} | {int(row.without_d2_honest_rows)} | {int(row.with_d2_honest_rows)} | "
                f"{_fmt(row.without_d2_full_oos_spearman)} | {_fmt(row.with_d2_full_oos_spearman)} | {_fmt(row.delta_with_minus_without)} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    available = _read_table_columns(duckdb_path=settings.paths.duckdb_path, table="universe_daily_snapshots")
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path, columns=_snapshot_columns(available))
    revisions = _read_revision_snapshots(duckdb_path=settings.paths.duckdb_path)
    frame = _merge_d2_features(service._prepare_snapshot_frame(snapshots), revisions)
    eligible = service._build_matured_eligible_universe(
        frame,
        target_column=TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    min_feature_ic = service._load_min_feature_ic()
    observation_fraction = service._load_min_feature_ic_observation_fraction()
    fold_rows = _fold_ic_rows(
        service,
        eligible,
        config=config,
        min_feature_ic=min_feature_ic,
        observation_fraction=observation_fraction,
    )
    summary = _ic_summary(fold_rows)
    feature_mask = eligible[D2_BASE_FEATURES].notna().any(axis=1) if not eligible.empty else pd.Series(dtype=bool)
    feature_dates = pd.to_datetime(eligible.loc[feature_mask, "snapshot_date"], errors="coerce").dropna()
    analyst_dates = pd.to_datetime(revisions["snapshot_date"], errors="coerce").dropna().dt.normalize()
    analyst_start = analyst_dates.min() if not analyst_dates.empty else pd.NaT
    first_possible = analyst_start + pd.tseries.offsets.BDay(14) if pd.notna(analyst_start) else pd.NaT
    eligible_dates = pd.to_datetime(eligible["snapshot_date"], errors="coerce").dropna() if not eligible.empty else pd.Series(dtype="datetime64[ns]")
    bakeoff = None
    if not summary.empty and bool(summary["mean_abs_ic"].ge(float(min_feature_ic)).any()):
        calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
        bakeoff = _bakeoff_rows(service, eligible, config=config, calendar_dates=calendar_dates)
    _render_report(
        output_path=output_path,
        summary=summary,
        fold_rows=fold_rows,
        bakeoff=bakeoff,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        feature_coverage_rows=int(feature_mask.sum()) if not eligible.empty else 0,
        first_feature_date=pd.Timestamp(feature_dates.min()).date().isoformat() if not feature_dates.empty else "n/a",
        analyst_history_start=pd.Timestamp(analyst_start).date().isoformat() if pd.notna(analyst_start) else "n/a",
        first_possible_lagged_date=pd.Timestamp(first_possible).date().isoformat() if pd.notna(first_possible) else "n/a",
        latest_eligible_date=pd.Timestamp(eligible_dates.max()).date().isoformat() if not eligible_dates.empty else "n/a",
        min_feature_ic=min_feature_ic,
        observation_fraction=observation_fraction,
    )
    return {
        "output_path": str(output_path),
        "eligible_rows": len(eligible.index),
        "eligible_dates": int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        "survivors": int(summary["mean_abs_ic"].ge(float(min_feature_ic)).sum()) if not summary.empty else 0,
        "best_mean_abs_ic": float(summary["mean_abs_ic"].max()) if not summary.empty else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit D2 analyst revision acceleration features.")
    parser.add_argument("--output", type=Path, default=Path("reports/d2_revision_acceleration.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} eligible_dates={eligible_dates} survivors={survivors} best_mean_abs_ic={best_mean_abs_ic:+.4f}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
