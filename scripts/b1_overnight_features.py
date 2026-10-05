from __future__ import annotations

import argparse
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
from src.utils.feature_engineering import add_overnight_rth_return_features


TARGET_COLUMN = "alpha_vs_sector_60d"
B1_BASE_FEATURES = [
    "overnight_ret_5d",
    "rth_ret_5d",
    "overnight_minus_rth_5d",
    "overnight_ret_20d",
    "rth_ret_20d",
    "overnight_minus_rth_20d",
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


def _read_price_history(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT ticker, date, open, high, low, close, volume, adj_close
            FROM historical_ohlcv
            ORDER BY ticker, date
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
    optional = [column for column in MODEL_FEATURE_COLUMNS if column in available and column not in B1_BASE_FEATURES]
    return list(dict.fromkeys([*required, *optional]))


def _b1_feature_frame(price_history: pd.DataFrame) -> pd.DataFrame:
    working = price_history.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    add_overnight_rth_return_features(working, windows=(5, 20))
    return working.rename(columns={"date": "snapshot_date"})[["snapshot_date", "ticker", *B1_BASE_FEATURES]].copy()


def _merge_b1_features(snapshots: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    working = snapshots.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    features = _b1_feature_frame(price_history)
    features["ticker"] = features["ticker"].astype(str).str.upper()
    return working.merge(features, on=["snapshot_date", "ticker"], how="left")


def _feature_candidates() -> list[str]:
    return [feature for feature in expand_model_feature_columns(B1_BASE_FEATURES) if feature.split("__")[0] in B1_BASE_FEATURES]


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
        feature_frame = train_frame[["snapshot_date", "sector", *B1_BASE_FEATURES]].copy()
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
        [feature for feature in expand_model_feature_columns(MODEL_FEATURE_COLUMNS) if feature.split("__")[0] not in B1_BASE_FEATURES]
    )
    with_b1_features = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_MODELS:
        dense_without = _run_model(service, eligible, model_name=model_name, feature_columns=baseline_features, config=config)
        dense_with = _run_model(service, eligible, model_name=model_name, feature_columns=with_b1_features, config=config)
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
                "without_b1_honest_rows": len(honest_without.index),
                "with_b1_honest_rows": len(honest_with.index),
                "without_b1_full_oos_spearman": without_s,
                "with_b1_full_oos_spearman": with_s,
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
    min_feature_ic: float,
    observation_fraction: float,
) -> None:
    surviving = summary[summary["mean_abs_ic"].ge(float(min_feature_ic))]["feature"].astype(str).tolist() if not summary.empty else []
    lines = [
        "# B1 Overnight/RTH Feature Audit",
        "",
        "- idea: split daily returns into overnight (prior close -> open) and RTH (open -> close), rolled 5d/20d",
        "- data_access: DuckDB read_only=True; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- min_feature_ic: {float(min_feature_ic):.4f}",
        f"- min_feature_ic_observation_fraction: {float(observation_fraction):.4f}",
        f"- screen_verdict: {'survived' if surviving else 'failed'}",
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
        lines.append("Skipped: no B1 feature reached mean_abs_ic >= min_feature_ic, so B1 is falsified at the IC screen.")
    elif bakeoff.empty:
        lines.append("No bakeoff rows produced.")
    else:
        lines.extend(
            [
                "| model | without_b1_rows | with_b1_rows | without_b1_full_oos_spearman | with_b1_full_oos_spearman | delta_with_minus_without |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bakeoff.itertuples(index=False):
            lines.append(
                f"| {row.model} | {int(row.without_b1_honest_rows)} | {int(row.with_b1_honest_rows)} | "
                f"{_fmt(row.without_b1_full_oos_spearman)} | {_fmt(row.with_b1_full_oos_spearman)} | {_fmt(row.delta_with_minus_without)} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    available = _read_table_columns(duckdb_path=settings.paths.duckdb_path, table="universe_daily_snapshots")
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path, columns=_snapshot_columns(available))
    price_history = _read_price_history(duckdb_path=settings.paths.duckdb_path)
    frame = _merge_b1_features(service._prepare_snapshot_frame(snapshots), price_history)
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
    parser = argparse.ArgumentParser(description="Audit B1 overnight/RTH return decomposition features.")
    parser.add_argument("--output", type=Path, default=Path("reports/b1_overnight_features.md"))
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
