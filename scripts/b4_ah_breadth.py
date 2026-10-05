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

from scripts.b1_overnight_features import (
    TARGET_COLUMN,
    _fmt,
    _pooled_spearman,
    _read_calendar_dates,
    _read_snapshots,
    _read_table_columns,
    _run_model,
    _shortlist_config,
)
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, build_rank_augmented_feature_frame, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings
from src.utils.db_manager import DatabaseManager
from src.utils.feature_engineering import compute_ah_breadth_features


B4_BASE_FEATURES = ["ah_breadth_pct_pos", "ah_breadth_zscore_5d"]
CANDIDATE_MODELS = ("ridge_model", "lasso_model", "xgboost_model")


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
    optional = [column for column in MODEL_FEATURE_COLUMNS if column in available and column not in B4_BASE_FEATURES]
    return list(dict.fromkeys([*required, *optional]))


def _read_ah_history(*, ah_duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    if not ah_duckdb_path.exists():
        return pd.DataFrame(columns=["snapshot_date", "ticker", "ah_price", "rth_close"])
    with duckdb.connect(str(ah_duckdb_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT snapshot_date, ticker, ah_price, rth_close
            FROM ah_snapshot_history
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _merge_b4_features(snapshots: pd.DataFrame, ah_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = snapshots.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    features = compute_ah_breadth_features(ah_history)
    if features.empty:
        for column in B4_BASE_FEATURES:
            working[column] = pd.NA
        return working, features
    merged = working.merge(features, on="snapshot_date", how="left")
    return merged, features


def _feature_candidates() -> list[str]:
    return [feature for feature in expand_model_feature_columns(B4_BASE_FEATURES) if feature.split("__")[0] in B4_BASE_FEATURES]


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
        feature_frame = train_frame[["snapshot_date", "sector", *B4_BASE_FEATURES]].copy()
        feature_frame, _ = build_rank_augmented_feature_frame(feature_frame)
        target = pd.to_numeric(train_frame[TARGET_COLUMN], errors="coerce")
        min_observations = service._feature_ic_min_observations(
            len(train_frame.index),
            min_observation_fraction=observation_fraction,
        )
        covered_mask = train_frame[B4_BASE_FEATURES].notna().any(axis=1)
        covered_dates = train_frame.loc[covered_mask, "snapshot_date"].nunique()
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
                    "covered_rows_any_b4": int(covered_mask.sum()),
                    "covered_dates_any_b4": int(covered_dates),
                    "coverage_rate_any_b4": float(covered_mask.mean()) if len(train_frame.index) else float("nan"),
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


def _bakeoff_rows(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
    covered_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    baseline_features = service._filter_model_feature_columns(
        [feature for feature in expand_model_feature_columns(MODEL_FEATURE_COLUMNS) if feature.split("__")[0] not in B4_BASE_FEATURES]
    )
    with_b4_features = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_MODELS:
        dense_without = _run_model(service, eligible, model_name=model_name, feature_columns=baseline_features, config=config)
        dense_with = _run_model(service, eligible, model_name=model_name, feature_columns=with_b4_features, config=config)
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
        honest_without = honest_without[honest_without["snapshot_date"].isin(covered_dates)].copy()
        honest_with = honest_with[honest_with["snapshot_date"].isin(covered_dates)].copy()
        without_s = _pooled_spearman(honest_without)
        with_s = _pooled_spearman(honest_with)
        rows.append(
            {
                "model": model_name,
                "without_b4_honest_covered_rows": len(honest_without.index),
                "with_b4_honest_covered_rows": len(honest_with.index),
                "without_b4_full_oos_spearman": without_s,
                "with_b4_full_oos_spearman": with_s,
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
    ah_feature_dates: int,
    matured_covered_rows: int,
    matured_covered_dates: int,
    min_feature_ic: float,
    observation_fraction: float,
) -> None:
    surviving = summary[summary["mean_abs_ic"].ge(float(min_feature_ic))]["feature"].astype(str).tolist() if not summary.empty else []
    lines = [
        "# B4 AH Breadth Feature Audit",
        "",
        "- idea: market-level after-hours breadth joined to every snapshot row by date",
        "- features: ah_breadth_pct_pos, ah_breadth_zscore_5d",
        "- data_access: DuckDB read_only=True; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- ah_feature_dates: {ah_feature_dates}",
        f"- matured_rows_with_any_b4_feature: {matured_covered_rows}",
        f"- matured_dates_with_any_b4_feature: {matured_covered_dates}",
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
    lines.extend(
        [
            "",
            "## Fold Coverage And IC",
            "",
            "| fold_start | feature | train_rows | covered_rows_any_b4 | covered_dates_any_b4 | coverage_rate_any_b4 | observations | min_observations | ic | cleared |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in fold_rows.itertuples(index=False):
        lines.append(
            f"| {row.fold_start} | {row.feature} | {int(row.train_rows)} | {int(row.covered_rows_any_b4)} | "
            f"{int(row.covered_dates_any_b4)} | {_fmt(row.coverage_rate_any_b4)} | {int(row.observations)} | "
            f"{int(row.min_observations)} | {_fmt(row.ic)} | {bool(row.cleared)} |"
        )
    lines.append("")
    lines.append("## Honest-Grid With/Without Bakeoff On AH-Covered Dates")
    lines.append("")
    if bakeoff is None:
        lines.append("Skipped: no B4 feature reached mean_abs_ic >= min_feature_ic with the configured observation floor.")
    elif bakeoff.empty:
        lines.append("No bakeoff rows produced.")
    else:
        lines.extend(
            [
                "| model | without_b4_rows | with_b4_rows | without_b4_full_oos_spearman | with_b4_full_oos_spearman | delta_with_minus_without |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bakeoff.itertuples(index=False):
            lines.append(
                f"| {row.model} | {int(row.without_b4_honest_covered_rows)} | {int(row.with_b4_honest_covered_rows)} | "
                f"{_fmt(row.without_b4_full_oos_spearman)} | {_fmt(row.with_b4_full_oos_spearman)} | {_fmt(row.delta_with_minus_without)} |"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    available = _read_table_columns(duckdb_path=settings.paths.duckdb_path, table="universe_daily_snapshots")
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path, columns=_snapshot_columns(available))
    ah_history = _read_ah_history(ah_duckdb_path=settings.paths.data_dir / "ah_history.duckdb")
    frame, ah_features = _merge_b4_features(service._prepare_snapshot_frame(snapshots), ah_history)
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
    covered_mask = eligible[B4_BASE_FEATURES].notna().any(axis=1) if not eligible.empty else pd.Series(dtype=bool)
    covered_dates = set(pd.to_datetime(eligible.loc[covered_mask, "snapshot_date"], errors="coerce").dropna().dt.normalize().tolist())
    bakeoff = None
    if not summary.empty and bool(summary["mean_abs_ic"].ge(float(min_feature_ic)).any()):
        calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
        bakeoff = _bakeoff_rows(
            service,
            eligible,
            config=config,
            calendar_dates=calendar_dates,
            covered_dates=covered_dates,
        )
    _render_report(
        output_path=output_path,
        summary=summary,
        fold_rows=fold_rows,
        bakeoff=bakeoff,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        ah_feature_dates=int(ah_features["snapshot_date"].nunique()) if not ah_features.empty else 0,
        matured_covered_rows=int(covered_mask.sum()) if not eligible.empty else 0,
        matured_covered_dates=len(covered_dates),
        min_feature_ic=min_feature_ic,
        observation_fraction=observation_fraction,
    )
    return {
        "output_path": str(output_path),
        "eligible_dates": int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        "matured_covered_dates": len(covered_dates),
        "survivors": int(summary["mean_abs_ic"].ge(float(min_feature_ic)).sum()) if not summary.empty else 0,
        "best_mean_abs_ic": float(summary["mean_abs_ic"].max()) if not summary.empty else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit B4 cross-sectional AH breadth features.")
    parser.add_argument("--output", type=Path, default=Path("reports/b4_ah_breadth.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} eligible_dates={eligible_dates} matured_covered_dates={matured_covered_dates} "
        "survivors={survivors} best_mean_abs_ic={best_mean_abs_ic:+.4f}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
