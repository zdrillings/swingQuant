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

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import filter_eligible_universe
from src.settings import get_settings
from src.utils.db_manager import DatabaseManager


BASELINE_SPEARMAN = 0.4644
BASELINE_PER_DATE_SPEARMAN = 0.4713
CURRENT_BASELINE_DATE_BEST = 0.0739
RUN65_GENERATED_AT = "2026-09-02T00:28:32+00:00"


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _mean_per_date_spearman(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score_column: str = "predicted_alpha",
) -> float:
    values: list[float] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        score = pd.to_numeric(day_frame[score_column], errors="coerce")
        target = pd.to_numeric(day_frame[target_column], errors="coerce")
        valid = score.notna() & target.notna()
        if int(valid.sum()) < 3:
            continue
        if score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
            continue
        corr = score[valid].corr(target[valid], method="spearman")
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    return float(pd.Series(values).mean()) if values else float("nan")


def _pooled_spearman(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score_column: str = "predicted_alpha",
) -> float:
    score = pd.to_numeric(frame[score_column], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid = score.notna() & target.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    if score[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
        return float("nan")
    corr = score[valid].corr(target[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _top_n_summary(
    frame: pd.DataFrame,
    *,
    target_column: str,
    top_n: int,
) -> dict[str, float | int]:
    rows: list[dict[str, float | int]] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        picks = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n))
        target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
        if target.empty:
            continue
        rows.append(
            {
                "mean_target": float(target.mean()),
                "hit_rate": float((target > 0.0).mean()),
                "pick_count": int(len(picks.index)),
            }
        )
    if not rows:
        return {"dates": 0, "mean_target": float("nan"), "hit_rate": float("nan"), "avg_pick_count": float("nan")}
    summary = pd.DataFrame(rows)
    return {
        "dates": int(len(summary.index)),
        "mean_target": float(summary["mean_target"].mean()),
        "hit_rate": float(summary["hit_rate"].mean()),
        "avg_pick_count": float(summary["pick_count"].mean()),
    }


def _decile_calibration(
    frame: pd.DataFrame,
    *,
    target_column: str,
    deciles: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        working = day_frame.copy()
        score = pd.to_numeric(working["predicted_alpha"], errors="coerce")
        target = pd.to_numeric(working[target_column], errors="coerce")
        valid = score.notna() & target.notna()
        working = working.loc[valid].copy()
        if len(working.index) < int(deciles):
            continue
        ranks = pd.to_numeric(working["predicted_alpha"], errors="coerce").rank(method="first", ascending=True)
        working["decile"] = pd.qcut(ranks, q=int(deciles), labels=False, duplicates="drop")
        for decile, decile_frame in working.groupby("decile", sort=True):
            decile_target = pd.to_numeric(decile_frame[target_column], errors="coerce").dropna()
            if decile_target.empty:
                continue
            rows.append(
                {
                    "decile": int(decile),
                    "rows": int(len(decile_target.index)),
                    "mean_target": float(decile_target.mean()),
                    "hit_rate": float((decile_target > 0.0).mean()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["decile", "rows", "mean_target", "hit_rate"])
    return (
        pd.DataFrame(rows)
        .groupby("decile", as_index=False)
        .agg(rows=("rows", "sum"), mean_target=("mean_target", "mean"), hit_rate=("hit_rate", "mean"))
        .sort_values("decile")
        .reset_index(drop=True)
    )


def _read_universe_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
    import duckdb

    safe_columns = ", ".join(columns)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT {safe_columns}
            FROM universe_daily_snapshots
            WHERE alpha_vs_sector_20d IS NOT NULL
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _read_run65_dates(*, sqlite_path: Path) -> set[pd.Timestamp]:
    import sqlite3

    uri = f"file:{sqlite_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT snapshot_date
            FROM Shortlist_Model_Predictions
            WHERE generated_at = ?
              AND dataset_split = 'oos'
            ORDER BY snapshot_date
            """,
            (RUN65_GENERATED_AT,),
        ).fetchall()
    return set(pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna().dt.normalize())


def _experiment_row(
    frame: pd.DataFrame,
    *,
    model: str,
    path: str,
    grid: str,
    target_column: str,
    top_n: int,
) -> dict[str, float | int | str]:
    top_summary = _top_n_summary(frame, target_column=target_column, top_n=top_n)
    return {
        "model": model,
        "path": path,
        "grid": grid,
        "target": target_column,
        "dates": int(frame["snapshot_date"].nunique()),
        "rows": int(len(frame.index)),
        "pooled_spearman": _pooled_spearman(frame, target_column=target_column),
        "per_date_spearman": _mean_per_date_spearman(frame, target_column=target_column),
        "top_n": int(top_n),
        "top_mean": top_summary["mean_target"],
    }


def _sector_xgboost_predictions(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    base_features: list[str],
    ic_screen: str,
) -> pd.DataFrame:
    if ic_screen not in {"on", "off"}:
        raise ValueError(f"Unsupported ic_screen={ic_screen}")
    predictions = service._walk_forward_predictions(
        eligible,
        target_column="alpha_vs_sector_20d",
        evaluation_target_column="alpha_vs_sector_20d",
        model_name="xgboost_model",
        min_train_dates=252,
        test_window_dates=20,
        evaluation_stride_dates=20,
        label_horizon_dates=20,
        model_scope="sector_specific",
        xgboost_params={**service._xgboost_params_for_config("balanced_depth4"), "n_jobs": 1},
        feature_columns_override=base_features,
        min_feature_ic=service._load_min_feature_ic() if ic_screen == "on" else None,
        min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        regime_matching_mode="off",
        regime_transition_purge_mode="off",
    )
    if predictions is None or predictions.empty:
        raise SystemExit(f"sector-specific xgboost produced no OOS predictions for ic_screen={ic_screen}")
    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.normalize()
    return predictions


def _current_global_rows(
    *,
    current: pd.DataFrame,
    grid_dates: set[pd.Timestamp],
    grid: str,
    path: str,
    target_column: str,
    top_n: int,
) -> list[dict[str, float | int | str]]:
    current = current[current["snapshot_date"].isin(grid_dates)].copy()
    rows: list[dict[str, float | int | str]] = []
    for model_name, model_frame in current.groupby("model_name", sort=True):
        rows.append(
            _experiment_row(
                model_frame,
                model=str(model_name),
                path=path,
                grid=grid,
                target_column=target_column,
                top_n=top_n,
            )
        )
    return rows


def _render_report(
    *,
    rows: list[dict[str, float | int | str]],
    deciles: pd.DataFrame,
    experiment_dates: dict[str, int],
    shared_dates: int,
    run65_dates: int,
    eligible_rows: int,
    eligible_dates: int,
    output_path: Path,
) -> str:
    sector_row = next(
        row
        for row in rows
        if row["model"] == "xgboost_model" and row["path"] == "sector-specific-20d-ic-off" and row["grid"] == "shared-current"
    )
    screen_on_row = next(
        row
        for row in rows
        if row["model"] == "xgboost_model" and row["path"] == "sector-specific-20d-ic-on" and row["grid"] == "shared-current"
    )
    run65_row = next(
        row
        for row in rows
        if row["model"] == "xgboost_model" and row["path"] == "sector-specific-20d-ic-off" and row["grid"] == "run65-dates"
    )
    ic_screen_delta = float(sector_row["per_date_spearman"]) - float(screen_on_row["per_date_spearman"])
    recoverable_gap = BASELINE_SPEARMAN - CURRENT_BASELINE_DATE_BEST
    recovered_share = ic_screen_delta / recoverable_gap if recoverable_gap else float("nan")
    verdict = "edge recovered" if float(sector_row["per_date_spearman"]) >= 0.15 else (
        "partially recovered" if float(sector_row["per_date_spearman"]) >= 0.05 else "flat"
    )

    lines = [
        "# A1 Run 65 Replication",
        "",
        "- generated_at: 2026-10-05",
        "- data_access: read-only DuckDB plus existing OOS CSV; no `./sq` writes; no `data/` mutation",
        "- baseline_reference: run 65 / 00144e4, 20d top-10 sector-specific xgboost, pooled Spearman +0.4644 and per-date Spearman +0.4713 on 20 OOS dates",
        "- experiment: 20d endpoint label, top-10 readout, sector-specific xgboost balanced_depth4, FULL feature pool, fold-local IC screen OFF, regime matching off",
        "- control: same sector-specific 20d setup with the current fold-local IC screen ON",
        "- feature_note: the exact 00144e4 fold-local survivor list was not retained in git artifacts; this isolates sector-specific 20d modeling without pretending to recover unavailable survivor state",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- sector_specific_ic_on_oos_dates: {experiment_dates.get('ic_on', 0)}",
        f"- sector_specific_ic_off_oos_dates: {experiment_dates.get('ic_off', 0)}",
        f"- shared_current_grid_dates: {shared_dates}",
        f"- run65_grid_dates: {run65_dates}",
        "",
        "## Spearman Comparison",
        "",
        "| grid | model | path | target | dates | rows | pooled_spearman | per_date_spearman | top_n | top_mean |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        f"| run65-dates | xgboost_model | forensic-run65-stored | alpha_vs_sector_20d | 20 | 7412 | {_fmt(BASELINE_SPEARMAN)} | {_fmt(BASELINE_PER_DATE_SPEARMAN)} | 10 | +0.1947 |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["grid"]), str(item["path"]), str(item["model"]))):
        lines.append(
            "| {grid} | {model} | {path} | {target} | {dates} | {rows} | {pooled} | {per_date} | {top_n} | {top_mean} |".format(
                grid=row["grid"],
                model=row["model"],
                path=row["path"],
                target=row["target"],
                dates=int(row["dates"]),
                rows=int(row["rows"]),
                pooled=_fmt(row["pooled_spearman"]),
                per_date=_fmt(row["per_date_spearman"]),
                top_n=int(row["top_n"]),
                top_mean=_fmt(row["top_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## IC-Off Decile Calibration",
            "",
            "| decile | rows | mean_target | hit_rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in deciles.itertuples(index=False):
        lines.append(f"| {int(row.decile)} | {int(row.rows)} | {_fmt(row.mean_target)} | {_fmt(row.hit_rate)} |")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"{verdict}: IC-screen-off sector-specific 20d xgboost per-date Spearman is {_fmt(sector_row['per_date_spearman'])} on the shared grid versus IC-screen-on {_fmt(screen_on_row['per_date_spearman'])}, a delta of {_fmt(ic_screen_delta)} Spearman, or {_fmt(recovered_share * 100.0, places=1)}% of the ~0.39 forensic gap. On run 65's 20-date grid, the IC-screen-off pooled Spearman is {_fmt(run65_row['pooled_spearman'])} versus the stored forensic +0.4644.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def run(output_path: Path, oos_csv: Path) -> dict[str, float | int | str]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    base_features = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    columns = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        "alpha_vs_sector_20d",
        *[
            column
            for column in [
                "regime_green",
                "atr_14",
                "relative_strength_index_vs_spy",
                "relative_strength_index_vs_qqq",
                "relative_strength_index_vs_xlk",
                "relative_strength_index_vs_subindustry",
                "rs_vs_spy_5d_change",
                "rs_vs_qqq_5d_change",
                "rs_vs_xlk_5d_change",
                "rs_vs_subindustry_5d_change",
                "rs_vs_subindustry_10d_change",
                "roc_63",
                "roc_126",
                "vol_alpha",
                "sma_200_dist",
                "sma_50_dist",
                "rsi_14",
                "atr_pct_14",
                "atr_pct_14_percentile_252",
                "realized_vol_20_percentile_252",
                "days_to_next_earnings",
                "days_since_last_earnings",
                "last_earnings_gap_pct",
                "last_earnings_volume_ratio_20",
                "last_earnings_open_vs_20d_high",
                "close_vs_last_earnings_close",
                "avg_abs_gap_pct_20",
                "max_gap_down_pct_60",
                "distance_above_20d_high",
                "base_range_pct_20",
                "base_atr_contraction_20",
                "base_volume_dryup_ratio_20",
                "breakout_volume_ratio_50",
                "dollar_volume_ratio_20_60",
                "volume_percentile_60",
                "distance_from_52w_high",
                "days_since_52w_high",
                "rsi_2",
                "ret_1d",
                "ret_5d",
                "close_vs_20d_low",
                "sector_pct_above_50",
                "sector_pct_above_200",
                "sector_median_roc_63",
                "analyst_target_upside",
                "analyst_target_range_pct",
                "analyst_count",
                "analyst_recommendation_score",
                "analyst_eps_revision_breadth",
                "analyst_upgrade_downgrade_score",
                "analyst_snapshot_age_days",
                "analyst_revision_snapshot_age_days",
            ]
        ],
    ]
    frame = _read_universe_snapshots(duckdb_path=settings.paths.duckdb_path, columns=list(dict.fromkeys(columns)))
    frame = service._prepare_snapshot_frame(frame)
    eligible = filter_eligible_universe(frame, eligible_universe_mode="passed_or_trend")
    eligible = eligible.dropna(subset=["alpha_vs_sector_20d"]).sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
    predictions_on = _sector_xgboost_predictions(service, eligible, base_features=base_features, ic_screen="on")
    predictions_off = _sector_xgboost_predictions(service, eligible, base_features=base_features, ic_screen="off")
    current = pd.read_csv(oos_csv, low_memory=False)
    current["snapshot_date"] = pd.to_datetime(current["snapshot_date"], errors="coerce").dt.normalize()
    current_dates = set(current["snapshot_date"].dropna().unique().tolist())
    alpha_20 = frame[["snapshot_date", "ticker", "alpha_vs_sector_20d"]].copy()
    alpha_20["snapshot_date"] = pd.to_datetime(alpha_20["snapshot_date"], errors="coerce").dt.normalize()
    current_with_alpha_20 = current.merge(alpha_20, on=["snapshot_date", "ticker"], how="left")
    sector_dates = set(predictions_off["snapshot_date"].dropna().unique().tolist())
    shared_dates = sector_dates & current_dates
    run65_dates = _read_run65_dates(sqlite_path=settings.paths.sqlite_path)
    rows: list[dict[str, float | int | str]] = [
        _experiment_row(
            predictions_on[predictions_on["snapshot_date"].isin(shared_dates)].copy(),
            model="xgboost_model",
            path="sector-specific-20d-ic-on",
            grid="shared-current",
            target_column="alpha_vs_sector_20d",
            top_n=10,
        ),
        _experiment_row(
            predictions_off[predictions_off["snapshot_date"].isin(shared_dates)].copy(),
            model="xgboost_model",
            path="sector-specific-20d-ic-off",
            grid="shared-current",
            target_column="alpha_vs_sector_20d",
            top_n=10,
        ),
        _experiment_row(
            predictions_off[predictions_off["snapshot_date"].isin(run65_dates)].copy(),
            model="xgboost_model",
            path="sector-specific-20d-ic-off",
            grid="run65-dates",
            target_column="alpha_vs_sector_20d",
            top_n=10,
        ),
    ]
    rows.extend(
        _current_global_rows(
            current=current,
            grid_dates=shared_dates,
            grid="shared-current",
            path="current-global-60d",
            target_column="alpha_vs_sector_60d",
            top_n=2,
        )
    )
    rows.extend(
        _current_global_rows(
            current=current_with_alpha_20,
            grid_dates=run65_dates,
            grid="run65-dates",
            path="current-roster-20d",
            target_column="alpha_vs_sector_20d",
            top_n=10,
        )
    )
    deciles = _decile_calibration(
        predictions_off[predictions_off["snapshot_date"].isin(shared_dates)].copy(),
        target_column="alpha_vs_sector_20d",
    )
    _render_report(
        rows=rows,
        deciles=deciles,
        experiment_dates={
            "ic_on": int(predictions_on["snapshot_date"].nunique()),
            "ic_off": int(predictions_off["snapshot_date"].nunique()),
        },
        shared_dates=len(shared_dates),
        run65_dates=len(run65_dates),
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        output_path=output_path,
    )
    sector_row = next(row for row in rows if row["path"] == "sector-specific-20d-ic-off" and row["grid"] == "shared-current")
    return {
        "output_path": str(output_path),
        "shared_dates": len(shared_dates),
        "run65_dates": len(run65_dates),
        "ic_off_per_date_spearman": float(sector_row["per_date_spearman"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the A1 run-65 IC-screen-off replication.")
    parser.add_argument("--output", type=Path, default=Path("reports/a1_run65_replication.md"))
    parser.add_argument("--oos-csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    args = parser.parse_args()
    result = run(output_path=args.output, oos_csv=args.oos_csv)
    print(
        "wrote {output_path} shared_dates={shared_dates} run65_dates={run65_dates} ic_off_per_date_spearman={ic_off_per_date_spearman:+.4f}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
