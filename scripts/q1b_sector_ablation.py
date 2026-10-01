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
CURRENT_BASELINE_DATE_BEST = 0.0739


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


def _current_global_rows(
    *,
    oos_csv: Path,
    shared_dates: set[pd.Timestamp],
) -> list[dict[str, float | int | str]]:
    current = pd.read_csv(oos_csv, low_memory=False)
    current["snapshot_date"] = pd.to_datetime(current["snapshot_date"], errors="coerce").dt.normalize()
    target_column = "alpha_vs_sector_60d"
    current = current[current["snapshot_date"].isin(shared_dates)].copy()
    rows: list[dict[str, float | int | str]] = []
    for model_name, model_frame in current.groupby("model_name", sort=True):
        rows.append(
            {
                "model": str(model_name),
                "path": "current-global-60d",
                "target": target_column,
                "dates": int(model_frame["snapshot_date"].nunique()),
                "rows": int(len(model_frame.index)),
                "pooled_spearman": _pooled_spearman(model_frame, target_column=target_column),
                "per_date_spearman": _mean_per_date_spearman(model_frame, target_column=target_column),
                "top_n": 2,
                "top_mean": _top_n_summary(model_frame, target_column=target_column, top_n=2)["mean_target"],
            }
        )
    return rows


def _render_report(
    *,
    rows: list[dict[str, float | int | str]],
    deciles: pd.DataFrame,
    sector_dates: int,
    shared_dates: int,
    eligible_rows: int,
    eligible_dates: int,
    output_path: Path,
) -> str:
    sector_row = next(row for row in rows if row["model"] == "xgboost_model" and row["path"] == "sector-specific-20d")
    current_best = max(
        (float(row["per_date_spearman"]) for row in rows if row["path"] == "current-global-60d" and math.isfinite(float(row["per_date_spearman"]))),
        default=float("nan"),
    )
    recovered_from_current = float(sector_row["per_date_spearman"]) - current_best
    recoverable_gap = BASELINE_SPEARMAN - CURRENT_BASELINE_DATE_BEST
    recovered_share = recovered_from_current / recoverable_gap if recoverable_gap else float("nan")
    verdict = "edge recovered" if float(sector_row["per_date_spearman"]) >= 0.15 else (
        "partially recovered" if float(sector_row["per_date_spearman"]) > current_best else "flat"
    )

    lines = [
        "# Q1b Sector-Specific 20d Ablation",
        "",
        "- generated_at: 2026-09-30",
        "- data_access: read-only DuckDB plus existing OOS CSV; no `./sq` writes; no `data/` mutation",
        "- baseline_reference: run 65 / 00144e4, 20d top-10 sector-specific xgboost, pooled Spearman +0.4644 on 20 OOS dates",
        "- experiment: 20d endpoint label, top-10 readout, sector-specific xgboost, horizon-strided training labels, current full feature set with fold-local IC screen, regime matching off",
        "- feature_note: the exact 00144e4 fold-local survivor list was not retained in git artifacts; this isolates sector-specific 20d modeling without pretending to recover unavailable survivor state",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- sector_specific_oos_dates: {sector_dates}",
        f"- shared_current_grid_dates: {shared_dates}",
        "",
        "## Shared-Grid Spearman",
        "",
        "| model | path | target | dates | rows | pooled_spearman | per_date_spearman | top_n | top_mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["path"]), str(item["model"]))):
        lines.append(
            "| {model} | {path} | {target} | {dates} | {rows} | {pooled} | {per_date} | {top_n} | {top_mean} |".format(
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
            "## Sector-Specific Decile Calibration",
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
            f"{verdict}: sector-specific 20d xgboost per-date Spearman is {_fmt(sector_row['per_date_spearman'])} on the shared grid versus the current global roster best {_fmt(current_best)}, recovering {_fmt(recovered_from_current)} Spearman, or {_fmt(recovered_share * 100.0, places=1)}% of the ~0.39 forensic gap.",
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
        min_feature_ic=service._load_min_feature_ic(),
        min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
        regime_matching_mode="off",
        regime_transition_purge_mode="off",
    )
    if predictions is None or predictions.empty:
        raise SystemExit("sector-specific xgboost produced no OOS predictions")
    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.normalize()
    current_dates = set(pd.to_datetime(pd.read_csv(oos_csv, usecols=["snapshot_date"])["snapshot_date"]).dt.normalize())
    sector_dates = set(predictions["snapshot_date"].dropna().unique().tolist())
    shared_dates = sector_dates & current_dates
    shared_predictions = predictions[predictions["snapshot_date"].isin(shared_dates)].copy()
    sector_top = _top_n_summary(shared_predictions, target_column="alpha_vs_sector_20d", top_n=10)
    rows: list[dict[str, float | int | str]] = [
        {
            "model": "xgboost_model",
            "path": "sector-specific-20d",
            "target": "alpha_vs_sector_20d",
            "dates": int(shared_predictions["snapshot_date"].nunique()),
            "rows": int(len(shared_predictions.index)),
            "pooled_spearman": _pooled_spearman(shared_predictions, target_column="alpha_vs_sector_20d"),
            "per_date_spearman": _mean_per_date_spearman(shared_predictions, target_column="alpha_vs_sector_20d"),
            "top_n": 10,
            "top_mean": sector_top["mean_target"],
        }
    ]
    rows.extend(_current_global_rows(oos_csv=oos_csv, shared_dates=shared_dates))
    deciles = _decile_calibration(shared_predictions, target_column="alpha_vs_sector_20d")
    _render_report(
        rows=rows,
        deciles=deciles,
        sector_dates=int(predictions["snapshot_date"].nunique()),
        shared_dates=len(shared_dates),
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        output_path=output_path,
    )
    sector_row = rows[0]
    return {
        "output_path": str(output_path),
        "shared_dates": len(shared_dates),
        "sector_per_date_spearman": float(sector_row["per_date_spearman"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Q1b sector-specific 20d shortlist ablation.")
    parser.add_argument("--output", type=Path, default=Path("reports/q1b_sector_ablation.md"))
    parser.add_argument("--oos-csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    args = parser.parse_args()
    result = run(output_path=args.output, oos_csv=args.oos_csv)
    print(
        "wrote {output_path} shared_dates={shared_dates} sector_per_date_spearman={sector_per_date_spearman:+.4f}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
