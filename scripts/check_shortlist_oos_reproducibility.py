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

from src.utils.performance_metrics import annualized_sharpe, newey_west_t_stat, years_required_for_tstat


PROMOTION_BASKET_SIZE = 2


def _evaluate(
    frame: pd.DataFrame,
    *,
    target_column: str,
    cost_fraction: float,
    top_n: int = PROMOTION_BASKET_SIZE,
) -> dict[str, float | int]:
    rows: list[dict[str, float | int]] = []
    for _snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
        ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
        picks = ordered.head(int(top_n)).copy()
        target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
        universe = pd.to_numeric(day_frame[target_column], errors="coerce").dropna()
        if target.empty or universe.empty:
            continue
        net_target = target - float(cost_fraction)
        net_universe = universe - float(cost_fraction)
        score = pd.to_numeric(ordered["predicted_alpha"], errors="coerce")
        full_target = pd.to_numeric(ordered[target_column], errors="coerce")
        valid = score.notna() & full_target.notna()
        spearman = float("nan")
        if int(valid.sum()) >= 3 and score[valid].nunique(dropna=True) > 1 and full_target[valid].nunique(dropna=True) > 1:
            corr = score[valid].corr(full_target[valid], method="spearman")
            if pd.notna(corr) and math.isfinite(float(corr)):
                spearman = float(corr)
        rows.append(
            {
                "pick_count": len(picks.index),
                "gross_mean_target": float(target.mean()),
                "mean_target": float(net_target.mean()),
                "hit_rate": float((net_target > 0.0).mean()),
                "universe_mean_target": float(net_universe.mean()),
                "universe_hit_rate": float((net_universe > 0.0).mean()),
                "spearman": spearman,
            }
        )
    if not rows:
        return {"dates": 0}
    summary = pd.DataFrame(rows)
    net_targets = pd.to_numeric(summary["mean_target"], errors="coerce").dropna()
    horizon = _target_horizon_days(target_column)
    sharpe = annualized_sharpe(net_targets, periods_per_year=max(252.0 / float(horizon), 1.0))
    return {
        "dates": len(summary.index),
        "avg_pick_count": float(summary["pick_count"].mean()),
        "gross_mean_target": float(summary["gross_mean_target"].mean()),
        "mean_target": float(summary["mean_target"].mean()),
        "hit_rate": float(summary["hit_rate"].mean()),
        "universe_mean_target": float(summary["universe_mean_target"].mean()),
        "universe_hit_rate": float(summary["universe_hit_rate"].mean()),
        "mean_target_excess": float((summary["mean_target"] - summary["universe_mean_target"]).mean()),
        "hit_rate_excess": float((summary["hit_rate"] - summary["universe_hit_rate"]).mean()),
        "beat_universe_rate": float((summary["mean_target"] > summary["universe_mean_target"]).mean()),
        "spearman": float(summary["spearman"].dropna().mean()) if summary["spearman"].notna().any() else float("nan"),
        "net_sharpe": sharpe,
        "newey_west_t": newey_west_t_stat(net_targets, lag=horizon),
        "years_for_t_1_96": years_required_for_tstat(sharpe),
        "round_trip_cost": float(cost_fraction),
    }


def _rolling_window_summaries(
    frame: pd.DataFrame,
    *,
    model_name: str,
    target_column: str,
    cost_fraction: float,
    windows: tuple[int, ...] = (20, 40, 60),
    fold_windows: tuple[int, ...] = (1, 3),
    fold_size: int = 20,
) -> list[tuple[str, dict[str, float | int]]]:
    rows: list[tuple[str, dict[str, float | int]]] = []
    dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
    for window in windows:
        selected_dates = dates[-min(int(window), len(dates)) :]
        scoped = frame[frame["snapshot_date"].isin(selected_dates)].copy()
        rows.append(
            (
                f"{model_name}_{int(window)}d",
                _evaluate(scoped, target_column=target_column, cost_fraction=cost_fraction),
            )
        )
    for fold_count in fold_windows:
        fold_date_count = max(int(fold_count), 1) * max(int(fold_size), 1)
        selected_dates = dates[-min(fold_date_count, len(dates)) :]
        scoped = frame[frame["snapshot_date"].isin(selected_dates)].copy()
        rows.append(
            (
                f"{model_name}_last_{int(fold_count)}fold",
                _evaluate(scoped, target_column=target_column, cost_fraction=cost_fraction),
            )
        )
    return rows


def _resolve_target_column(frame: pd.DataFrame, requested_target_column: str | None) -> str:
    if requested_target_column:
        if requested_target_column not in frame.columns:
            raise SystemExit(
                f"Requested target column '{requested_target_column}' is absent from the OOS artifact."
            )
        return requested_target_column
    if "artifact_evaluation_target_column" not in frame.columns:
        raise SystemExit(
            "OOS artifact does not declare artifact_evaluation_target_column; "
            "pass --target-column explicitly for legacy artifacts."
        )
    declared = frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist()
    if len(declared) != 1:
        raise SystemExit(
            "OOS artifact has inconsistent artifact_evaluation_target_column values: "
            + ", ".join(sorted(declared))
        )
    target_column = declared[0]
    if target_column not in frame.columns:
        raise SystemExit(
            f"OOS artifact declares evaluation target '{target_column}', but that column is absent."
        )
    return target_column


def _target_horizon_days(target_column: str) -> int:
    for piece in str(target_column).split("_"):
        if piece.endswith("d") and piece[:-1].isdigit():
            return max(int(piece[:-1]), 1)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute shortlist model OOS acceptance summaries from persisted CSV rows.")
    parser.add_argument("--csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--costs", choices=("net", "gross"), default="net")
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--commission-bps-per-side", type=float, default=0.0)
    parser.add_argument("--recent-dates", type=int, default=60)
    parser.add_argument("--fold-size", type=int, default=20)
    args = parser.parse_args()
    if args.costs == "net":
        cost_fraction = ((float(args.slippage_bps_per_side) + float(args.commission_bps_per_side)) * 2.0) / 10_000.0
    else:
        cost_fraction = 0.0

    frame = pd.read_csv(args.csv, low_memory=False)
    target_column = _resolve_target_column(frame, args.target_column)
    required = {"snapshot_date", "ticker", "model_name", "predicted_alpha", target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {args.csv}: {', '.join(missing)}")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    print(f"target_column: {target_column}")
    for model_name, model_frame in frame.groupby("model_name", sort=True):
        print(f"{model_name}:")
        print(f"  full: {_evaluate(model_frame, target_column=target_column, cost_fraction=cost_fraction)}")
        recent_dates = sorted(model_frame["snapshot_date"].drop_duplicates().tolist())[-max(int(args.recent_dates), 1):]
        recent = model_frame[model_frame["snapshot_date"].isin(recent_dates)].copy()
        print(f"  recent_{len(recent_dates)}: {_evaluate(recent, target_column=target_column, cost_fraction=cost_fraction)}")
        for label, summary in _rolling_window_summaries(
            model_frame,
            model_name=str(model_name),
            target_column=target_column,
            cost_fraction=cost_fraction,
            fold_size=int(args.fold_size),
        ):
            print(f"  {label}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
