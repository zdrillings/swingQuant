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

from src.settings import get_settings
from src.utils.performance_metrics import annualized_sharpe, newey_west_t_stat, years_required_for_tstat


PROMOTION_BASKET_SIZE = 2
DEFAULT_COMPARISON_TARGETS = ("alpha_vs_sector_60d", "path_alpha_vs_sector_60d")


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


def _attach_snapshot_targets(
    frame: pd.DataFrame,
    *,
    target_columns: tuple[str, ...],
    duckdb_path: Path,
) -> pd.DataFrame:
    missing_targets = [column for column in target_columns if column not in frame.columns]
    if not missing_targets:
        return frame.copy()
    import duckdb

    keys = frame[["snapshot_date", "ticker"]].drop_duplicates().copy()
    keys["snapshot_date"] = pd.to_datetime(keys["snapshot_date"]).dt.strftime("%Y-%m-%d")
    select_columns = ", ".join(f"s.{column}" for column in missing_targets)
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        conn.register("oos_keys", keys)
        targets = conn.execute(
            f"""
            SELECT
                k.snapshot_date,
                k.ticker,
                {select_columns}
            FROM oos_keys k
            LEFT JOIN universe_daily_snapshots s
              ON CAST(s.snapshot_date AS DATE) = CAST(k.snapshot_date AS DATE)
             AND s.ticker = k.ticker
            """
        ).fetchdf()
        conn.unregister("oos_keys")
    targets["snapshot_date"] = pd.to_datetime(targets["snapshot_date"]).dt.normalize()
    working = frame.copy()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
    return working.merge(targets, on=["snapshot_date", "ticker"], how="left")


def _comparison_rows(
    frame: pd.DataFrame,
    *,
    target_columns: tuple[str, ...],
    cost_fraction: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for model_name, model_frame in frame.groupby("model_name", sort=True):
        for target_column in target_columns:
            summary = _evaluate(model_frame, target_column=target_column, cost_fraction=cost_fraction)
            rows.append(
                {
                    "model": str(model_name),
                    "target_column": target_column,
                    "dates": int(summary.get("dates", 0)),
                    "spearman": float(summary.get("spearman", float("nan"))),
                    "top2_mean_target": float(summary.get("mean_target", float("nan"))),
                    "top2_hit_rate": float(summary.get("hit_rate", float("nan"))),
                    "top2_mean_target_excess": float(summary.get("mean_target_excess", float("nan"))),
                    "top2_hit_rate_excess": float(summary.get("hit_rate_excess", float("nan"))),
                    "beat_universe_rate": float(summary.get("beat_universe_rate", float("nan"))),
                    "universe_mean_target": float(summary.get("universe_mean_target", float("nan"))),
                    "universe_hit_rate": float(summary.get("universe_hit_rate", float("nan"))),
                }
            )
    return rows


def _format_float(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{places}f}"


def _render_comparison_report(rows: list[dict[str, float | int | str]], *, source_csv: Path) -> str:
    lines = [
        "# Shortlist Path vs Endpoint Label Comparison",
        "",
        f"- source_oos_csv: {source_csv}",
        f"- promotion_basket_size: {PROMOTION_BASKET_SIZE}",
        "",
        "| model | target | dates | spearman | top2_mean | top2_hit | top2_mean_excess | top2_hit_excess | beat_universe | universe_mean | universe_hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {target_column} | {dates} | {spearman} | {top2_mean_target} | {top2_hit_rate} | "
            "{top2_mean_target_excess} | {top2_hit_rate_excess} | {beat_universe_rate} | "
            "{universe_mean_target} | {universe_hit_rate} |".format(
                model=row["model"],
                target_column=row["target_column"],
                dates=row["dates"],
                spearman=_format_float(row["spearman"]),
                top2_mean_target=_format_float(row["top2_mean_target"]),
                top2_hit_rate=_format_float(row["top2_hit_rate"]),
                top2_mean_target_excess=_format_float(row["top2_mean_target_excess"]),
                top2_hit_rate_excess=_format_float(row["top2_hit_rate_excess"]),
                beat_universe_rate=_format_float(row["beat_universe_rate"]),
                universe_mean_target=_format_float(row["universe_mean_target"]),
                universe_hit_rate=_format_float(row["universe_hit_rate"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute shortlist model OOS acceptance summaries from persisted CSV rows.")
    parser.add_argument("--csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--costs", choices=("net", "gross"), default="net")
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--commission-bps-per-side", type=float, default=0.0)
    parser.add_argument("--recent-dates", type=int, default=60)
    parser.add_argument("--fold-size", type=int, default=20)
    parser.add_argument(
        "--compare-path-labels",
        action="store_true",
        help="Compare the same persisted predictions against endpoint and path 60d labels.",
    )
    parser.add_argument("--duckdb", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/path_vs_endpoint_label_comparison.md"))
    args = parser.parse_args()
    if args.costs == "net":
        cost_fraction = ((float(args.slippage_bps_per_side) + float(args.commission_bps_per_side)) * 2.0) / 10_000.0
    else:
        cost_fraction = 0.0

    frame = pd.read_csv(args.csv, low_memory=False)
    if args.compare_path_labels:
        target_columns = DEFAULT_COMPARISON_TARGETS
        duckdb_path = args.duckdb or get_settings().paths.duckdb_path
        frame = _attach_snapshot_targets(frame, target_columns=target_columns, duckdb_path=duckdb_path)
        required = {"snapshot_date", "ticker", "model_name", "predicted_alpha", *target_columns}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise SystemExit(f"Missing required columns for label comparison: {', '.join(missing)}")
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
        rows = _comparison_rows(frame, target_columns=target_columns, cost_fraction=cost_fraction)
        report = _render_comparison_report(rows, source_csv=args.csv)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(report)
        print(f"wrote: {args.output}")
        return 0

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
