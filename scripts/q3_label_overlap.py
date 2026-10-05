from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class OverlapSummary:
    label: str
    raw_rows: int
    tickers: int
    dates: int
    row_overlap_share: float
    pair_overlap_share: float
    max_overlap_days: int
    median_overlap_days: float | None
    p90_overlap_days: float | None
    median_gap_days: float | None
    p10_gap_days: float | None
    p90_gap_days: float | None
    independent_rows: int

    @property
    def effective_independent_share(self) -> float:
        if self.raw_rows <= 0:
            return 0.0
        return float(self.independent_rows) / float(self.raw_rows)


def _normalize_label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "snapshot_date"])
    working = frame[["ticker", "snapshot_date"]].copy()
    working["ticker"] = working["ticker"].astype(str).str.strip()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["ticker", "snapshot_date"])
    working = working[working["ticker"].ne("")]
    return working.drop_duplicates(subset=["ticker", "snapshot_date"]).sort_values(
        ["ticker", "snapshot_date"]
    ).reset_index(drop=True)


def _calendar_index(dates: pd.Series | list[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    normalized = pd.to_datetime(pd.Series(dates), errors="coerce").dt.normalize().dropna()
    unique_dates = sorted(set(normalized.tolist()))
    return {date_value: index for index, date_value in enumerate(unique_dates)}


def _greedy_independent_count(indices: list[int], *, horizon_days: int) -> int:
    count = 0
    last_kept: int | None = None
    for index in sorted(set(int(value) for value in indices)):
        if last_kept is None or index - last_kept >= int(horizon_days):
            count += 1
            last_kept = index
    return count


def summarize_label_overlap(
    frame: pd.DataFrame,
    *,
    label: str,
    horizon_days: int,
    calendar_dates: pd.Series | list[pd.Timestamp] | None = None,
) -> OverlapSummary:
    rows = _normalize_label_rows(frame)
    if rows.empty:
        return OverlapSummary(
            label=label,
            raw_rows=0,
            tickers=0,
            dates=0,
            row_overlap_share=0.0,
            pair_overlap_share=0.0,
            max_overlap_days=0,
            median_overlap_days=None,
            p90_overlap_days=None,
            median_gap_days=None,
            p10_gap_days=None,
            p90_gap_days=None,
            independent_rows=0,
        )
    index_source = calendar_dates if calendar_dates is not None else rows["snapshot_date"]
    date_index = _calendar_index(index_source)
    rows["_date_index"] = rows["snapshot_date"].map(date_index)
    rows = rows.dropna(subset=["_date_index"]).copy()
    rows["_date_index"] = rows["_date_index"].astype(int)

    row_overlap_flags: list[bool] = []
    row_overlap_depths: list[int] = []
    pair_gaps: list[int] = []
    pair_overlaps: list[int] = []
    independent_rows = 0
    horizon = max(int(horizon_days), 1)

    for _, ticker_frame in rows.groupby("ticker", sort=False):
        indices = sorted(ticker_frame["_date_index"].astype(int).tolist())
        independent_rows += _greedy_independent_count(indices, horizon_days=horizon)
        previous_gap: int | None = None
        next_gaps: list[int | None] = []
        for left, right in zip(indices, indices[1:]):
            gap = int(right) - int(left)
            pair_gaps.append(gap)
            pair_overlaps.append(max(0, horizon - gap))
            next_gaps.append(gap)
        next_gaps.append(None)
        for position, _index in enumerate(indices):
            candidate_overlaps: list[int] = []
            if previous_gap is not None:
                candidate_overlaps.append(max(0, horizon - previous_gap))
            if next_gaps[position] is not None:
                candidate_overlaps.append(max(0, horizon - int(next_gaps[position])))
            max_depth = max(candidate_overlaps) if candidate_overlaps else 0
            row_overlap_depths.append(max_depth)
            row_overlap_flags.append(max_depth > 0)
            previous_gap = next_gaps[position]

    gap_series = pd.Series(pair_gaps, dtype="float64")
    pair_overlap_series = pd.Series(pair_overlaps, dtype="float64")
    depth_series = pd.Series(row_overlap_depths, dtype="float64")
    overlap_rows = int(sum(row_overlap_flags))
    overlap_pairs = int((pair_overlap_series > 0).sum()) if not pair_overlap_series.empty else 0

    return OverlapSummary(
        label=label,
        raw_rows=int(len(rows.index)),
        tickers=int(rows["ticker"].nunique()),
        dates=int(rows["snapshot_date"].nunique()),
        row_overlap_share=float(overlap_rows / len(rows.index)) if len(rows.index) else 0.0,
        pair_overlap_share=float(overlap_pairs / len(pair_overlap_series.index)) if len(pair_overlap_series.index) else 0.0,
        max_overlap_days=int(max(row_overlap_depths) if row_overlap_depths else 0),
        median_overlap_days=float(depth_series.median()) if not depth_series.empty else None,
        p90_overlap_days=float(depth_series.quantile(0.90)) if not depth_series.empty else None,
        median_gap_days=float(gap_series.median()) if not gap_series.empty else None,
        p10_gap_days=float(gap_series.quantile(0.10)) if not gap_series.empty else None,
        p90_gap_days=float(gap_series.quantile(0.90)) if not gap_series.empty else None,
        independent_rows=int(independent_rows),
    )


def walk_forward_fold_training_rows(
    labeled_rows: pd.DataFrame,
    *,
    min_train_dates: int,
    max_train_dates: int | None,
    test_window_dates: int,
    oos_stride_dates: int,
    label_horizon_dates: int,
) -> pd.DataFrame:
    rows = _normalize_label_rows(labeled_rows)
    if rows.empty:
        return pd.DataFrame(columns=["fold_start", "ticker", "snapshot_date"])
    dates = sorted(rows["snapshot_date"].drop_duplicates().tolist())
    date_index = {date_value: index for index, date_value in enumerate(dates)}
    fold_rows: list[pd.DataFrame] = []
    start_index = int(min_train_dates)
    stride = max(int(oos_stride_dates or test_window_dates), 1)
    horizon = max(int(label_horizon_dates or 0), 0)
    while start_index < len(dates):
        test_dates = dates[start_index : start_index + max(int(test_window_dates), 1)]
        if not test_dates:
            break
        train_end_index = max(0, start_index - horizon)
        train_dates = dates[:train_end_index]
        if len(train_dates) >= int(min_train_dates):
            if max_train_dates is not None:
                train_dates = train_dates[-max(int(max_train_dates), 1):]
            train_date_set = set(train_dates)
            train = rows[rows["snapshot_date"].isin(train_date_set)].copy()
            if horizon > 1:
                train["_date_index"] = train["snapshot_date"].map(date_index)
                train = train[
                    train["_date_index"].notna()
                    & ((((int(start_index) - 1) - train["_date_index"].astype(int)) % horizon) == 0)
                ].copy()
                train = train.drop(columns=["_date_index"])
            train["fold_start"] = pd.Timestamp(test_dates[0]).normalize()
            fold_rows.append(train)
        start_index += stride
    if not fold_rows:
        return pd.DataFrame(columns=["fold_start", "ticker", "snapshot_date"])
    return pd.concat(fold_rows, ignore_index=True)


def summarize_fold_training_overlap(
    fold_rows: pd.DataFrame,
    *,
    horizon_days: int,
    calendar_dates: pd.Series | list[pd.Timestamp],
) -> OverlapSummary:
    if fold_rows.empty:
        return summarize_label_overlap(
            fold_rows,
            label="walk_forward_training_rows_per_fold",
            horizon_days=horizon_days,
            calendar_dates=calendar_dates,
        )
    summaries: list[OverlapSummary] = []
    for fold_start, frame in fold_rows.groupby("fold_start", sort=True):
        summaries.append(
            summarize_label_overlap(
                frame,
                label=str(pd.Timestamp(fold_start).date()),
                horizon_days=horizon_days,
                calendar_dates=calendar_dates,
            )
        )
    raw_rows = sum(summary.raw_rows for summary in summaries)
    independent_rows = sum(summary.independent_rows for summary in summaries)
    overlap_rows = sum(round(summary.row_overlap_share * summary.raw_rows) for summary in summaries)
    all_pairs = []
    all_depths: list[int] = []
    max_overlap = max((summary.max_overlap_days for summary in summaries), default=0)
    for _, frame in fold_rows.groupby("fold_start", sort=True):
        normalized = _normalize_label_rows(frame)
        date_index = _calendar_index(calendar_dates)
        normalized["_date_index"] = normalized["snapshot_date"].map(date_index)
        for _, ticker_frame in normalized.dropna(subset=["_date_index"]).groupby("ticker", sort=False):
            indices = sorted(ticker_frame["_date_index"].astype(int).tolist())
            gaps = [int(right) - int(left) for left, right in zip(indices, indices[1:])]
            all_pairs.extend(gaps)
            previous_gap: int | None = None
            next_gaps = [*gaps, None]
            for position, _index in enumerate(indices):
                candidates: list[int] = []
                if previous_gap is not None:
                    candidates.append(max(0, int(horizon_days) - previous_gap))
                if next_gaps[position] is not None:
                    candidates.append(max(0, int(horizon_days) - int(next_gaps[position])))
                all_depths.append(max(candidates) if candidates else 0)
                previous_gap = next_gaps[position]
    gap_series = pd.Series(all_pairs, dtype="float64")
    depth_series = pd.Series(all_depths, dtype="float64")
    overlap_pair_count = int((gap_series < int(horizon_days)).sum()) if not gap_series.empty else 0
    return OverlapSummary(
        label="walk_forward_training_rows_per_fold",
        raw_rows=int(raw_rows),
        tickers=int(fold_rows["ticker"].nunique()),
        dates=int(fold_rows["snapshot_date"].nunique()),
        row_overlap_share=float(overlap_rows / raw_rows) if raw_rows else 0.0,
        pair_overlap_share=float(overlap_pair_count / len(gap_series.index)) if not gap_series.empty else 0.0,
        max_overlap_days=int(max_overlap),
        median_overlap_days=float(depth_series.median()) if not depth_series.empty else None,
        p90_overlap_days=float(depth_series.quantile(0.90)) if not depth_series.empty else None,
        median_gap_days=float(gap_series.median()) if not gap_series.empty else None,
        p10_gap_days=float(gap_series.quantile(0.10)) if not gap_series.empty else None,
        p90_gap_days=float(gap_series.quantile(0.90)) if not gap_series.empty else None,
        independent_rows=int(independent_rows),
    )


def _read_labeled_rows(*, duckdb_path: Path, target_column: str) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT ticker, snapshot_date
            FROM universe_daily_snapshots
            WHERE {target_column} IS NOT NULL
            ORDER BY ticker, snapshot_date
            """
        ).fetchdf()


def _read_oos_artifact(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "snapshot_date"])
    frame = pd.read_csv(path, usecols=lambda column: column in {"ticker", "snapshot_date"}, low_memory=False)
    return frame[["ticker", "snapshot_date"]].copy()


def _fmt_float(value: float | None, *, places: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{places}f}"


def _render_summary_row(summary: OverlapSummary) -> str:
    return (
        f"| {summary.label} | {summary.raw_rows} | {summary.tickers} | {summary.dates} | "
        f"{_fmt_float(summary.row_overlap_share)} | {_fmt_float(summary.pair_overlap_share)} | "
        f"{summary.max_overlap_days} | {_fmt_float(summary.median_overlap_days, places=1)} | "
        f"{_fmt_float(summary.p90_overlap_days, places=1)} | {_fmt_float(summary.median_gap_days, places=1)} | "
        f"{summary.independent_rows} | {_fmt_float(summary.effective_independent_share)} |"
    )


def render_report(
    *,
    output_path: Path,
    summaries: list[OverlapSummary],
    target_column: str,
    horizon_days: int,
    min_train_dates: int,
    max_train_dates: int | None,
    test_window_dates: int,
    oos_stride_dates: int,
    raw_label_date_min: str,
    raw_label_date_max: str,
) -> str:
    raw = next((summary for summary in summaries if summary.label == "raw_labeled_snapshot_rows"), None)
    train = next((summary for summary in summaries if summary.label == "walk_forward_training_rows_per_fold"), None)
    oos = next((summary for summary in summaries if summary.label == "persisted_oos_prediction_grid"), None)
    train_verdict = "clean" if train is not None and train.max_overlap_days == 0 else "labels overlap"
    oos_verdict = "clean" if oos is not None and oos.max_overlap_days == 0 else "labels overlap"
    raw_verdict = "clean" if raw is not None and raw.max_overlap_days == 0 else "labels overlap"
    lines = [
        "# Q3 Label Overlap Audit",
        "",
        f"- target_column: {target_column}",
        f"- horizon_days: {int(horizon_days)}",
        f"- min_train_dates: {int(min_train_dates)}",
        f"- max_train_dates: {int(max_train_dates) if max_train_dates is not None else 'none'}",
        f"- test_window_dates: {int(test_window_dates)}",
        f"- oos_evaluation_stride_dates: {int(oos_stride_dates)}",
        f"- raw_label_date_range: {raw_label_date_min} -> {raw_label_date_max}",
        "- label_construction: `src/research/universe_snapshot_service.py` computes `alpha_vs_sector_60d` as ticker 60-session forward return minus sector ETF 60-session forward return.",
        "- walk_forward_sampler: `src/research/shortlist_model_service.py` applies a 60-session train/test label embargo, then `_stride_training_labels(..., label_horizon_dates=60)` within each fold.",
        "- label_window_definition: sessions `t+1` through `t+horizon`; consecutive labels overlap when trading-day gap < horizon",
        "",
        f"VERDICT raw labels: {raw_verdict}; walk-forward training rows: {train_verdict}; OOS grid: {oos_verdict}.",
        "",
        "## Summary",
        "",
        "| sample | rows | tickers | dates | row_overlap_share | adjacent_pair_overlap_share | max_overlap_days | median_overlap_days | p90_overlap_days | median_gap_days | independent_rows | independent_share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_render_summary_row(summary) for summary in summaries)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `raw_labeled_snapshot_rows` audits the stored label table directly; daily stored labels are expected to overlap unless a downstream sampler removes them.",
            "- `walk_forward_training_rows_per_fold` audits the model's training sampler inside each fold after the 60-session embargo and horizon stride.",
            "- `persisted_oos_prediction_grid` audits the latest persisted OOS prediction artifact; dense 20-session test blocks overlap for a 60-session forward label.",
            "- `independent_rows` is a greedy non-overlapping count per ticker using the same 60-session forward window.",
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def run_audit(
    *,
    duckdb_path: Path,
    oos_predictions_path: Path,
    output_path: Path,
    target_column: str = "alpha_vs_sector_60d",
    horizon_days: int = 60,
    min_train_dates: int = 252,
    max_train_dates: int | None = 252,
    test_window_dates: int = 20,
    oos_stride_dates: int = 20,
) -> str:
    labeled = _read_labeled_rows(duckdb_path=duckdb_path, target_column=target_column)
    normalized_labeled = _normalize_label_rows(labeled)
    calendar_dates = normalized_labeled["snapshot_date"].drop_duplicates()
    fold_rows = walk_forward_fold_training_rows(
        normalized_labeled,
        min_train_dates=min_train_dates,
        max_train_dates=max_train_dates,
        test_window_dates=test_window_dates,
        oos_stride_dates=oos_stride_dates,
        label_horizon_dates=horizon_days,
    )
    oos_rows = _read_oos_artifact(oos_predictions_path)
    summaries = [
        summarize_label_overlap(
            normalized_labeled,
            label="raw_labeled_snapshot_rows",
            horizon_days=horizon_days,
            calendar_dates=calendar_dates,
        ),
        summarize_fold_training_overlap(
            fold_rows,
            horizon_days=horizon_days,
            calendar_dates=calendar_dates,
        ),
        summarize_label_overlap(
            oos_rows,
            label="persisted_oos_prediction_grid",
            horizon_days=horizon_days,
            calendar_dates=calendar_dates,
        ),
    ]
    date_min = (
        pd.Timestamp(normalized_labeled["snapshot_date"].min()).date().isoformat()
        if not normalized_labeled.empty
        else "n/a"
    )
    date_max = (
        pd.Timestamp(normalized_labeled["snapshot_date"].max()).date().isoformat()
        if not normalized_labeled.empty
        else "n/a"
    )
    return render_report(
        output_path=output_path,
        summaries=summaries,
        target_column=target_column,
        horizon_days=horizon_days,
        min_train_dates=min_train_dates,
        max_train_dates=max_train_dates,
        test_window_dates=test_window_dates,
        oos_stride_dates=oos_stride_dates,
        raw_label_date_min=date_min,
        raw_label_date_max=date_max,
    )


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Audit 60d label overlap in universe snapshots and OOS grids.")
    parser.add_argument("--duckdb-path", type=Path, default=settings.paths.duckdb_path)
    parser.add_argument(
        "--oos-predictions",
        type=Path,
        default=settings.paths.reports_dir / "shortlist_model_oos_predictions.csv",
    )
    parser.add_argument("--output", type=Path, default=settings.paths.reports_dir / "q3_label_overlap.md")
    parser.add_argument("--target-column", default="alpha_vs_sector_60d")
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--min-train-dates", type=int, default=252)
    parser.add_argument("--max-train-dates", type=int, default=252)
    parser.add_argument("--test-window-dates", type=int, default=20)
    parser.add_argument("--oos-stride-dates", type=int, default=20)
    args = parser.parse_args()
    run_audit(
        duckdb_path=args.duckdb_path,
        oos_predictions_path=args.oos_predictions,
        output_path=args.output,
        target_column=args.target_column,
        horizon_days=args.horizon_days,
        min_train_dates=args.min_train_dates,
        max_train_dates=args.max_train_dates,
        test_window_dates=args.test_window_dates,
        oos_stride_dates=args.oos_stride_dates,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
