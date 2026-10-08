from __future__ import annotations

from datetime import UTC, datetime
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

from src.research.orthogonal_ensemble import (
    acceptance_window_summary,
    build_rank_ensemble_predictions,
    build_two_member_rank_blend,
    passes_acceptance,
)
from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.settings import get_settings


REPORT_PATH = Path("reports/p4_blend_follow.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLDS = 3
TOP_N = 2
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
SIGNAL_MODEL = "signal_proxy"
TWO_WAY_GRID = tuple(round(value / 10.0, 1) for value in range(2, 9))
SIGNAL_WEIGHTS = (0.1, 0.2)

FLOORS = (
    ("full_oos", "spearman", "full_oos_spearman", ">=", 0.0),
    ("last_fold", "hit_excess", "last_fold_hit_rate_excess", ">=", 0.02),
    ("last_fold", "beat", "last_fold_beat_universe_rate", ">=", 0.50),
    ("last_fold", "mean_excess", "last_fold_mean_target_excess", ">=", 0.0),
    ("last_fold", "spearman", "last_fold_spearman", ">=", 0.0),
    ("last_fold", "top_ticker_date_rate", "last_fold_top_ticker_date_rate", "<=", 0.40),
    ("trailing_3fold", "hit_excess", "trailing_3fold_hit_rate_excess", ">=", 0.02),
    ("trailing_3fold", "beat", "trailing_3fold_beat_universe_rate", ">=", 0.50),
    ("trailing_3fold", "mean_excess", "trailing_3fold_mean_target_excess", ">=", 0.0),
    ("trailing_3fold", "spearman", "trailing_3fold_spearman", ">=", 0.0),
    ("trailing_3fold", "top_ticker_date_rate", "trailing_3fold_top_ticker_date_rate", "<=", 0.40),
)


def main() -> None:
    raw = load_oos_predictions(OOS_PATH)
    calendar_dates = load_snapshot_calendar_dates()
    evaluation_keys = load_fixed_evaluation_keys(OOS_PATH, calendar_dates=calendar_dates)
    rows = evaluate_variants(raw, evaluation_keys=evaluation_keys)
    REPORT_PATH.write_text(
        render_report(raw=raw, calendar_dates=calendar_dates, evaluation_keys=evaluation_keys, rows=rows),
        encoding="utf-8",
    )


def load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = {
        "snapshot_date",
        "ticker",
        "sector",
        "model_name",
        "predicted_alpha",
        TARGET_COLUMN,
        "artifact_evaluation_target_column",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = {"snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise SystemExit(f"OOS artifact is missing required columns: {', '.join(sorted(missing))}")
    if "artifact_evaluation_target_column" in frame.columns:
        declared = frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist()
        if declared and set(declared) != {TARGET_COLUMN}:
            raise SystemExit("Refusing to audit non-production target artifact: " + ", ".join(sorted(set(declared))))
    frame = frame[frame["model_name"].astype(str).isin({RIDGE_MODEL, OVERNIGHT_MODEL, SIGNAL_MODEL})].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN])


def load_snapshot_calendar_dates() -> list[pd.Timestamp]:
    try:
        import duckdb
    except ModuleNotFoundError:
        return []
    duckdb_path = get_settings().paths.duckdb_path
    if not duckdb_path.exists():
        return []
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
        ).fetchall()
    parsed = pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dt.normalize().dropna()
    return sorted(parsed.drop_duplicates().tolist())


def load_fixed_evaluation_keys(path: Path, *, calendar_dates: list[pd.Timestamp] | None = None) -> set[tuple[str, pd.Timestamp]]:
    key_frame = pd.read_csv(path, usecols=["snapshot_date", "ticker"])
    key_frame["snapshot_date"] = pd.to_datetime(key_frame["snapshot_date"], errors="coerce").dt.normalize()
    key_frame["ticker"] = key_frame["ticker"].astype(str).str.strip()
    key_frame = key_frame.dropna(subset=["snapshot_date", "ticker"]).drop_duplicates(["ticker", "snapshot_date"])
    fixed = deoverlap_oos_predictions(
        key_frame,
        horizon_days=HORIZON_DAYS,
        calendar_dates=calendar_dates or key_frame["snapshot_date"].dropna().drop_duplicates().tolist(),
    )
    return {
        (str(row.ticker), pd.Timestamp(row.snapshot_date).normalize())
        for row in fixed[["ticker", "snapshot_date"]].itertuples(index=False)
    }


def evaluate_variants(raw: pd.DataFrame, *, evaluation_keys: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    variants = _baseline_and_two_way_variants(raw)
    rows = _summarize_variants(variants, evaluation_keys=evaluation_keys)
    if _should_try_signal_three_way(rows):
        variants.update(_three_way_variants(raw))
        rows = _summarize_variants(variants, evaluation_keys=evaluation_keys)
    return rows


def _summarize_variants(
    variants: dict[str, pd.DataFrame],
    *,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, predictions in variants.items():
        honest = _filter_to_evaluation_keys(predictions, evaluation_keys=evaluation_keys)
        summary = acceptance_window_summary(
            honest,
            target_column=TARGET_COLUMN,
            top_n=TOP_N,
            fold_size=FOLD_SIZE,
            trailing_folds=TRAILING_FOLDS,
        )
        rows.append(
            {
                "variant": variant,
                "rows": len(honest.index),
                "dates": int(honest["snapshot_date"].nunique()) if not honest.empty else 0,
                "passes_acceptance": passes_acceptance(summary),
                "failed_floors": ", ".join(failed_floors(summary)) or "none",
                "max_floor_gap": max_floor_gap(summary),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _baseline_and_two_way_variants(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {
        RIDGE_MODEL: raw[raw["model_name"].eq(RIDGE_MODEL)].copy(),
        OVERNIGHT_MODEL: raw[raw["model_name"].eq(OVERNIGHT_MODEL)].copy(),
        SIGNAL_MODEL: raw[raw["model_name"].eq(SIGNAL_MODEL)].copy(),
    }
    for ridge_weight in TWO_WAY_GRID:
        overnight_weight = round(1.0 - ridge_weight, 1)
        blend = build_two_member_rank_blend(
            raw,
            left_model=RIDGE_MODEL,
            right_model=OVERNIGHT_MODEL,
            left_weight=ridge_weight,
            right_weight=overnight_weight,
            target_column=TARGET_COLUMN,
        )
        variants[f"ridge_overnight_rank_blend_{ridge_weight:.1f}_{overnight_weight:.1f}"] = blend
    return variants


def _three_way_variants(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for ridge_share in TWO_WAY_GRID:
        overnight_share = round(1.0 - ridge_share, 1)
        for signal_weight in SIGNAL_WEIGHTS:
            residual = 1.0 - signal_weight
            weights = {
                RIDGE_MODEL: ridge_share * residual,
                OVERNIGHT_MODEL: overnight_share * residual,
                SIGNAL_MODEL: signal_weight,
            }
            blend = build_rank_ensemble_predictions(
                raw,
                members=(RIDGE_MODEL, OVERNIGHT_MODEL, SIGNAL_MODEL),
                target_column=TARGET_COLUMN,
                weights=weights,
            )
            variants[
                "ridge_overnight_signal_rank_blend_"
                f"{weights[RIDGE_MODEL]:.2f}_{weights[OVERNIGHT_MODEL]:.2f}_{weights[SIGNAL_MODEL]:.2f}"
            ] = blend
    return variants


def _should_try_signal_three_way(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    blends = rows[rows["variant"].astype(str).str.startswith("ridge_overnight_rank_blend_")]
    if blends.empty:
        return False
    return bool(
        pd.to_numeric(blends["trailing_3fold_hit_rate_excess"], errors="coerce").ge(0.02).any()
        and pd.to_numeric(blends["trailing_3fold_beat_universe_rate"], errors="coerce").ge(0.50).any()
    )


def _filter_to_evaluation_keys(predictions: pd.DataFrame, *, evaluation_keys: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    if predictions.empty or not evaluation_keys:
        return predictions.iloc[0:0].copy()
    working = predictions.copy()
    normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    row_keys = list(zip(working["ticker"].astype(str), normalized_dates))
    return working.loc[[key in evaluation_keys for key in row_keys]].copy().reset_index(drop=True)


def failed_floors(summary: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for window, metric, column, operator, threshold in FLOORS:
        value = _as_float(summary.get(column))
        passed = (
            math.isfinite(value)
            and ((operator == ">=" and value >= threshold) or (operator == "<=" and value <= threshold))
        )
        if not passed:
            failures.append(f"{window} {metric} {_fmt(value)} {operator} {_fmt(threshold)}")
    return failures


def max_floor_gap(summary: dict[str, object]) -> float:
    gaps: list[float] = []
    for _, _, column, operator, threshold in FLOORS:
        value = _as_float(summary.get(column))
        if not math.isfinite(value):
            gaps.append(float("inf"))
        elif operator == ">=":
            gaps.append(max(0.0, threshold - value))
        else:
            gaps.append(max(0.0, value - threshold))
    return max(gaps) if gaps else float("inf")


def render_report(
    *,
    raw: pd.DataFrame,
    calendar_dates: list[pd.Timestamp] | None,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
    rows: pd.DataFrame,
) -> str:
    passers = rows[rows["passes_acceptance"].astype(bool)].copy() if not rows.empty else pd.DataFrame()
    best = _best_row(passers if not passers.empty else rows)
    ridge = _row_for(rows, RIDGE_MODEL)
    p4 = _row_for(rows, "ridge_overnight_rank_blend_0.4_0.6")
    universal = _universal_blockers(rows)
    lines = [
        "# P4 Blend Follow-Up",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: P4-BLEND follow-up over ridge_adaptive and overnight_session_specialist",
        "- data_access: read-only OOS artifact and DuckDB calendar; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- output_report: {REPORT_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- top_n: {TOP_N}",
        f"- fixed_label_calendar_horizon_days: {HORIZON_DAYS}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- raw_dates_loaded: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- fixed_evaluation_keys: {len(evaluation_keys)}",
        "- acceptance: full-OOS Spearman >= 0; last-fold and trailing-3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, Spearman >= 0, top_ticker_date_rate <= 0.40",
        f"- verdict: {_verdict(passers=passers, rows=rows)}",
        f"- best_variant: {best.get('variant', 'n/a') if best else 'n/a'}",
        f"- best_full_oos_spearman: {_fmt(best.get('full_oos_spearman') if best else float('nan'))}",
        f"- ridge_baseline_full_oos_spearman: {_fmt(ridge.get('full_oos_spearman') if ridge else float('nan'))}",
        f"- best_delta_vs_ridge_full_oos_spearman: {_fmt(_delta(best, ridge, 'full_oos_spearman'))}",
        f"- first_thread_variant_failed_floors: {p4.get('failed_floors', 'n/a') if p4 else 'n/a'}",
        f"- universal_blocker: {universal or 'none'}",
        "",
        "## Floor Table",
        "",
        "| variant | pass | rows | dates | full_sp | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top_ticker | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top_ticker | trailing_top_rate | failed_floors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in _ordered_rows(rows).itertuples(index=False):
        lines.append(
            f"| {row.variant} | {'yes' if bool(row.passes_acceptance) else 'no'} | {int(row.rows)} | {int(row.dates)} | "
            f"{_fmt(row.full_oos_spearman)} | {_fmt(row.last_fold_hit_rate_excess)} | "
            f"{_fmt(row.last_fold_beat_universe_rate)} | {_fmt(row.last_fold_mean_target_excess)} | "
            f"{_fmt(row.last_fold_spearman)} | {_text(row.last_fold_top_ticker)} | "
            f"{_fmt(row.last_fold_top_ticker_date_rate)} | {_fmt(row.trailing_3fold_hit_rate_excess)} | "
            f"{_fmt(row.trailing_3fold_beat_universe_rate)} | {_fmt(row.trailing_3fold_mean_target_excess)} | "
            f"{_fmt(row.trailing_3fold_spearman)} | {_text(row.trailing_3fold_top_ticker)} | "
            f"{_fmt(row.trailing_3fold_top_ticker_date_rate)} | {row.failed_floors} |"
        )
    lines.extend(
        [
            "",
            "## Implementation Decision",
            "",
            "Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.",
            "",
            "Tomorrow's critic can verify this by checking this report's `verdict`, `first_thread_variant_failed_floors`, `universal_blocker`, and the floor table against the 2026-10-07 production OOS artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _verdict(*, passers: pd.DataFrame, rows: pd.DataFrame) -> str:
    if rows.empty:
        return "NO VERDICT: no variant rows"
    if not passers.empty:
        best = _best_row(passers)
        return f"PASS: {best['variant']} clears all promotion floors"
    closest = _closest_miss(rows)
    return f"NO PASS: no blend clears all floors; closest miss is {closest.get('variant', 'n/a')}"


def _universal_blockers(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    blockers: list[str] = []
    for window, metric, column, operator, threshold in FLOORS:
        values = pd.to_numeric(rows[column], errors="coerce")
        if operator == ">=":
            blocked = values.lt(threshold) | values.isna()
        else:
            blocked = values.gt(threshold) | values.isna()
        if bool(blocked.all()):
            blockers.append(f"{window} {metric}")
    return ", ".join(blockers)


def _ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ordered = rows.copy()
    for column in ("max_floor_gap", "trailing_3fold_beat_universe_rate", "trailing_3fold_hit_rate_excess", "full_oos_spearman"):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.sort_values(
        ["passes_acceptance", "max_floor_gap", "trailing_3fold_beat_universe_rate", "trailing_3fold_hit_rate_excess", "full_oos_spearman"],
        ascending=[False, True, False, False, False],
    )


def _best_row(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {}
    return _ordered_rows(rows).iloc[0].to_dict()


def _closest_miss(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {}
    return _ordered_rows(rows).iloc[0].to_dict()


def _row_for(rows: pd.DataFrame, variant: str) -> dict[str, object]:
    if rows.empty:
        return {}
    scoped = rows[rows["variant"].astype(str).eq(str(variant))]
    return scoped.iloc[0].to_dict() if not scoped.empty else {}


def _delta(left: dict[str, object], right: dict[str, object], column: str) -> float:
    left_value = _as_float(left.get(column))
    right_value = _as_float(right.get(column))
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return left_value - right_value


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: object, *, places: int = 4) -> str:
    number = _as_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _text(value: object) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return text if text and text.lower() != "nan" else "n/a"


if __name__ == "__main__":
    main()
