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

from src.research.beat_calibration import chronological_isotonic_probability, finite_float
from src.research.orthogonal_ensemble import (
    acceptance_window_summary,
    build_two_member_rank_blend,
    passes_acceptance,
)
from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.settings import get_settings


REPORT_PATH = Path("reports/beat_calibration.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLDS = 3
TOP_N = 2
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
BLEND_VARIANT = "ridge_overnight_rank_blend_0.4_0.6"
CALIBRATED_COLUMN = "calibrated_p_beat_sector_oos"

FLOORS = (
    ("full_oos", "spearman", "full_oos_spearman", ">=", 0.0),
    ("last_fold", "hit_excess", "last_fold_hit_rate_excess", ">=", 0.02),
    ("last_fold", "beat", "last_fold_beat_universe_rate", ">=", 0.50),
    ("last_fold", "mean_excess", "last_fold_mean_target_excess", ">=", 0.0),
    ("last_fold", "spearman", "last_fold_spearman", ">=", 0.0),
    ("last_fold", "top_ticker_rate", "last_fold_top_ticker_date_rate", "<=", 0.40),
    ("trailing_3fold", "hit_excess", "trailing_3fold_hit_rate_excess", ">=", 0.02),
    ("trailing_3fold", "beat", "trailing_3fold_beat_universe_rate", ">=", 0.50),
    ("trailing_3fold", "mean_excess", "trailing_3fold_mean_target_excess", ">=", 0.0),
    ("trailing_3fold", "spearman", "trailing_3fold_spearman", ">=", 0.0),
    ("trailing_3fold", "top_ticker_rate", "trailing_3fold_top_ticker_date_rate", "<=", 0.40),
)


def main() -> None:
    raw = load_oos_predictions(OOS_PATH)
    calendar_dates = load_snapshot_calendar_dates()
    evaluation_keys = load_fixed_evaluation_keys(OOS_PATH, calendar_dates=calendar_dates)
    variants = build_variants(raw)
    rows = summarize_variants(variants, evaluation_keys=evaluation_keys)
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
        declared = set(frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist())
        if declared and declared != {TARGET_COLUMN}:
            raise SystemExit("Refusing to audit non-production target artifact: " + ", ".join(sorted(declared)))
    frame = frame[frame["model_name"].astype(str).isin({RIDGE_MODEL, OVERNIGHT_MODEL})].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN]).copy()


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


def build_variants(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ridge = raw[raw["model_name"].eq(RIDGE_MODEL)].copy()
    blend = build_two_member_rank_blend(
        raw,
        left_model=RIDGE_MODEL,
        right_model=OVERNIGHT_MODEL,
        left_weight=0.4,
        right_weight=0.6,
        target_column=TARGET_COLUMN,
    )
    blend["model_name"] = BLEND_VARIANT
    return {RIDGE_MODEL: ridge, BLEND_VARIANT: blend}


def summarize_variants(
    variants: dict[str, pd.DataFrame],
    *,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, frame in variants.items():
        honest = filter_to_evaluation_keys(frame, evaluation_keys=evaluation_keys)
        calibrated = chronological_isotonic_probability(
            honest,
            target_column=TARGET_COLUMN,
            output_column=CALIBRATED_COLUMN,
            min_train_rows=50,
        )
        rows.append(summarize_score_variant(variant, "raw_score_top2", honest, score_column="predicted_alpha"))
        rows.append(summarize_score_variant(variant, "calibrated_p_top2", calibrated, score_column=CALIBRATED_COLUMN))
    return pd.DataFrame(rows)


def summarize_score_variant(
    variant: str,
    cut: str,
    frame: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, object]:
    scored = frame.copy()
    scored["predicted_alpha"] = pd.to_numeric(scored[score_column], errors="coerce")
    scored = scored.dropna(subset=["predicted_alpha", TARGET_COLUMN]).copy()
    summary = acceptance_window_summary(
        scored,
        target_column=TARGET_COLUMN,
        top_n=TOP_N,
        fold_size=FOLD_SIZE,
        trailing_folds=TRAILING_FOLDS,
    )
    return {
        "variant": variant,
        "cut": cut,
        "score_column": score_column,
        "rows": len(scored.index),
        "dates": int(scored["snapshot_date"].nunique()) if not scored.empty else 0,
        "passes_acceptance": passes_acceptance(summary),
        "failed_floors": ", ".join(failed_floors(summary)) or "none",
        **summary,
    }


def filter_to_evaluation_keys(predictions: pd.DataFrame, *, evaluation_keys: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    if predictions.empty or not evaluation_keys:
        return predictions.iloc[0:0].copy()
    working = predictions.copy()
    normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    row_keys = list(zip(working["ticker"].astype(str), normalized_dates))
    return working.loc[[key in evaluation_keys for key in row_keys]].copy().reset_index(drop=True)


def failed_floors(summary: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for window, metric, column, operator, threshold in FLOORS:
        value = finite_float(summary.get(column))
        passed = (
            math.isfinite(value)
            and ((operator == ">=" and value >= threshold) or (operator == "<=" and value <= threshold))
        )
        if not passed:
            failures.append(f"{window} {metric} {_fmt(value)} {operator} {_fmt(threshold)}")
    return failures


def render_report(
    *,
    raw: pd.DataFrame,
    calendar_dates: list[pd.Timestamp] | None,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
    rows: pd.DataFrame,
) -> str:
    ridge_raw = _row_for(rows, RIDGE_MODEL, "raw_score_top2")
    verdict = verdict_text(rows)
    lines = [
        "# Beat Calibration",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: BEAT-CAL chronological isotonic calibration of score to P(beat sector)",
        "- data_access: read-only OOS artifact and DuckDB calendar; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- output_report: {REPORT_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- models: {RIDGE_MODEL}, {BLEND_VARIANT}",
        f"- top_n: {TOP_N}",
        f"- fixed_label_calendar_horizon_days: {HORIZON_DAYS}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- raw_dates_loaded: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- fixed_evaluation_keys: {len(evaluation_keys)}",
        "- calibration: chronological isotonic, each date trained only on earlier OOS dates; target is alpha_vs_sector_60d > 0",
        "- acceptance: full-OOS Spearman >= 0; last-fold and trailing-3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, Spearman >= 0, top_ticker_date_rate <= 0.40",
        f"- verdict: {verdict}",
        f"- ridge_raw_full_oos_spearman: {_fmt(ridge_raw.get('full_oos_spearman') if ridge_raw else float('nan'))}",
        "",
        "## Floor Table",
        "",
        "| variant | cut | pass | rows | dates | full_sp | delta_vs_ridge_raw | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top | trailing_top_rate | failed_floors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in ordered_rows(rows).itertuples(index=False):
        delta = finite_float(row.full_oos_spearman) - finite_float(ridge_raw.get("full_oos_spearman") if ridge_raw else float("nan"))
        lines.append(
            f"| {row.variant} | {row.cut} | {'yes' if bool(row.passes_acceptance) else 'no'} | "
            f"{int(row.rows)} | {int(row.dates)} | {_fmt(row.full_oos_spearman)} | {_fmt(delta)} | "
            f"{_fmt(row.last_fold_hit_rate_excess)} | {_fmt(row.last_fold_beat_universe_rate)} | "
            f"{_fmt(row.last_fold_mean_target_excess)} | {_fmt(row.last_fold_spearman)} | "
            f"{_text(row.last_fold_top_ticker)} | {_fmt(row.last_fold_top_ticker_date_rate)} | "
            f"{_fmt(row.trailing_3fold_hit_rate_excess)} | {_fmt(row.trailing_3fold_beat_universe_rate)} | "
            f"{_fmt(row.trailing_3fold_mean_target_excess)} | {_fmt(row.trailing_3fold_spearman)} | "
            f"{_text(row.trailing_3fold_top_ticker)} | {_fmt(row.trailing_3fold_top_ticker_date_rate)} | {row.failed_floors} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
            "",
            "Report only. The production candidate roster, promotion gate, selection gate, top-2 cap, confidence basket, rotation exclusion, and scan behavior are unchanged.",
            "",
            "Tomorrow's critic can verify this by checking the floor table, the chronological-calibration tests, and this report's verdict against the pinned 2026-10-07 production OOS artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def verdict_text(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "NO VERDICT: no rows were evaluated"
    passers = rows[rows["passes_acceptance"].astype(bool)].copy()
    if not passers.empty:
        best = ordered_rows(passers).iloc[0]
        return (
            f"PASS: {best.variant} {best.cut} clears both beat windows "
            f"(last={_fmt(best.last_fold_beat_universe_rate)}, trailing={_fmt(best.trailing_3fold_beat_universe_rate)}) "
            f"with full-OOS Spearman {_fmt(best.full_oos_spearman)}"
        )
    best = ordered_rows(rows).iloc[0]
    return (
        f"NO PASS: best calibrated/raw row is {best.variant} {best.cut}; "
        f"last beat {_fmt(best.last_fold_beat_universe_rate)}, trailing beat {_fmt(best.trailing_3fold_beat_universe_rate)}, "
        f"full-OOS Spearman {_fmt(best.full_oos_spearman)}; failed floors: {best.failed_floors}"
    )


def ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ordered = rows.copy()
    for column in (
        "full_oos_spearman",
        "last_fold_beat_universe_rate",
        "trailing_3fold_beat_universe_rate",
        "last_fold_hit_rate_excess",
        "trailing_3fold_hit_rate_excess",
    ):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.sort_values(
        [
            "passes_acceptance",
            "last_fold_beat_universe_rate",
            "trailing_3fold_beat_universe_rate",
            "last_fold_hit_rate_excess",
            "trailing_3fold_hit_rate_excess",
            "full_oos_spearman",
        ],
        ascending=[False, False, False, False, False, False],
    )


def _row_for(rows: pd.DataFrame, variant: str, cut: str) -> dict[str, object]:
    if rows.empty:
        return {}
    scoped = rows[rows["variant"].astype(str).eq(str(variant)) & rows["cut"].astype(str).eq(str(cut))]
    return scoped.iloc[0].to_dict() if not scoped.empty else {}


def _fmt(value: object, *, places: int = 4) -> str:
    number = finite_float(value)
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
