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

from src.research.basket_size_sweep import summarize_basket_sizes
from src.research.beat_calibration import chronological_isotonic_probability, finite_float
from src.research.orthogonal_ensemble import build_two_member_rank_blend
from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.settings import get_settings


REPORT_PATH = Path("reports/basket_size_sweep.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLDS = 3
BASKET_SIZES = (2, 3, 4, 5, 6)
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
BLEND_VARIANT = "ridge_overnight_rank_blend_0.4_0.6"
CALIBRATED_COLUMN = "calibrated_p_beat_sector_oos"


def main() -> None:
    raw = load_oos_predictions(OOS_PATH)
    calendar_dates = load_snapshot_calendar_dates()
    evaluation_keys = load_fixed_evaluation_keys(OOS_PATH, calendar_dates=calendar_dates)
    rows = summarize_variants(build_variants(raw), evaluation_keys=evaluation_keys)
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


def build_variants(raw: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
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
    return {
        (RIDGE_MODEL, "raw_score"): ridge,
        (RIDGE_MODEL, "calibrated_p"): ridge,
        (BLEND_VARIANT, "raw_score"): blend,
    }


def summarize_variants(
    variants: dict[tuple[str, str], pd.DataFrame],
    *,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (variant, score_mode), frame in variants.items():
        honest = filter_to_evaluation_keys(frame, evaluation_keys=evaluation_keys)
        if score_mode == "calibrated_p":
            honest = chronological_isotonic_probability(
                honest,
                target_column=TARGET_COLUMN,
                output_column=CALIBRATED_COLUMN,
                min_train_rows=50,
            )
            honest["predicted_alpha"] = pd.to_numeric(honest[CALIBRATED_COLUMN], errors="coerce")
        summary = summarize_basket_sizes(
            honest,
            basket_sizes=BASKET_SIZES,
            target_column=TARGET_COLUMN,
            fold_size=FOLD_SIZE,
            trailing_folds=TRAILING_FOLDS,
        )
        summary.insert(0, "score_mode", score_mode)
        summary.insert(0, "variant", variant)
        summary.insert(2, "rows", len(honest.index))
        summary.insert(3, "dates", int(honest["snapshot_date"].nunique()) if not honest.empty else 0)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def filter_to_evaluation_keys(predictions: pd.DataFrame, *, evaluation_keys: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    if predictions.empty or not evaluation_keys:
        return predictions.iloc[0:0].copy()
    working = predictions.copy()
    normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    row_keys = list(zip(working["ticker"].astype(str), normalized_dates))
    return working.loc[[key in evaluation_keys for key in row_keys]].copy().reset_index(drop=True)


def render_report(
    *,
    raw: pd.DataFrame,
    calendar_dates: list[pd.Timestamp] | None,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
    rows: pd.DataFrame,
) -> str:
    verdict = verdict_text(rows)
    lines = [
        "# Basket Size Sweep",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: BASKET-SIZE audit of gate floor statistics from top-2 through top-6",
        "- data_access: read-only OOS artifact and DuckDB calendar; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- output_report: {REPORT_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- variants: {RIDGE_MODEL} raw score; {RIDGE_MODEL} chronological calibrated-P; {BLEND_VARIANT} raw score",
        f"- basket_sizes: {', '.join(str(size) for size in BASKET_SIZES)}",
        f"- fixed_label_calendar_horizon_days: {HORIZON_DAYS}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- raw_dates_loaded: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- fixed_evaluation_keys: {len(evaluation_keys)}",
        "- beat_clearance_test: last_fold_beat_universe_rate >= 0.50 AND trailing_3fold_beat_universe_rate >= 0.50",
        f"- verdict: {verdict}",
        "",
        "## Basket Matrix",
        "",
        "| variant | score_mode | top_n | rows | dates | full_sp | last_hit_ex | last_beat | last_mean_ex | last_sp | last_top_rate | trailing_hit_ex | trailing_beat | trailing_mean_ex | trailing_sp | trailing_top_rate | beat_clear | window_floor_clear |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered_rows(rows).itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.score_mode} | {int(row.basket_size)} | {int(row.rows)} | {int(row.dates)} | "
            f"{_fmt(row.full_oos_spearman)} | {_fmt(row.last_fold_hit_rate_excess)} | "
            f"{_fmt(row.last_fold_beat_universe_rate)} | {_fmt(row.last_fold_mean_target_excess)} | "
            f"{_fmt(row.last_fold_spearman)} | {_fmt(row.last_fold_top_ticker_date_rate)} | "
            f"{_fmt(row.trailing_3fold_hit_rate_excess)} | {_fmt(row.trailing_3fold_beat_universe_rate)} | "
            f"{_fmt(row.trailing_3fold_mean_target_excess)} | {_fmt(row.trailing_3fold_spearman)} | "
            f"{_fmt(row.trailing_3fold_top_ticker_date_rate)} | "
            f"{'yes' if bool(row.clears_both_beat_windows) else 'no'} | "
            f"{'yes' if bool(row.clears_window_floors) else 'no'} |"
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
            "Tomorrow's critic can verify this by checking that every requested top-N row appears in the matrix and that the verdict names the first basket size clearing both beat windows, or says none.",
            "",
        ]
    )
    return "\n".join(lines)


def verdict_text(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "NO VERDICT: no basket-size rows were evaluated"
    passers = rows[rows["clears_both_beat_windows"].astype(bool)].copy()
    best_spearman = _best_full_spearman_by_size(rows)
    if passers.empty:
        closest = ordered_rows(rows).iloc[0]
        return (
            "NO BEAT CLEAR: no tested basket size clears beat >= 0.50 on both last-fold and trailing-3fold windows; "
            f"closest row is {closest.variant} {closest.score_mode} top-{int(closest.basket_size)} "
            f"(last beat {_fmt(closest.last_fold_beat_universe_rate)}, trailing beat {_fmt(closest.trailing_3fold_beat_universe_rate)}). "
            f"Full-OOS Spearman by size: {best_spearman}."
        )
    ordered = passers.sort_values(["basket_size", "full_oos_spearman"], ascending=[True, False])
    first = ordered.iloc[0]
    return (
        f"BEAT CLEAR: first clearing size is top-{int(first.basket_size)} via {first.variant} {first.score_mode} "
        f"(last beat {_fmt(first.last_fold_beat_universe_rate)}, trailing beat {_fmt(first.trailing_3fold_beat_universe_rate)}, "
        f"full-OOS Spearman {_fmt(first.full_oos_spearman)}). Full-OOS Spearman by size: {best_spearman}."
    )


def ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ordered = rows.copy()
    for column in (
        "last_fold_beat_universe_rate",
        "trailing_3fold_beat_universe_rate",
        "last_fold_hit_rate_excess",
        "trailing_3fold_hit_rate_excess",
        "full_oos_spearman",
    ):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.sort_values(
        [
            "clears_both_beat_windows",
            "basket_size",
            "last_fold_beat_universe_rate",
            "trailing_3fold_beat_universe_rate",
            "last_fold_hit_rate_excess",
            "trailing_3fold_hit_rate_excess",
            "full_oos_spearman",
        ],
        ascending=[False, True, False, False, False, False, False],
    )


def _best_full_spearman_by_size(rows: pd.DataFrame) -> str:
    parts: list[str] = []
    for basket_size in sorted(rows["basket_size"].dropna().astype(int).unique().tolist()):
        scoped = rows[rows["basket_size"].astype(int).eq(int(basket_size))].copy()
        scoped["full_oos_spearman"] = pd.to_numeric(scoped["full_oos_spearman"], errors="coerce")
        if scoped["full_oos_spearman"].notna().any():
            best = scoped.sort_values("full_oos_spearman", ascending=False).iloc[0]
            parts.append(f"top-{basket_size} {best.variant}/{best.score_mode} {_fmt(best.full_oos_spearman)}")
    return "; ".join(parts) if parts else "n/a"


def _fmt(value: object, *, places: int = 4) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


if __name__ == "__main__":
    main()
