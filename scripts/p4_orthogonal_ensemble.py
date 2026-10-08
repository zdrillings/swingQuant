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
    build_two_member_rank_blend,
    passes_acceptance,
)
from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.settings import get_settings


REPORT_PATH = Path("reports/p4_orthogonal_ensemble.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLDS = 3
TOP_N = 2
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
BLEND_WEIGHTS = tuple(round(value / 10.0, 1) for value in range(0, 11))


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
            raise SystemExit(
                "Refusing to audit a non-production target artifact: "
                + ", ".join(sorted(set(declared)))
            )
    frame = frame[frame["model_name"].astype(str).isin({RIDGE_MODEL, OVERNIGHT_MODEL})].copy()
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


def load_fixed_evaluation_keys(
    path: Path,
    *,
    calendar_dates: list[pd.Timestamp] | None = None,
) -> set[tuple[str, pd.Timestamp]]:
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
    rows: list[dict[str, object]] = []
    for variant, predictions in _variant_predictions(raw).items():
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
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _filter_to_evaluation_keys(
    predictions: pd.DataFrame,
    *,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    if predictions.empty or not evaluation_keys:
        return predictions.iloc[0:0].copy()
    working = predictions.copy()
    normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
    row_keys = list(zip(working["ticker"].astype(str), normalized_dates))
    return working.loc[[key in evaluation_keys for key in row_keys]].copy().reset_index(drop=True)


def _variant_predictions(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {
        RIDGE_MODEL: raw[raw["model_name"].eq(RIDGE_MODEL)].copy(),
        OVERNIGHT_MODEL: raw[raw["model_name"].eq(OVERNIGHT_MODEL)].copy(),
    }
    for ridge_weight in BLEND_WEIGHTS:
        overnight_weight = round(1.0 - ridge_weight, 1)
        if ridge_weight in (0.0, 1.0):
            continue
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


def render_report(
    *,
    raw: pd.DataFrame,
    calendar_dates: list[pd.Timestamp] | None,
    evaluation_keys: set[tuple[str, pd.Timestamp]],
    rows: pd.DataFrame,
) -> str:
    verdict = _verdict(rows)
    best = _best_row(rows)
    ridge = _row_for(rows, RIDGE_MODEL)
    lines = [
        "# P4 Orthogonal Ensemble Bakeoff",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: P4 orthogonal ensemble over ridge_adaptive and overnight_session_specialist",
        "- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- top_n: {TOP_N}",
        f"- fixed_label_calendar_horizon_days: {HORIZON_DAYS}",
        f"- fold_size_dates: {FOLD_SIZE}",
        f"- trailing_folds: {TRAILING_FOLDS}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- raw_dates_loaded: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- fixed_evaluation_keys: {len(evaluation_keys)}",
        "- acceptance: full_oos_spearman >= 0, 1fold/3fold hit_ex >= 0.02, beat >= 0.50, mean_excess >= 0, spearman >= 0",
        f"- verdict: {verdict}",
        f"- best_variant: {best.get('variant', 'n/a') if best else 'n/a'}",
        f"- best_full_oos_spearman: {_fmt(best.get('full_oos_spearman') if best else float('nan'))}",
        f"- ridge_baseline_full_oos_spearman: {_fmt(ridge.get('full_oos_spearman') if ridge else float('nan'))}",
        f"- best_delta_vs_ridge_full_oos_spearman: {_fmt(_delta(best, ridge, 'full_oos_spearman'))}",
        "",
        "## Acceptance Windows",
        "",
        "| variant | pass | rows | dates | full_oos_spearman | last_fold_spearman | last_fold_hit_ex | last_fold_beat | last_fold_mean_ex | trailing_3fold_spearman | trailing_3fold_hit_ex | trailing_3fold_beat | trailing_3fold_mean_ex |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if rows.empty:
        lines.append("| n/a | no | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        ordered = rows.sort_values(
            ["passes_acceptance", "trailing_3fold_beat_universe_rate", "trailing_3fold_hit_rate_excess", "full_oos_spearman"],
            ascending=[False, False, False, False],
        )
        for row in ordered.itertuples(index=False):
            lines.append(
                f"| {row.variant} | {'yes' if bool(row.passes_acceptance) else 'no'} | {int(row.rows)} | {int(row.dates)} | "
                f"{_fmt(row.full_oos_spearman)} | {_fmt(row.last_fold_spearman)} | {_fmt(row.last_fold_hit_rate_excess)} | "
                f"{_fmt(row.last_fold_beat_universe_rate)} | {_fmt(row.last_fold_mean_target_excess)} | "
                f"{_fmt(row.trailing_3fold_spearman)} | {_fmt(row.trailing_3fold_hit_rate_excess)} | "
                f"{_fmt(row.trailing_3fold_beat_universe_rate)} | {_fmt(row.trailing_3fold_mean_target_excess)} |"
            )
    lines.extend(
        [
            "",
            "## Implementation Decision",
            "",
            "This is an artifact audit only. The production candidate roster, promotion gate, selection gate, top-2 cap, and scan behavior are unchanged.",
            "",
            "Tomorrow's critic can verify the result by checking this report's `verdict`, `best_delta_vs_ridge_full_oos_spearman`, and the acceptance-window table against the 2026-10-07 production OOS artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _verdict(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no verdict: no variant rows"
    passers = rows[rows["passes_acceptance"].astype(bool)].copy()
    if not passers.empty:
        best = _best_row(passers)
        return (
            f"PASS: {best['variant']} clears full-OOS and 1fold/3fold hit/beat floors "
            f"(3fold beat={_fmt(best['trailing_3fold_beat_universe_rate'])})"
        )
    best = _best_row(rows)
    return (
        f"NO PASS: best 3fold beat is {best['variant']} at {_fmt(best['trailing_3fold_beat_universe_rate'])} "
        f"with hit_ex={_fmt(best['trailing_3fold_hit_rate_excess'])}"
    )


def _best_row(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {}
    ordered = rows.copy()
    for column in ("trailing_3fold_beat_universe_rate", "trailing_3fold_hit_rate_excess", "full_oos_spearman"):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    return ordered.sort_values(
        ["passes_acceptance", "trailing_3fold_beat_universe_rate", "trailing_3fold_hit_rate_excess", "full_oos_spearman"],
        ascending=[False, False, False, False],
    ).iloc[0].to_dict()


def _row_for(rows: pd.DataFrame, variant: str) -> dict[str, object]:
    if rows.empty:
        return {}
    scoped = rows[rows["variant"].astype(str).eq(str(variant))]
    return scoped.iloc[0].to_dict() if not scoped.empty else {}


def _delta(left: dict[str, object], right: dict[str, object], column: str) -> float:
    try:
        left_value = float(left.get(column))
        right_value = float(right.get(column))
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return float("nan")
    return left_value - right_value


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


if __name__ == "__main__":
    main()
