from __future__ import annotations

from dataclasses import dataclass
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

from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE, ShortlistModelService
from src.settings import get_settings


REPORT_PATH = Path("reports/basket_calibration.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
AUDIT_MODEL = "ridge_adaptive"
HORIZON_DAYS = 60
FOLD_SIZE = 20
TRAILING_FOLD_COUNT = 3
HIT_EXCESS_FLOOR = 0.0200
BEAT_UNIVERSE_FLOOR = 0.5000
SCORE_THRESHOLD_QUANTILES = (0.90, 0.925, 0.95, 0.975, 0.99)
CALIBRATED_P_THRESHOLDS = (0.50, 0.525, 0.55, 0.575, 0.60)


@dataclass(frozen=True)
class BasketVariant:
    variant: str
    score_column: str
    top_n: int
    min_score: float | None = None
    threshold_label: str = "none"


def main() -> None:
    raw = load_oos_predictions(OOS_PATH)
    calendar_dates = load_snapshot_calendar_dates()
    ridge = prepare_ridge_predictions(raw, calendar_dates=calendar_dates)
    calibrated = add_chronological_isotonic_probability(ridge, target_column=TARGET_COLUMN)
    variants = build_variants(calibrated)
    rows = evaluate_variants(calibrated, variants=variants, target_column=TARGET_COLUMN)
    REPORT_PATH.write_text(
        render_report(raw=raw, ridge=calibrated, rows=rows, variants=variants, calendar_dates=calendar_dates),
        encoding="utf-8",
    )


def load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = [
        "snapshot_date",
        "ticker",
        "sector",
        "model_name",
        "predicted_alpha",
        TARGET_COLUMN,
        "artifact_evaluation_target_column",
    ]
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = {"snapshot_date", "ticker", "model_name", "predicted_alpha", TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise SystemExit(f"OOS artifact is missing required columns: {', '.join(sorted(missing))}")
    if "artifact_evaluation_target_column" in frame.columns:
        declared = frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist()
        if declared and set(declared) != {TARGET_COLUMN}:
            raise SystemExit("Refusing to audit a non-production target artifact: " + ", ".join(sorted(set(declared))))
    frame = frame[frame["model_name"].astype(str).eq(AUDIT_MODEL)].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["sector"] = frame.get("sector", "").fillna("").astype(str).str.strip()
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "predicted_alpha", TARGET_COLUMN]).copy()


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


def prepare_ridge_predictions(raw: pd.DataFrame, *, calendar_dates: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    resolved_calendar_dates = calendar_dates or raw["snapshot_date"].dropna().drop_duplicates().tolist()
    return deoverlap_oos_predictions(
        raw,
        horizon_days=HORIZON_DAYS,
        calendar_dates=resolved_calendar_dates,
    )


def add_chronological_isotonic_probability(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score_column: str = "predicted_alpha",
    min_train_rows: int = 50,
) -> pd.DataFrame:
    working = frame.copy()
    working["calibrated_p_beat_sector_oos"] = float("nan")
    if working.empty:
        return working
    dates = sorted(working["snapshot_date"].dropna().drop_duplicates().tolist())
    for snapshot_date in dates:
        train = working[working["snapshot_date"].lt(snapshot_date)].copy()
        test_mask = working["snapshot_date"].eq(snapshot_date)
        test_scores = pd.to_numeric(working.loc[test_mask, score_column], errors="coerce")
        train_scores = pd.to_numeric(train.get(score_column), errors="coerce")
        train_labels = (pd.to_numeric(train.get(target_column), errors="coerce") > 0.0).astype(float)
        valid_train = train_scores.notna() & train_labels.notna()
        fallback = float(train_labels[valid_train].mean()) if int(valid_train.sum()) else 0.5
        if int(valid_train.sum()) >= int(min_train_rows) and train_labels[valid_train].nunique(dropna=True) > 1:
            try:
                from sklearn.isotonic import IsotonicRegression

                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(train_scores[valid_train].astype(float), train_labels[valid_train].astype(float))
                calibrated = calibrator.predict(test_scores.astype(float))
                working.loc[test_mask, "calibrated_p_beat_sector_oos"] = calibrated
                continue
            except Exception:
                pass
        working.loc[test_mask, "calibrated_p_beat_sector_oos"] = fallback
    working["calibrated_p_beat_sector_oos"] = pd.to_numeric(
        working["calibrated_p_beat_sector_oos"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    return working


def build_variants(frame: pd.DataFrame) -> list[BasketVariant]:
    variants = [
        BasketVariant("as_is_top2_by_score", "predicted_alpha", PROMOTION_BASKET_SIZE, None, "none"),
        BasketVariant("top1_by_score", "predicted_alpha", 1, None, "none"),
    ]
    thresholds = score_threshold_values(frame, score_column="predicted_alpha", quantiles=SCORE_THRESHOLD_QUANTILES)
    for quantile, threshold in zip(SCORE_THRESHOLD_QUANTILES, thresholds, strict=False):
        variants.append(
            BasketVariant(
                f"score_gate_q{int(round(quantile * 1000)):03d}",
                "predicted_alpha",
                PROMOTION_BASKET_SIZE,
                threshold,
                f"q{quantile:.3f}={threshold:.6f}",
            )
        )
    for threshold in CALIBRATED_P_THRESHOLDS:
        variants.append(
            BasketVariant(
                f"calibrated_p_gate_{threshold:.3f}",
                "calibrated_p_beat_sector_oos",
                PROMOTION_BASKET_SIZE,
                threshold,
                f"p>={threshold:.3f}",
            )
        )
    return variants


def score_threshold_values(
    frame: pd.DataFrame,
    *,
    score_column: str,
    quantiles: tuple[float, ...],
) -> list[float]:
    scores = pd.to_numeric(frame.get(score_column), errors="coerce").dropna()
    if scores.empty:
        return [float("nan") for _ in quantiles]
    return [float(scores.quantile(float(quantile))) for quantile in quantiles]


def evaluate_variants(
    frame: pd.DataFrame,
    *,
    variants: list[BasketVariant],
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
    trailing_dates = dates[-min(TRAILING_FOLD_COUNT * FOLD_SIZE, len(dates)) :]
    last_dates = dates[-min(FOLD_SIZE, len(dates)) :]
    for variant in variants:
        full = evaluate_selection_window(
            frame,
            variant=variant,
            target_column=target_column,
            selected_dates=dates,
            label="full_oos",
        )
        trailing = evaluate_selection_window(
            frame,
            variant=variant,
            target_column=target_column,
            selected_dates=trailing_dates,
            label="trailing_3fold",
        )
        last = evaluate_selection_window(
            frame,
            variant=variant,
            target_column=target_column,
            selected_dates=last_dates,
            label="last_fold",
        )
        row = {
            "variant": variant.variant,
            "score_column": variant.score_column,
            "top_n": variant.top_n,
            "threshold": variant.threshold_label,
            "full_oos_spearman": full["spearman"],
            "full_oos_dates": full["dates"],
            "full_oos_active_dates": full["active_dates"],
            "full_oos_total_picks": full["total_picks"],
            "full_oos_avg_pick_count": full["avg_pick_count"],
            "trailing_3fold_hit_rate_excess": trailing["hit_rate_excess"],
            "trailing_3fold_beat_universe_rate": trailing["beat_universe_rate"],
            "trailing_3fold_mean_target_excess": trailing["mean_target_excess"],
            "trailing_3fold_active_dates": trailing["active_dates"],
            "trailing_3fold_total_picks": trailing["total_picks"],
            "trailing_3fold_avg_pick_count": trailing["avg_pick_count"],
            "last_fold_hit_rate_excess": last["hit_rate_excess"],
            "last_fold_beat_universe_rate": last["beat_universe_rate"],
            "last_fold_mean_target_excess": last["mean_target_excess"],
            "last_fold_active_dates": last["active_dates"],
            "last_fold_total_picks": last["total_picks"],
            "last_fold_avg_pick_count": last["avg_pick_count"],
        }
        row["clears_trailing_floors"] = bool(
            _finite(row["full_oos_spearman"])
            and float(row["full_oos_spearman"]) >= 0.0
            and _finite(row["trailing_3fold_hit_rate_excess"])
            and float(row["trailing_3fold_hit_rate_excess"]) >= HIT_EXCESS_FLOOR
            and _finite(row["trailing_3fold_beat_universe_rate"])
            and float(row["trailing_3fold_beat_universe_rate"]) >= BEAT_UNIVERSE_FLOOR
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_selection_window(
    frame: pd.DataFrame,
    *,
    variant: BasketVariant,
    target_column: str,
    selected_dates: list[pd.Timestamp],
    label: str,
) -> dict[str, object]:
    service = ShortlistModelService(db_manager=object())
    cost_fraction = service._round_trip_cost_fraction()
    scoped = frame[frame["snapshot_date"].isin(selected_dates)].copy()
    daily_rows: list[dict[str, object]] = []
    for snapshot_date, day_frame in scoped.groupby("snapshot_date", sort=True):
        day_frame = day_frame.copy()
        day_frame[variant.score_column] = pd.to_numeric(day_frame.get(variant.score_column), errors="coerce")
        universe_target = pd.to_numeric(day_frame[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        score_valid = day_frame.dropna(subset=[variant.score_column, target_column]).copy()
        if variant.min_score is not None and math.isfinite(float(variant.min_score)):
            score_valid = score_valid[score_valid[variant.score_column].ge(float(variant.min_score))].copy()
        selected = score_valid.sort_values([variant.score_column, "ticker"], ascending=[False, True]).head(int(variant.top_n))
        spearman = _daily_spearman(day_frame, score_column=variant.score_column, target_column=target_column)
        base_row = {
            "date": pd.Timestamp(snapshot_date),
            "pick_count": len(selected.index),
            "spearman": spearman,
        }
        if selected.empty or universe_target.empty:
            daily_rows.append(
                {
                    **base_row,
                    "mean_target_excess": float("nan"),
                    "hit_rate_excess": float("nan"),
                    "beat_universe": float("nan"),
                }
            )
            continue
        target = pd.to_numeric(selected[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        if target.empty:
            daily_rows.append(
                {
                    **base_row,
                    "mean_target_excess": float("nan"),
                    "hit_rate_excess": float("nan"),
                    "beat_universe": float("nan"),
                }
            )
            continue
        net_target = target - cost_fraction
        net_universe = universe_target - cost_fraction
        daily_rows.append(
            {
                **base_row,
                "mean_target_excess": float(net_target.mean() - net_universe.mean()),
                "hit_rate_excess": float((net_target > 0.0).mean() - (net_universe > 0.0).mean()),
                "beat_universe": float(net_target.mean() > net_universe.mean()),
            }
        )
    if not daily_rows:
        return _empty_window(label)
    daily = pd.DataFrame(daily_rows)
    active = daily[pd.to_numeric(daily["pick_count"], errors="coerce").gt(0)].copy()
    return {
        "window": label,
        "dates": int(len(daily.index)),
        "active_dates": int(len(active.index)),
        "total_picks": int(pd.to_numeric(daily["pick_count"], errors="coerce").fillna(0).sum()),
        "avg_pick_count": float(pd.to_numeric(daily["pick_count"], errors="coerce").fillna(0).mean()),
        "mean_target_excess": _mean_or_nan(daily["mean_target_excess"]),
        "hit_rate_excess": _mean_or_nan(daily["hit_rate_excess"]),
        "beat_universe_rate": _mean_or_nan(daily["beat_universe"]),
        "spearman": _mean_or_nan(daily["spearman"]),
    }


def _daily_spearman(frame: pd.DataFrame, *, score_column: str, target_column: str) -> float:
    scores = pd.to_numeric(frame.get(score_column), errors="coerce")
    targets = pd.to_numeric(frame.get(target_column), errors="coerce")
    valid = scores.notna() & targets.notna()
    if int(valid.sum()) < 3 or scores[valid].nunique(dropna=True) <= 1 or targets[valid].nunique(dropna=True) <= 1:
        return float("nan")
    corr = scores[valid].corr(targets[valid], method="spearman")
    return float(corr) if pd.notna(corr) and math.isfinite(float(corr)) else float("nan")


def _empty_window(label: str) -> dict[str, object]:
    return {
        "window": label,
        "dates": 0,
        "active_dates": 0,
        "total_picks": 0,
        "avg_pick_count": float("nan"),
        "mean_target_excess": float("nan"),
        "hit_rate_excess": float("nan"),
        "beat_universe_rate": float("nan"),
        "spearman": float("nan"),
    }


def render_report(
    *,
    raw: pd.DataFrame,
    ridge: pd.DataFrame,
    rows: pd.DataFrame,
    variants: list[BasketVariant],
    calendar_dates: list[pd.Timestamp] | None = None,
) -> str:
    verdict = verdict_text(rows)
    as_is = rows[rows["variant"].eq("as_is_top2_by_score")]
    as_is_picks = int(as_is.iloc[0]["trailing_3fold_total_picks"]) if not as_is.empty else 0
    lines = [
        "# Basket Calibration",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: BASKET-CAL from the 2026-10-06 critique",
        "- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- model: {AUDIT_MODEL}",
        f"- target_column: {TARGET_COLUMN}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- pinned_rows_audited: {len(ridge.index)}",
        f"- pinned_dates_audited: {int(ridge['snapshot_date'].nunique()) if not ridge.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- variants_tested: {len(variants)}",
        f"- trailing_3fold_floor_hit_rate_excess: {_fmt(HIT_EXCESS_FLOOR)}",
        f"- trailing_3fold_floor_beat_universe_rate: {_fmt(BEAT_UNIVERSE_FLOOR)}",
        f"- verdict: {verdict}",
        "",
        "## Variant Table",
        "",
        "| variant | score | cut | top_n | full_oos_spearman | trailing_hit_excess | trailing_beat_rate | trailing_mean_excess | trailing_picks | trailing_avg_picks | pick_cost_vs_as_is | last_hit_excess | last_beat_rate | last_mean_excess | last_picks | clears |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if rows.empty:
        lines.append("| n/a | n/a | n/a | 0 | n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a | n/a | n/a | 0 | no |")
    else:
        ordered = rows.sort_values(
            ["clears_trailing_floors", "trailing_3fold_hit_rate_excess", "trailing_3fold_beat_universe_rate"],
            ascending=[False, False, False],
        )
        for row in ordered.itertuples(index=False):
            trailing_picks = int(row.trailing_3fold_total_picks)
            pick_cost = trailing_picks - as_is_picks
            lines.append(
                f"| {row.variant} | {row.score_column} | {row.threshold} | {int(row.top_n)} | "
                f"{_fmt(row.full_oos_spearman)} | {_fmt(row.trailing_3fold_hit_rate_excess)} | "
                f"{_fmt(row.trailing_3fold_beat_universe_rate)} | {_fmt(row.trailing_3fold_mean_target_excess)} | "
                f"{trailing_picks} | {_fmt(row.trailing_3fold_avg_pick_count)} | {pick_cost:+d} | "
                f"{_fmt(row.last_fold_hit_rate_excess)} | {_fmt(row.last_fold_beat_universe_rate)} | "
                f"{_fmt(row.last_fold_mean_target_excess)} | {int(row.last_fold_total_picks)} | "
                f"{'yes' if bool(row.clears_trailing_floors) else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Score-gate thresholds are fixed quantiles of ridge_adaptive OOS scores; names such as `q0.950` identify the scanned cut.",
            "- Calibrated-P rows use chronological isotonic calibration: each OOS date is scored from earlier OOS dates only.",
            "- Empty gated slots stay empty and are reflected in total picks and average picks; floor rates are measured on dates where the cut selected at least one name.",
            "- This is report-only. Production selection, promotion gates, top-2 basket size, confidence basket, rotation exclusion, and strategy files are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def verdict_text(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no verdict: no variant rows"
    passers = rows[rows["clears_trailing_floors"].astype(bool)].copy()
    if passers.empty:
        best = rows.sort_values(["trailing_3fold_hit_rate_excess", "trailing_3fold_beat_universe_rate"], ascending=False).iloc[0]
        return (
            f"NO PASS; best trailing hit_ex={_fmt(best['trailing_3fold_hit_rate_excess'])}, "
            f"beat={_fmt(best['trailing_3fold_beat_universe_rate'])} on {best['variant']}"
        )
    passers["pick_cost_abs"] = passers["trailing_3fold_total_picks"].astype(float).sub(
        float(rows.loc[rows["variant"].eq("as_is_top2_by_score"), "trailing_3fold_total_picks"].iloc[0])
    ).abs()
    best = passers.sort_values(
        ["pick_cost_abs", "trailing_3fold_hit_rate_excess", "trailing_3fold_beat_universe_rate"],
        ascending=[True, False, False],
    ).iloc[0]
    return (
        f"PASS: {best['variant']} clears trailing floors with hit_ex={_fmt(best['trailing_3fold_hit_rate_excess'])}, "
        f"beat={_fmt(best['trailing_3fold_beat_universe_rate'])}, full Spearman={_fmt(best['full_oos_spearman'])}, "
        f"trailing picks={int(best['trailing_3fold_total_picks'])}"
    )


def _mean_or_nan(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else float("nan")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
