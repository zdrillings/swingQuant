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

from src.research.sector_neutral_selection import deoverlap_oos_predictions
from src.research.shortlist_model_service import B1_OVERNIGHT_FEATURES, PROMOTION_BASKET_SIZE, ShortlistModelService
from src.settings import get_settings


REPORT_PATH = Path("reports/p1_ridge_recency_autopsy.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
TARGET_COLUMN = "alpha_vs_sector_60d"
HORIZON_DAYS = 60
FOLD_SIZE = 20
RIDGE_MODEL = "ridge_adaptive"
OVERNIGHT_MODEL = "overnight_session_specialist"
BLEND_WEIGHTS = (0.7, 0.3)
TRAILING_FOLD_COUNT = 3


def main() -> None:
    raw = load_oos_predictions(OOS_PATH)
    calendar_dates = load_snapshot_calendar_dates()
    variants = build_variant_predictions(raw, calendar_dates=calendar_dates)
    rows = variant_metric_rows(variants)
    ablation = ridge_trailing_reason_ablation(variants.get(RIDGE_MODEL, pd.DataFrame()))
    REPORT_PATH.write_text(
        render_report(
            raw=raw,
            calendar_dates=calendar_dates,
            rows=rows,
            ablation=ablation,
        ),
        encoding="utf-8",
    )


def load_oos_predictions(path: Path) -> pd.DataFrame:
    columns = [
        "snapshot_date",
        "ticker",
        "sector",
        "model_name",
        "predicted_alpha",
        "model_top_reasons",
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


def build_variant_predictions(raw: pd.DataFrame, *, calendar_dates: list[pd.Timestamp] | None = None) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    resolved_calendar_dates = calendar_dates or raw["snapshot_date"].dropna().drop_duplicates().tolist()
    for model_name in (RIDGE_MODEL, OVERNIGHT_MODEL):
        model_frame = raw[raw["model_name"].eq(model_name)].copy()
        variants[model_name] = deoverlap_oos_predictions(
            model_frame,
            horizon_days=HORIZON_DAYS,
            calendar_dates=resolved_calendar_dates,
        )
    blend = rank_blend_predictions(
        raw,
        left_model=RIDGE_MODEL,
        right_model=OVERNIGHT_MODEL,
        left_weight=BLEND_WEIGHTS[0],
        right_weight=BLEND_WEIGHTS[1],
    )
    variants[f"ridge_overnight_rank_blend_{BLEND_WEIGHTS[0]:.1f}_{BLEND_WEIGHTS[1]:.1f}"] = deoverlap_oos_predictions(
        blend,
        horizon_days=HORIZON_DAYS,
        calendar_dates=resolved_calendar_dates,
    )
    return variants


def rank_blend_predictions(
    raw: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    left_weight: float,
    right_weight: float,
) -> pd.DataFrame:
    left = raw[raw["model_name"].eq(left_model)].copy()
    right = raw[raw["model_name"].eq(right_model)].copy()
    keys = ["snapshot_date", "ticker"]
    left["left_rank_score"] = (
        pd.to_numeric(left["predicted_alpha"], errors="coerce")
        .groupby(left["snapshot_date"])
        .rank(method="average", pct=True)
    )
    right["right_rank_score"] = (
        pd.to_numeric(right["predicted_alpha"], errors="coerce")
        .groupby(right["snapshot_date"])
        .rank(method="average", pct=True)
    )
    merged = left.merge(
        right[keys + ["right_rank_score"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged["predicted_alpha"] = (float(left_weight) * merged["left_rank_score"]) + (
        float(right_weight) * merged["right_rank_score"]
    )
    merged["model_name"] = f"ridge_overnight_rank_blend_{left_weight:.1f}_{right_weight:.1f}"
    merged["model_top_reasons"] = merged.get("model_top_reasons", "")
    return merged.drop(columns=["left_rank_score", "right_rank_score"], errors="ignore")


def variant_metric_rows(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    service = ShortlistModelService(db_manager=object())
    rows: list[dict[str, object]] = []
    for variant, predictions in variants.items():
        if predictions.empty:
            rows.append(
                {
                    "variant": variant,
                    "rows": 0,
                    "dates": 0,
                    "full_oos_spearman": float("nan"),
                    "trailing_3fold_spearman": float("nan"),
                    "trailing_3fold_hit_rate_excess": float("nan"),
                    "trailing_3fold_beat_universe_rate": float("nan"),
                    "last_fold_hit_rate_excess": float("nan"),
                    "last_fold_beat_universe_rate": float("nan"),
                }
            )
            continue
        full = service._evaluate_predictions(
            predictions=predictions,
            top_n=PROMOTION_BASKET_SIZE,
            target_column=TARGET_COLUMN,
            model_name=variant,
        )
        windows = service._rolling_window_summaries(
            predictions=predictions,
            target_column=TARGET_COLUMN,
            model_name=variant,
            top_n=PROMOTION_BASKET_SIZE,
            windows=(),
            fold_windows=(1, TRAILING_FOLD_COUNT),
            fold_size=FOLD_SIZE,
            include_full_oos=False,
        )
        window_by_name = {str(row["model"]): row for row in windows.to_dict(orient="records")}
        last = window_by_name.get(f"{variant}_last_fold", {})
        trailing = window_by_name.get(f"{variant}_trailing_3folds", {})
        rows.append(
            {
                "variant": variant,
                "rows": len(predictions.index),
                "dates": int(predictions["snapshot_date"].nunique()),
                "full_oos_spearman": full.get("spearman"),
                "trailing_3fold_spearman": trailing.get("spearman"),
                "trailing_3fold_hit_rate_excess": trailing.get("hit_rate_excess"),
                "trailing_3fold_beat_universe_rate": trailing.get("beat_universe_rate"),
                "last_fold_hit_rate_excess": last.get("hit_rate_excess"),
                "last_fold_beat_universe_rate": last.get("beat_universe_rate"),
            }
        )
    return pd.DataFrame(rows)


def ridge_trailing_reason_ablation(ridge_predictions: pd.DataFrame) -> pd.DataFrame:
    if ridge_predictions.empty or "model_top_reasons" not in ridge_predictions.columns:
        return pd.DataFrame()
    unique_dates = sorted(ridge_predictions["snapshot_date"].drop_duplicates().tolist())
    selected_dates = unique_dates[-min(TRAILING_FOLD_COUNT * FOLD_SIZE, len(unique_dates)) :]
    scoped = ridge_predictions[ridge_predictions["snapshot_date"].isin(selected_dates)].copy()
    rows: list[dict[str, object]] = []
    for snapshot_date, day_frame in scoped.groupby("snapshot_date", sort=True):
        ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(PROMOTION_BASKET_SIZE).copy()
        universe = pd.to_numeric(day_frame[TARGET_COLUMN], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        if ordered.empty or universe.empty:
            continue
        for _, pick in ordered.iterrows():
            target = pd.to_numeric(pd.Series([pick.get(TARGET_COLUMN)]), errors="coerce").clip(lower=-1.0, upper=1.0).iloc[0]
            if pd.isna(target):
                continue
            reason_text = str(pick.get("model_top_reasons") or "")
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "ticker": str(pick.get("ticker")),
                    "target": float(target),
                    "universe_mean": float(universe.mean()),
                    "has_b1_reason": has_b1_reason(reason_text),
                }
            )
    if not rows:
        return pd.DataFrame()
    picks = pd.DataFrame(rows)
    output: list[dict[str, object]] = []
    for label, group in (
        ("all_ridge_adaptive_top2", picks),
        ("b1_reason_pick", picks[picks["has_b1_reason"]].copy()),
        ("non_b1_reason_pick", picks[~picks["has_b1_reason"]].copy()),
    ):
        if group.empty:
            output.append({"bucket": label, "picks": 0, "mean_target": float("nan"), "hit_rate": float("nan"), "beat_pick_rate": float("nan")})
            continue
        output.append(
            {
                "bucket": label,
                "picks": len(group.index),
                "mean_target": float(group["target"].mean()),
                "hit_rate": float((group["target"] > 0.0).mean()),
                "beat_pick_rate": float((group["target"] > group["universe_mean"]).mean()),
            }
        )
    return pd.DataFrame(output)


def has_b1_reason(reason_text: str) -> bool:
    base_names = set(B1_OVERNIGHT_FEATURES)
    return any(feature in str(reason_text) for feature in base_names)


def render_report(
    *,
    raw: pd.DataFrame,
    calendar_dates: list[pd.Timestamp] | None = None,
    rows: pd.DataFrame,
    ablation: pd.DataFrame,
) -> str:
    verdict = _verdict(rows)
    lines = [
        "# P1 Ridge Recency Autopsy",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- idea: P1 autopsy ridge_adaptive recency and close the 3fold hit/beat gap",
        "- data_access: read-only OOS artifact; no `./sq` write command; no `data/` mutation",
        f"- source_artifact: {OOS_PATH}",
        f"- target_column: {TARGET_COLUMN}",
        f"- raw_rows_loaded: {len(raw.index)}",
        f"- raw_dates_loaded: {int(raw['snapshot_date'].nunique()) if not raw.empty else 0}",
        f"- calendar_dates_loaded: {len(calendar_dates or [])}",
        f"- ridge_b1_family: {', '.join(B1_OVERNIGHT_FEATURES)}",
        f"- tested_blend: {BLEND_WEIGHTS[0]:.1f} ridge_adaptive rank + {BLEND_WEIGHTS[1]:.1f} overnight_session_specialist rank",
        "- acceptance_floor_3fold_hit_rate_excess: +0.0200",
        "- acceptance_floor_3fold_beat_universe_rate: +0.5000",
        f"- verdict: {verdict}",
        "",
        "## Variant Metrics",
        "",
        "| variant | rows | dates | full_oos_spearman | trailing_3fold_spearman | trailing_3fold_hit_rate_excess | trailing_3fold_beat_universe_rate | last_fold_hit_rate_excess | last_fold_beat_universe_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if rows.empty:
        lines.append("| n/a | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in rows.sort_values("variant").itertuples(index=False):
            lines.append(
                f"| {row.variant} | {int(row.rows)} | {int(row.dates)} | {_fmt(row.full_oos_spearman)} | "
                f"{_fmt(row.trailing_3fold_spearman)} | {_fmt(row.trailing_3fold_hit_rate_excess)} | "
                f"{_fmt(row.trailing_3fold_beat_universe_rate)} | {_fmt(row.last_fold_hit_rate_excess)} | "
                f"{_fmt(row.last_fold_beat_universe_rate)} |"
            )
    lines.extend(
        [
            "",
            "## Ridge Trailing-3fold Reason Ablation",
            "",
            "| bucket | picks | mean_target | hit_rate | beat_pick_rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if ablation.empty:
        lines.append("| n/a | 0 | n/a | n/a | n/a |")
    else:
        for row in ablation.itertuples(index=False):
            lines.append(
                f"| {row.bucket} | {int(row.picks)} | {_fmt(row.mean_target)} | {_fmt(row.hit_rate)} | {_fmt(row.beat_pick_rate)} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Production code now gives ridge_adaptive the explicit B1 overnight family and applies B1-specific observation counts during the fold-local IC screen.",
            "- The blend row is an audit variant only; it is not added to the production candidate roster.",
            "- Tomorrow's pipeline should verify whether ridge_adaptive no longer loses B1 columns to the global observation floor and whether the live gate windows improve.",
            "",
        ]
    )
    return "\n".join(lines)


def _verdict(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "no verdict: no variant rows"
    finite = rows.copy()
    finite["trailing_3fold_hit_rate_excess"] = pd.to_numeric(finite["trailing_3fold_hit_rate_excess"], errors="coerce")
    finite["trailing_3fold_beat_universe_rate"] = pd.to_numeric(finite["trailing_3fold_beat_universe_rate"], errors="coerce")
    finite["full_oos_spearman"] = pd.to_numeric(finite["full_oos_spearman"], errors="coerce")
    passers = finite[
        finite["full_oos_spearman"].ge(0.0)
        & finite["trailing_3fold_hit_rate_excess"].ge(0.0200)
        & finite["trailing_3fold_beat_universe_rate"].ge(0.5000)
    ]
    if not passers.empty:
        best = passers.sort_values(["trailing_3fold_hit_rate_excess", "trailing_3fold_beat_universe_rate"], ascending=False).iloc[0]
        return (
            f"PASS on artifact audit: {best['variant']} hit_ex={_fmt(best['trailing_3fold_hit_rate_excess'])}, "
            f"beat={_fmt(best['trailing_3fold_beat_universe_rate'])}"
        )
    best = finite.sort_values(["trailing_3fold_hit_rate_excess", "trailing_3fold_beat_universe_rate"], ascending=False).iloc[0]
    return (
        f"no artifact passer: best {best['variant']} hit_ex={_fmt(best['trailing_3fold_hit_rate_excess'])}, "
        f"beat={_fmt(best['trailing_3fold_beat_universe_rate'])}"
    )


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
