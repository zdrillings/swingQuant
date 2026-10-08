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

from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE, ShortlistModelService
from src.settings import load_feature_config
from src.utils.shortlist_selection_gate import (
    ShortlistSelectionGate,
    apply_score_quantile_gate,
    latest_score_quantile_threshold,
    rolling_score_quantile_thresholds,
)


REPORT_PATH = Path("reports/gated_selection_dryrun_2026-10-07.md")
OOS_PATH = Path("reports/shortlist_model_oos_predictions.csv")
LIVE_PATH = Path("reports/shortlist_model_live_predictions.csv")
AUDIT_MODEL = "ridge_adaptive"
TARGET_COLUMN = "alpha_vs_sector_60d"
TEST_WINDOW_DATES = 20


def main() -> None:
    config = load_feature_config()
    gate = _load_selection_gate(config)
    promotion_gate = _load_promotion_gate(config)
    oos = load_oos_predictions(OOS_PATH, model_name=AUDIT_MODEL)
    service = ShortlistModelService(db_manager=object())
    summaries = acceptance_windows(
        oos,
        service=service,
        gate=gate,
        promotion_top_n=PROMOTION_BASKET_SIZE,
        target_column=TARGET_COLUMN,
    )
    floor_rows = floor_verdict_rows(summaries, promotion_gate=promotion_gate)
    passes = bool(floor_rows) and all(row["passes"] for row in floor_rows)
    live = load_live_predictions(LIVE_PATH)
    live_threshold = latest_score_quantile_threshold(
        oos,
        score_column="predicted_alpha",
        quantile=gate.quantile,
        lookback_sessions=gate.lookback_sessions,
    )
    gated_live = apply_score_quantile_gate(
        live,
        gate=gate,
        threshold=live_threshold,
        score_column="predicted_alpha",
    ).sort_values(["predicted_alpha", "ticker"], ascending=[False, True])
    REPORT_PATH.write_text(
        render_report(
            gate=gate,
            oos=oos,
            summaries=summaries,
            floor_rows=floor_rows,
            passes=passes,
            live=live,
            gated_live=gated_live,
            live_threshold=live_threshold,
        ),
        encoding="utf-8",
    )


def load_oos_predictions(path: Path, *, model_name: str, chunksize: int = 250_000) -> pd.DataFrame:
    columns = {
        "snapshot_date",
        "ticker",
        "sector",
        "model_name",
        "predicted_alpha",
        TARGET_COLUMN,
        "artifact_evaluation_target_column",
    }
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda column: column in columns, chunksize=chunksize):
        if "model_name" not in chunk.columns:
            raise SystemExit("OOS artifact is missing model_name.")
        scoped = chunk[chunk["model_name"].astype(str).eq(model_name)].copy()
        if not scoped.empty:
            frames.append(scoped)
    if not frames:
        raise SystemExit(f"OOS artifact contains no rows for {model_name}.")
    frame = pd.concat(frames, ignore_index=True)
    missing = {"snapshot_date", "ticker", "predicted_alpha", TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise SystemExit(f"OOS artifact is missing required columns: {', '.join(sorted(missing))}")
    if "artifact_evaluation_target_column" in frame.columns:
        declared = frame["artifact_evaluation_target_column"].dropna().astype(str).unique().tolist()
        if declared and set(declared) != {TARGET_COLUMN}:
            raise SystemExit("Refusing non-production evaluation target artifact: " + ", ".join(sorted(set(declared))))
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return frame.dropna(subset=["snapshot_date", "ticker", "predicted_alpha"]).copy()


def load_live_predictions(path: Path) -> pd.DataFrame:
    columns = {"snapshot_date", "ticker", "sector", "predicted_alpha", "calibrated_p_beat_sector"}
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = {"snapshot_date", "ticker", "predicted_alpha"} - set(frame.columns)
    if missing:
        raise SystemExit(f"Live artifact is missing required columns: {', '.join(sorted(missing))}")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["predicted_alpha"] = pd.to_numeric(frame["predicted_alpha"], errors="coerce")
    if "sector" not in frame.columns:
        frame["sector"] = ""
    return frame.dropna(subset=["snapshot_date", "ticker", "predicted_alpha"]).copy()


def acceptance_windows(
    predictions: pd.DataFrame,
    *,
    service: ShortlistModelService,
    gate: ShortlistSelectionGate,
    promotion_top_n: int,
    target_column: str,
) -> pd.DataFrame:
    working = predictions.copy()
    working["_selection_gate_threshold"] = rolling_score_quantile_thresholds(
        working,
        score_column="predicted_alpha",
        quantile=gate.quantile,
        lookback_sessions=gate.lookback_sessions,
    )
    unique_dates = sorted(working["snapshot_date"].drop_duplicates().tolist())
    rows = []
    for fold_count, label in ((1, "last_fold"), (3, "trailing_3folds")):
        date_count = max(int(fold_count), 1) * TEST_WINDOW_DATES
        selected_dates = unique_dates[-min(date_count, len(unique_dates)) :]
        rows.append(
            evaluate_gated_window(
                working[working["snapshot_date"].isin(selected_dates)].copy(),
                service=service,
                model_name=f"{AUDIT_MODEL}_{label}",
                top_n=promotion_top_n,
                target_column=target_column,
                gate=gate,
            )
        )
    rows.append(
        evaluate_gated_window(
            working,
            service=service,
            model_name=f"{AUDIT_MODEL}_full_oos",
            top_n=promotion_top_n,
            target_column=target_column,
            gate=gate,
        )
    )
    return pd.DataFrame(rows)


def evaluate_gated_window(
    predictions: pd.DataFrame,
    *,
    service: ShortlistModelService,
    model_name: str,
    top_n: int,
    target_column: str,
    gate: ShortlistSelectionGate,
) -> dict[str, object]:
    if predictions.empty:
        summary = service._empty_summary(model_name)
        if gate.active:
            summary.update({"total_dates": 0, "active_dates": 0, "empty_gated_dates": 0})
        return summary
    cost_fraction = service._round_trip_cost_fraction()
    rows: list[dict[str, object]] = []
    total_dates = int(predictions["snapshot_date"].nunique())
    empty_gated_dates = 0
    pick_tickers: list[str] = []
    for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
        ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
        if gate.active:
            threshold = pd.to_numeric(ordered["_selection_gate_threshold"], errors="coerce").dropna()
            if threshold.empty:
                empty_gated_dates += 1
                continue
            ordered_for_picks = ordered[
                pd.to_numeric(ordered["predicted_alpha"], errors="coerce") >= float(threshold.iloc[0])
            ].copy()
        else:
            ordered_for_picks = ordered
        if ordered_for_picks.empty:
            empty_gated_dates += 1
            continue
        picks = ordered_for_picks.head(int(top_n)).copy()
        target = pd.to_numeric(picks[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        universe_target = pd.to_numeric(day_frame[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
        if target.empty or universe_target.empty:
            continue
        full_target = pd.to_numeric(ordered[target_column], errors="coerce")
        full_score = pd.to_numeric(ordered["predicted_alpha"], errors="coerce")
        full_valid = full_target.notna() & full_score.notna()
        spearman = float("nan")
        if (
            int(full_valid.sum()) >= 3
            and full_target[full_valid].nunique(dropna=True) > 1
            and full_score[full_valid].nunique(dropna=True) > 1
        ):
            corr = full_score[full_valid].corr(full_target[full_valid], method="spearman")
            if pd.notna(corr) and math.isfinite(float(corr)):
                spearman = float(corr)
        net_target = target - cost_fraction
        net_universe = universe_target - cost_fraction
        pick_tickers.extend(picks["ticker"].astype(str).tolist())
        rows.append(
            {
                "date": pd.Timestamp(snapshot_date),
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
        summary = service._empty_summary(model_name)
        if gate.active:
            summary.update({"total_dates": total_dates, "active_dates": 0, "empty_gated_dates": empty_gated_dates})
        return summary
    frame = pd.DataFrame(rows)
    top_ticker = None
    top_ticker_date_rate = float("nan")
    top_ticker_pick_share = float("nan")
    if pick_tickers:
        counts = pd.Series(pick_tickers).value_counts()
        top_ticker = str(counts.index[0])
        top_count = int(counts.iloc[0])
        top_ticker_date_rate = float(top_count / max(int(frame["date"].nunique()), 1))
        top_ticker_pick_share = float(top_count / len(pick_tickers))
    summary = {
        "model": model_name,
        "dates": len(frame.index),
        "avg_pick_count": float(frame["pick_count"].mean()),
        "gross_mean_target": float(pd.to_numeric(frame["gross_mean_target"], errors="coerce").mean()),
        "mean_target": float(frame["mean_target"].mean()),
        "hit_rate": float(frame["hit_rate"].mean()),
        "universe_mean_target": float(frame["universe_mean_target"].mean()),
        "universe_hit_rate": float(frame["universe_hit_rate"].mean()),
        "mean_target_excess": float((frame["mean_target"] - frame["universe_mean_target"]).mean()),
        "hit_rate_excess": float((frame["hit_rate"] - frame["universe_hit_rate"]).mean()),
        "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
        "spearman": float(frame["spearman"].dropna().mean()) if frame["spearman"].notna().any() else float("nan"),
        "top_ticker": top_ticker,
        "top_ticker_date_rate": top_ticker_date_rate,
        "top_ticker_pick_share": top_ticker_pick_share,
    }
    if gate.active:
        summary.update(
            {
                "total_dates": total_dates,
                "active_dates": len(frame.index),
                "empty_gated_dates": empty_gated_dates,
            }
        )
    return summary


def floor_verdict_rows(
    summaries: pd.DataFrame,
    *,
    promotion_gate: dict[str, float],
) -> list[dict[str, object]]:
    checks = [
        ("last_fold", f"{AUDIT_MODEL}_last_fold", "hit_rate_excess", ">=", promotion_gate["min_recent_1fold_hit_rate_excess"]),
        ("last_fold", f"{AUDIT_MODEL}_last_fold", "beat_universe_rate", ">=", promotion_gate["min_recent_1fold_beat_universe_rate"]),
        ("last_fold", f"{AUDIT_MODEL}_last_fold", "mean_target_excess", ">=", promotion_gate["min_recent_1fold_mean_target_excess"]),
        ("last_fold", f"{AUDIT_MODEL}_last_fold", "spearman", ">=", promotion_gate["min_recent_1fold_spearman"]),
        ("last_fold", f"{AUDIT_MODEL}_last_fold", "top_ticker_date_rate", "<=", promotion_gate["max_recent_1fold_top_ticker_date_rate"]),
        ("trailing_3folds", f"{AUDIT_MODEL}_trailing_3folds", "hit_rate_excess", ">=", promotion_gate["min_recent_3fold_hit_rate_excess"]),
        ("trailing_3folds", f"{AUDIT_MODEL}_trailing_3folds", "beat_universe_rate", ">=", promotion_gate["min_recent_3fold_beat_universe_rate"]),
        ("trailing_3folds", f"{AUDIT_MODEL}_trailing_3folds", "mean_target_excess", ">=", promotion_gate["min_recent_3fold_mean_target_excess"]),
        ("trailing_3folds", f"{AUDIT_MODEL}_trailing_3folds", "spearman", ">=", promotion_gate["min_recent_3fold_spearman"]),
        ("trailing_3folds", f"{AUDIT_MODEL}_trailing_3folds", "top_ticker_date_rate", "<=", promotion_gate["max_recent_3fold_top_ticker_date_rate"]),
        ("full_oos", f"{AUDIT_MODEL}_full_oos", "spearman", ">=", promotion_gate["min_full_oos_spearman"]),
    ]
    rows: list[dict[str, object]] = []
    for window, model, metric, op, floor in checks:
        row = summaries[summaries["model"].astype(str).eq(model)]
        value = float("nan") if row.empty or metric not in row.columns else row.iloc[0].get(metric)
        passes = _passes_floor(value, op=op, floor=floor)
        rows.append(
            {
                "window": window,
                "metric": metric,
                "value": value,
                "op": op,
                "floor": floor,
                "passes": passes,
            }
        )
    return rows


def render_report(
    *,
    gate: ShortlistSelectionGate,
    oos: pd.DataFrame,
    summaries: pd.DataFrame,
    floor_rows: list[dict[str, object]],
    passes: bool,
    live: pd.DataFrame,
    gated_live: pd.DataFrame,
    live_threshold: float | None,
) -> str:
    latest_live_date = None if live.empty else pd.Timestamp(live["snapshot_date"].max()).date().isoformat()
    oos_dates = sorted(oos["snapshot_date"].dropna().drop_duplicates().tolist())
    top_after_gate = gated_live.head(PROMOTION_BASKET_SIZE).copy()
    verdict = "PASS: gate-enabled ridge_adaptive clears tonight's promotion floors" if passes else "FAIL: gate-enabled ridge_adaptive does not clear tonight's promotion floors"
    lines = [
        "# Gated Selection Dry Run - 2026-10-07",
        "",
        f"- generated_at: {datetime.now(UTC).replace(microsecond=0).isoformat()}",
        "- data_access: read-only CSV artifacts; no `./sq` write command; no `data/` mutation",
        f"- oos_artifact: {OOS_PATH}",
        f"- live_artifact: {LIVE_PATH}",
        f"- model: {AUDIT_MODEL}",
        f"- target_column: {TARGET_COLUMN}",
        f"- selection_gate: {gate.method}",
        f"- selection_gate_enabled: {gate.enabled}",
        f"- selection_gate_quantile: {gate.quantile:.4f}",
        f"- selection_gate_lookback_sessions: {gate.lookback_sessions}",
        f"- oos_rows_loaded: {len(oos.index)}",
        f"- oos_dates_loaded: {len(oos_dates)}",
        f"- oos_date_min: {_date_or_na(oos_dates[0] if oos_dates else None)}",
        f"- oos_date_max: {_date_or_na(oos_dates[-1] if oos_dates else None)}",
        f"- latest_live_date: {latest_live_date or 'n/a'}",
        f"- latest_live_rows: {len(live.index)}",
        f"- live_gate_threshold: {_fmt(live_threshold, places=6)}",
        f"- gate_qualified_live_count: {len(gated_live.index)}",
        f"- runtime_top_n_after_gate: {PROMOTION_BASKET_SIZE}",
        f"- would_be_champion: {AUDIT_MODEL if passes else 'n/a'}",
        f"- numeric_verdict: {verdict}",
        "",
        "## Gated Acceptance Windows",
        "",
        "| window | dates | active_dates | empty_gated_dates | avg_pick_count | hit_rate_excess | beat_universe_rate | mean_target_excess | spearman | top_ticker | top_ticker_date_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summaries.sort_values("model").itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.dates)} | {int(getattr(row, 'active_dates', row.dates))} | "
            f"{int(getattr(row, 'empty_gated_dates', 0))} | {_fmt(row.avg_pick_count)} | "
            f"{_fmt(row.hit_rate_excess)} | {_fmt(row.beat_universe_rate)} | {_fmt(row.mean_target_excess)} | "
            f"{_fmt(row.spearman)} | {getattr(row, 'top_ticker', None) or 'n/a'} | {_fmt(getattr(row, 'top_ticker_date_rate', float('nan')))} |"
        )
    lines.extend(
        [
            "",
            "## Floor Verdict",
            "",
            "| window | metric | value | floor | pass |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in floor_rows:
        lines.append(
            f"| {row['window']} | {row['metric']} | {_fmt(row['value'])} | "
            f"{row['op']} {_fmt(row['floor'])} | {'yes' if row['passes'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Would-Be Live Picks",
            "",
            f"- note: {len(gated_live.index)} latest-live rows clear the rolling score threshold; the runtime model path remains capped at top-2 picks.",
            "",
            "| rank_after_gate | ticker | sector | score | threshold |",
            "|---:|---|---|---:|---:|",
        ]
    )
    if top_after_gate.empty:
        lines.append("| 0 | n/a | n/a | n/a | n/a |")
    else:
        for rank, row in enumerate(top_after_gate.itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | {row.ticker} | {getattr(row, 'sector', '') or 'n/a'} | "
                f"{_fmt(row.predicted_alpha, places=6)} | {_fmt(live_threshold, places=6)} |"
            )
    lines.extend(
        [
            "",
            "## Gate-Qualified Live Roster",
            "",
            "| rank_after_gate | ticker | sector | score | threshold |",
            "|---:|---|---|---:|---:|",
        ]
    )
    if gated_live.empty:
        lines.append("| 0 | n/a | n/a | n/a | n/a |")
    else:
        for rank, row in enumerate(gated_live.itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | {row.ticker} | {getattr(row, 'sector', '') or 'n/a'} | "
                f"{_fmt(row.predicted_alpha, places=6)} | {_fmt(live_threshold, places=6)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _load_selection_gate(config: dict) -> ShortlistSelectionGate:
    payload = config.get("scan_policy", {}).get("shortlist_model", {}).get("selection_gate", {})
    return ShortlistSelectionGate.from_config(payload)


def _load_promotion_gate(config: dict) -> dict[str, float]:
    payload = config.get("scan_policy", {}).get("shortlist_model", {}).get("promotion_gate", {})
    return {
        "min_recent_1fold_hit_rate_excess": float(payload.get("min_recent_1fold_hit_rate_excess", 0.02)),
        "min_recent_1fold_beat_universe_rate": float(payload.get("min_recent_1fold_beat_universe_rate", 0.50)),
        "min_recent_1fold_mean_target_excess": float(payload.get("min_recent_1fold_mean_target_excess", 0.0)),
        "min_recent_1fold_spearman": float(payload.get("min_recent_1fold_spearman", 0.0)),
        "max_recent_1fold_top_ticker_date_rate": float(payload.get("max_recent_1fold_top_ticker_date_rate", 0.40)),
        "min_recent_3fold_hit_rate_excess": float(payload.get("min_recent_3fold_hit_rate_excess", 0.02)),
        "min_recent_3fold_beat_universe_rate": float(payload.get("min_recent_3fold_beat_universe_rate", 0.50)),
        "min_recent_3fold_mean_target_excess": float(payload.get("min_recent_3fold_mean_target_excess", 0.0)),
        "min_recent_3fold_spearman": float(payload.get("min_recent_3fold_spearman", 0.0)),
        "max_recent_3fold_top_ticker_date_rate": float(payload.get("max_recent_3fold_top_ticker_date_rate", 0.40)),
        "min_full_oos_spearman": float(payload.get("min_full_oos_spearman", 0.0)),
    }


def _passes_floor(value: object, *, op: str, floor: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    if op == "<=":
        return numeric <= float(floor)
    return numeric >= float(floor)


def _date_or_na(value: object) -> str:
    if value is None:
        return "n/a"
    return pd.Timestamp(value).date().isoformat()


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
