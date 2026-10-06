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

from scripts import a4_regime_interactions as a4
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings
from src.utils.db_manager import DatabaseManager


FAST_REGIME_TICKER = "SPY"
FAST_REGIME_SHORT_WINDOW = 5
FAST_REGIME_LONG_WINDOW = 20
CANDIDATE_MODELS = ("ridge_model", "ic_sign_model")


def _read_spy_ohlc(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT date, adj_close
            FROM historical_ohlcv
            WHERE ticker = ?
            ORDER BY date
            """,
            [FAST_REGIME_TICKER],
        ).fetchdf()


def fast_regime_by_snapshot_date(
    prices: pd.DataFrame,
    snapshot_dates: list[pd.Timestamp],
    *,
    short_window: int = FAST_REGIME_SHORT_WINDOW,
    long_window: int = FAST_REGIME_LONG_WINDOW,
) -> dict[pd.Timestamp, str]:
    """Classify each snapshot from the latest available SPY 5d/20d momentum."""
    if prices.empty or not snapshot_dates:
        return {}
    working = prices.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
    working = working.dropna(subset=["date", "adj_close"]).sort_values("date").reset_index(drop=True)
    if working.empty:
        return {}
    working["ret_short"] = working["adj_close"].pct_change(max(int(short_window), 1))
    working["ret_long"] = working["adj_close"].pct_change(max(int(long_window), 1))
    working["classification"] = working.apply(_classify_fast_regime_row, axis=1)

    dates = pd.DataFrame(
        {
            "snapshot_date": sorted(
                pd.to_datetime(pd.Series(snapshot_dates), errors="coerce").dropna().dt.normalize().drop_duplicates().tolist()
            )
        }
    )
    if dates.empty:
        return {}
    mapped = pd.merge_asof(
        dates.sort_values("snapshot_date"),
        working[["date", "classification"]].sort_values("date"),
        left_on="snapshot_date",
        right_on="date",
        direction="backward",
    )
    mapped = mapped.dropna(subset=["snapshot_date", "classification"])
    return {
        pd.Timestamp(row.snapshot_date).normalize(): str(row.classification)
        for row in mapped.itertuples(index=False)
        if str(row.classification) in {"trending", "reversal"}
    }


def _classify_fast_regime_row(row: pd.Series) -> str | None:
    ret_short = row.get("ret_short")
    ret_long = row.get("ret_long")
    if pd.isna(ret_short) or pd.isna(ret_long):
        return None
    return "trending" if float(ret_short) >= 0.0 and float(ret_long) >= 0.0 else "reversal"


def _source_interactions(
    service: ShortlistModelService,
    frame: pd.DataFrame,
    *,
    source_name: str,
    top_features: list[str],
    regime_by_date: dict[pd.Timestamp, str],
    config: dict[str, object],
) -> dict[str, object]:
    interaction_frame, interaction_columns = a4.build_regime_interaction_frame(
        frame,
        top_features=top_features,
        regime_by_date=regime_by_date,
    )
    fold_rows = a4._fold_ic_rows(
        service,
        interaction_frame,
        interaction_columns=interaction_columns,
        config=config,
        min_feature_ic=service._load_min_feature_ic(),
        observation_fraction=service._load_min_feature_ic_observation_fraction(),
    )
    summary = a4._ic_summary(fold_rows)
    survivors = (
        summary[summary["mean_abs_ic"].ge(float(service._load_min_feature_ic()))]["feature"].astype(str).tolist()
        if not summary.empty
        else []
    )
    counts = pd.Series(list(regime_by_date.values()), dtype="object").value_counts().to_dict()
    return {
        "source_name": source_name,
        "frame": interaction_frame,
        "columns": interaction_columns,
        "fold_rows": fold_rows,
        "summary": summary,
        "survivors": survivors,
        "regime_counts": counts,
    }


def _run_source_model(
    service: ShortlistModelService,
    source: dict[str, object],
    *,
    model_name: str,
    baseline_features: list[str],
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    survivors = list(source["survivors"])
    if not survivors:
        return pd.DataFrame()
    feature_columns = service._filter_model_feature_columns([*baseline_features, *survivors])
    dense = a4._run_model(
        service,
        source["frame"],
        model_name=model_name,
        feature_columns=feature_columns,
        config=config,
    )
    return service._non_overlapping_oos_predictions(
        dense,
        horizon_days=int(config["label_horizon_dates"]),
        calendar_dates=calendar_dates,
    )


def _bakeoff(
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    *,
    baseline_features: list[str],
    meter_source: dict[str, object],
    fast_source: dict[str, object],
    config: dict[str, object],
    calendar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    extra_features = list(dict.fromkeys([*meter_source["columns"], *fast_source["columns"]]))
    with a4._temporary_model_features(extra_features):
        for model_name in CANDIDATE_MODELS:
            dense_without = a4._run_model(
                service,
                eligible,
                model_name=model_name,
                feature_columns=baseline_features,
                config=config,
            )
            honest_without = service._non_overlapping_oos_predictions(
                dense_without,
                horizon_days=int(config["label_horizon_dates"]),
                calendar_dates=calendar_dates,
            )
            meter = _run_source_model(
                service,
                meter_source,
                model_name=model_name,
                baseline_features=baseline_features,
                config=config,
                calendar_dates=calendar_dates,
            )
            fast = _run_source_model(
                service,
                fast_source,
                model_name=model_name,
                baseline_features=baseline_features,
                config=config,
                calendar_dates=calendar_dates,
            )
            without_s = a4._pooled_spearman(honest_without)
            meter_s = a4._pooled_spearman(meter)
            fast_s = a4._pooled_spearman(fast)
            rows.append(
                {
                    "model": model_name,
                    "without_rows": len(honest_without.index),
                    "meter_rows": len(meter.index),
                    "fast_rows": len(fast.index),
                    "without_spearman": without_s,
                    "meter_spearman": meter_s,
                    "fast_spearman": fast_s,
                    "meter_delta_vs_without": _delta(meter_s, without_s),
                    "fast_delta_vs_without": _delta(fast_s, without_s),
                    "fast_delta_vs_meter": _delta(fast_s, meter_s),
                }
            )
    return pd.DataFrame(rows)


def _delta(left: float, right: float) -> float:
    return float(left) - float(right) if math.isfinite(float(left)) and math.isfinite(float(right)) else float("nan")


def _verdict(bakeoff: pd.DataFrame) -> str:
    deltas = pd.to_numeric(bakeoff.get("fast_delta_vs_meter"), errors="coerce").dropna()
    if deltas.empty:
        return "loses: fast regime produced no comparable honest-grid rows"
    if float(deltas.max()) >= 0.005:
        return "fast regime wins: >= +0.005 Spearman on at least one model"
    if float(deltas.min()) <= -0.005:
        return "fast regime loses: meter source remains better by >= 0.005 on at least one model"
    return "fast regime ties: all model deltas versus meter are inside +/-0.005 Spearman"


def _render_report(
    *,
    output_path: Path,
    top_features: list[str],
    meter_source: dict[str, object],
    fast_source: dict[str, object],
    bakeoff: pd.DataFrame,
) -> None:
    verdict = _verdict(bakeoff)
    lines = [
        "# P2 Fast-Regime A4 Interaction Audit",
        "",
        "- idea: replace A4's stale lagged regime-meter interaction source with a same-grid OHLC fast regime, then compare honestly",
        "- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation",
        f"- fast_regime: `{FAST_REGIME_TICKER}` {FAST_REGIME_SHORT_WINDOW}d and {FAST_REGIME_LONG_WINDOW}d adjusted-close returns; trending only when both are non-negative, otherwise reversal",
        "- production_wiring: unchanged unless fast regime wins this audit",
        f"- top_features: {', '.join(top_features) if top_features else 'none'}",
        f"- meter_surviving_interactions: {len(meter_source['survivors'])}",
        f"- fast_surviving_interactions: {len(fast_source['survivors'])}",
        f"- meter_regime_counts: {meter_source['regime_counts']}",
        f"- fast_regime_counts: {fast_source['regime_counts']}",
        f"- verdict: {verdict}",
        "",
        "## Honest-Grid Three-Way Bakeoff",
        "",
        "| model | without_rows | meter_rows | fast_rows | without_spearman | with_a4_meter | with_a4_fast | meter_delta_vs_without | fast_delta_vs_without | fast_delta_vs_meter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bakeoff.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.without_rows)} | {int(row.meter_rows)} | {int(row.fast_rows)} | "
            f"{a4._fmt(row.without_spearman)} | {a4._fmt(row.meter_spearman)} | {a4._fmt(row.fast_spearman)} | "
            f"{a4._fmt(row.meter_delta_vs_without)} | {a4._fmt(row.fast_delta_vs_without)} | {a4._fmt(row.fast_delta_vs_meter)} |"
        )
    lines.extend(["", "## Interaction IC Survivors", ""])
    for source in (meter_source, fast_source):
        lines.extend(
            [
                f"### {source['source_name']}",
                "",
                "| feature | folds | mean_ic | mean_abs_ic | max_abs_ic | clear_folds | clear_rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        summary = source["summary"]
        for row in summary.itertuples(index=False):
            if row.feature not in set(source["survivors"]):
                continue
            lines.append(
                f"| {row.feature} | {int(row.folds)} | {a4._fmt(row.mean_ic)} | {a4._fmt(row.mean_abs_ic)} | "
                f"{a4._fmt(row.max_abs_ic)} | {int(row.clear_folds)} | {a4._fmt(row.clear_rate)} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = a4._shortlist_config()
    available = a4._read_table_columns(duckdb_path=settings.paths.duckdb_path, table="universe_daily_snapshots")
    snapshots = a4._read_snapshots(duckdb_path=settings.paths.duckdb_path, columns=a4._snapshot_columns(available))
    prepared = service._prepare_snapshot_frame(snapshots)
    eligible = service._build_matured_eligible_universe(
        prepared,
        target_column=a4.TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    calendar_dates = a4._read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    prediction_dates = sorted(eligible["snapshot_date"].drop_duplicates().tolist())
    top_features = a4._parse_top_feature_ic_features(
        settings.paths.reports_dir / "feature_ic_report.md",
        limit=a4.TOP_FEATURE_COUNT,
    )
    regime_meter = a4._read_regime_meter(duckdb_path=settings.paths.duckdb_path)
    meter_by_date = a4._regime_classifications_by_prediction_date(
        prediction_dates=prediction_dates,
        universe_dates=calendar_dates,
        regime_meter=regime_meter,
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    fast_by_date = fast_regime_by_snapshot_date(
        _read_spy_ohlc(duckdb_path=settings.paths.duckdb_path),
        prediction_dates,
    )
    meter_source = _source_interactions(
        service,
        eligible,
        source_name="with_a4_meter",
        top_features=top_features,
        regime_by_date=meter_by_date,
        config=config,
    )
    fast_source = _source_interactions(
        service,
        eligible,
        source_name="with_a4_fast",
        top_features=top_features,
        regime_by_date=fast_by_date,
        config=config,
    )
    bakeoff = _bakeoff(
        service,
        eligible,
        baseline_features=top_features,
        meter_source=meter_source,
        fast_source=fast_source,
        config=config,
        calendar_dates=calendar_dates,
    )
    _render_report(
        output_path=output_path,
        top_features=top_features,
        meter_source=meter_source,
        fast_source=fast_source,
        bakeoff=bakeoff,
    )
    return {
        "output_path": str(output_path),
        "verdict": _verdict(bakeoff),
        "meter_survivors": len(meter_source["survivors"]),
        "fast_survivors": len(fast_source["survivors"]),
        "best_fast_delta_vs_meter": float(pd.to_numeric(bakeoff["fast_delta_vs_meter"], errors="coerce").max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P2 fast-regime source for A4 interactions.")
    parser.add_argument("--output", type=Path, default=Path("reports/p2_fast_regime.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} verdict='{verdict}' meter_survivors={meter_survivors} "
        "fast_survivors={fast_survivors} best_fast_delta_vs_meter={best_fast_delta_vs_meter:+.4f}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
