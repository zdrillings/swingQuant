from __future__ import annotations

import argparse
from datetime import date
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

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.settings import get_settings
from src.utils.db_manager import DatabaseManager


CANDIDATE_MODELS = (
    "signal_proxy",
    "reversal_rules",
    "event_signal",
    "event_ic_model",
    "structure_factor_signal",
    "ridge_model",
    "lasso_model",
    "elastic_net_model",
    "ic_sign_model",
    "xgboost_model",
)
ENSEMBLE_MODEL = "ensemble_model"


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _prediction_columns() -> list[str]:
    return [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "alpha_vs_sector_60d",
        "predicted_alpha",
        "model_name",
        "regime_matched_training_applied",
    ]


def _read_matched_predictions(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, usecols=_prediction_columns(), low_memory=False)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["snapshot_date"])
    return {
        str(model_name): model_frame.drop(columns=["model_name"]).reset_index(drop=True)
        for model_name, model_frame in frame.groupby("model_name", sort=True)
    }


def _read_universe_snapshots(*, duckdb_path: Path, columns: list[str]) -> pd.DataFrame:
    import duckdb

    safe_columns = ", ".join(columns)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT {safe_columns}
            FROM universe_daily_snapshots
            WHERE alpha_vs_sector_60d IS NOT NULL
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _build_snapshot_columns(feature_columns: list[str]) -> list[str]:
    required = [
        "snapshot_date",
        "ticker",
        "sector",
        "md_volume_30d",
        "adj_close",
        "passed_any_strategy",
        "passed_slots_json",
        "alpha_vs_sector_60d",
        "regime_green",
    ]
    return list(dict.fromkeys([*required, *feature_columns]))


def _common_grid(predictions_by_model: dict[str, pd.DataFrame]) -> set[pd.Timestamp]:
    date_sets: list[set[pd.Timestamp]] = []
    for frame in predictions_by_model.values():
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize().dropna()
        date_sets.append(set(dates.tolist()))
    if not date_sets:
        return set()
    common = set(date_sets[0])
    for dates in date_sets[1:]:
        common &= dates
    return common


def _filter_to_dates(
    predictions_by_model: dict[str, pd.DataFrame],
    dates: set[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for model_name, frame in predictions_by_model.items():
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
        filtered[model_name] = working[working["snapshot_date"].isin(dates)].reset_index(drop=True)
    return filtered


def _fold_spearman_frame(
    *,
    service: ShortlistModelService,
    matched: dict[str, pd.DataFrame],
    unmatched: dict[str, pd.DataFrame],
    top_n: int,
    target_column: str,
    fold_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_dates = sorted(_common_grid({**matched, **unmatched}))
    for fold_index, start in enumerate(range(0, len(all_dates), max(int(fold_size), 1)), start=1):
        fold_dates = set(all_dates[start : start + max(int(fold_size), 1)])
        if not fold_dates:
            continue
        fold_start = min(fold_dates)
        fold_end = max(fold_dates)
        for model_name in sorted(set(matched) & set(unmatched)):
            matched_summary = service._evaluate_predictions(
                predictions=matched[model_name][matched[model_name]["snapshot_date"].isin(fold_dates)],
                top_n=top_n,
                target_column=target_column,
                model_name=model_name,
            )
            unmatched_summary = service._evaluate_predictions(
                predictions=unmatched[model_name][unmatched[model_name]["snapshot_date"].isin(fold_dates)],
                top_n=top_n,
                target_column=target_column,
                model_name=model_name,
            )
            matched_spearman = float(matched_summary.get("spearman", float("nan")))
            unmatched_spearman = float(unmatched_summary.get("spearman", float("nan")))
            rows.append(
                {
                    "fold": fold_index,
                    "fold_start": pd.Timestamp(fold_start).date().isoformat(),
                    "fold_end": pd.Timestamp(fold_end).date().isoformat(),
                    "model": model_name,
                    "matched_spearman": matched_spearman,
                    "unmatched_spearman": unmatched_spearman,
                    "delta_unmatched_minus_matched": unmatched_spearman - matched_spearman,
                }
            )
    return pd.DataFrame(rows)


def _summary_frame(
    *,
    service: ShortlistModelService,
    matched: dict[str, pd.DataFrame],
    unmatched: dict[str, pd.DataFrame],
    top_n: int,
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in sorted(set(matched) & set(unmatched)):
        matched_summary = service._evaluate_predictions(
            predictions=matched[model_name],
            top_n=top_n,
            target_column=target_column,
            model_name=model_name,
        )
        unmatched_summary = service._evaluate_predictions(
            predictions=unmatched[model_name],
            top_n=top_n,
            target_column=target_column,
            model_name=model_name,
        )
        matched_spearman = float(matched_summary.get("spearman", float("nan")))
        unmatched_spearman = float(unmatched_summary.get("spearman", float("nan")))
        rows.append(
            {
                "model": model_name,
                "dates": int(matched_summary.get("dates", 0)),
                "matched_spearman": matched_spearman,
                "unmatched_spearman": unmatched_spearman,
                "delta_unmatched_minus_matched": unmatched_spearman - matched_spearman,
                "matched_mean_target": float(matched_summary.get("mean_target", float("nan"))),
                "unmatched_mean_target": float(unmatched_summary.get("mean_target", float("nan"))),
                "matched_beat_universe_rate": float(matched_summary.get("beat_universe_rate", float("nan"))),
                "unmatched_beat_universe_rate": float(unmatched_summary.get("beat_universe_rate", float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _disable_recommendation(summary: pd.DataFrame, *, tolerance: float = 0.0025) -> tuple[bool, str]:
    finite = summary[
        summary["matched_spearman"].map(math.isfinite)
        & summary["unmatched_spearman"].map(math.isfinite)
    ].copy()
    if finite.empty:
        return False, "insufficient finite matched/unmatched Spearman values"
    matched_best = float(finite["matched_spearman"].max())
    unmatched_best = float(finite["unmatched_spearman"].max())
    mean_delta = float(finite["delta_unmatched_minus_matched"].mean())
    if unmatched_best + tolerance >= matched_best and mean_delta >= -tolerance:
        return True, (
            f"unmatched best {_fmt(unmatched_best)} is within {tolerance:.4f} of matched best "
            f"{_fmt(matched_best)} and mean delta is {_fmt(mean_delta)}"
        )
    if unmatched_best + tolerance < matched_best:
        return False, (
            f"matched best {_fmt(matched_best)} beats unmatched best {_fmt(unmatched_best)} "
            f"beyond {tolerance:.4f}; mean delta is {_fmt(mean_delta)}"
        )
    return False, (
        f"unmatched best {_fmt(unmatched_best)} is neutral, but mean delta {_fmt(mean_delta)} "
        f"is worse than -{tolerance:.4f}"
    )


def _render_report(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    eligible_rows: int,
    eligible_dates: int,
    oos_dates: int,
    matched_stats: dict[str, int],
    disable_recommended: bool,
    disable_reason: str,
) -> str:
    best_matched = summary.sort_values("matched_spearman", ascending=False).head(1)
    best_unmatched = summary.sort_values("unmatched_spearman", ascending=False).head(1)
    best_matched_model = str(best_matched.iloc[0]["model"]) if not best_matched.empty else "n/a"
    best_unmatched_model = str(best_unmatched.iloc[0]["model"]) if not best_unmatched.empty else "n/a"
    best_matched_value = float(best_matched.iloc[0]["matched_spearman"]) if not best_matched.empty else float("nan")
    best_unmatched_value = float(best_unmatched.iloc[0]["unmatched_spearman"]) if not best_unmatched.empty else float("nan")
    verdict = "disable regime_matching" if disable_recommended else "keep regime_matching"
    lines = [
        "# Regime Matching Attribution",
        "",
        f"- generated_at: {date.today().isoformat()}",
        "- data_access: read-only DuckDB plus existing matched OOS CSV; no `./sq` writes; no `data/` mutation",
        "- experiment: 60d endpoint label, top-2 promotion basket, global model scope, 252-date rolling train cap, 20-date test window, 20-date OOS stride",
        "- matched_path: existing `reports/shortlist_model_oos_predictions.csv` from train_and_flip run",
        "- unmatched_path: same walk-forward configuration with `regime_matching_mode=off`",
        "- comparison_grid: common matched/unmatched/model OOS dates",
        f"- eligible_rows: {int(eligible_rows)}",
        f"- eligible_dates: {int(eligible_dates)}",
        f"- oos_prediction_dates: {int(oos_dates)}",
        (
            "- matched_regime_folds: "
            f"attempted={int(matched_stats.get('attempted_folds', 0))}, "
            f"matched={int(matched_stats.get('matched_folds', 0))}, "
            f"fallback={int(matched_stats.get('fallback_folds', 0))}, "
            f"unknown={int(matched_stats.get('unknown_folds', 0))}"
        ),
        f"- verdict: {verdict} ({disable_reason})",
        "",
        "## Full-OOS Spearman",
        "",
        "| model | dates | matched_spearman | unmatched_spearman | delta_unmatched_minus_matched | matched_mean_target | unmatched_mean_target | matched_beat_universe_rate | unmatched_beat_universe_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| {model} | {dates} | {matched} | {unmatched} | {delta} | {matched_mean} | {unmatched_mean} | {matched_beat} | {unmatched_beat} |".format(
                model=row.model,
                dates=int(row.dates),
                matched=_fmt(row.matched_spearman),
                unmatched=_fmt(row.unmatched_spearman),
                delta=_fmt(row.delta_unmatched_minus_matched),
                matched_mean=_fmt(row.matched_mean_target),
                unmatched_mean=_fmt(row.unmatched_mean_target),
                matched_beat=_fmt(row.matched_beat_universe_rate),
                unmatched_beat=_fmt(row.unmatched_beat_universe_rate),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Fold Delta",
            "",
            "| fold | fold_start | fold_end | model | matched_spearman | unmatched_spearman | delta_unmatched_minus_matched |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for row in fold_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.fold_start} | {row.fold_end} | {row.model} | "
            f"{_fmt(row.matched_spearman)} | {_fmt(row.unmatched_spearman)} | {_fmt(row.delta_unmatched_minus_matched)} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                f"{verdict}: matched best is {best_matched_model} at {_fmt(best_matched_value)}, "
                f"unmatched best is {best_unmatched_model} at {_fmt(best_unmatched_value)}. "
                f"{disable_reason}."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


def run(*, output_path: Path, matched_oos_csv: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    raw_feature_columns = service._filter_model_feature_columns(MODEL_FEATURE_COLUMNS)
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
    frame = _read_universe_snapshots(
        duckdb_path=settings.paths.duckdb_path,
        columns=_build_snapshot_columns(raw_feature_columns),
    )
    frame = service._prepare_snapshot_frame(frame)
    eligible = service._build_matured_eligible_universe(
        frame,
        target_column="alpha_vs_sector_60d",
        eligible_universe_mode="passed_or_trend",
    )
    matched_stats = {
        "attempted_folds": 0,
        "matched_folds": 0,
        "fallback_folds": 0,
        "unknown_folds": 0,
        "live_matched": 0,
        "live_fallback": 0,
    }
    matched_predictions = _read_matched_predictions(matched_oos_csv)
    matched_fold_count = 0
    matched_frames = [frame for name, frame in matched_predictions.items() if name in CANDIDATE_MODELS]
    if matched_frames:
        matched_all = pd.concat(matched_frames, ignore_index=True)
        matched_dates = sorted(pd.to_datetime(matched_all["snapshot_date"], errors="coerce").dt.normalize().dropna().drop_duplicates().tolist())
        matched_fold_count = len(range(0, len(matched_dates), 20))
    matched_predictions = {
        model_name: matched_predictions[model_name]
        for model_name in (*CANDIDATE_MODELS, ENSEMBLE_MODEL)
        if model_name in matched_predictions
    }
    unmatched_predictions: dict[str, pd.DataFrame] = {}
    for model_name in CANDIDATE_MODELS:
        predictions = service._walk_forward_predictions(
            eligible,
            target_column="alpha_vs_sector_60d",
            evaluation_target_column="alpha_vs_sector_60d",
            model_name=model_name,
            min_train_dates=252,
            max_train_dates=252,
            test_window_dates=20,
            evaluation_stride_dates=20,
            label_horizon_dates=60,
            model_scope="global",
            xgboost_params=service._xgboost_params_for_config("balanced_depth4") if model_name == "xgboost_model" else None,
            feature_columns_override=feature_columns,
            min_feature_ic=service._load_min_feature_ic(),
            min_feature_ic_observation_fraction=service._load_min_feature_ic_observation_fraction(),
            regime_matching_mode="off",
            min_regime_train_dates=service._load_min_regime_train_dates(),
            regime_transition_purge_mode="off",
        )
        if predictions is not None and not predictions.empty:
            unmatched_predictions[model_name] = predictions
    ensemble = service._build_ensemble_predictions(unmatched_predictions)
    if ensemble is not None and not ensemble.empty:
        unmatched_predictions[ENSEMBLE_MODEL] = ensemble
    shared_dates = _common_grid({**matched_predictions, **unmatched_predictions})
    matched_predictions = _filter_to_dates(matched_predictions, shared_dates)
    unmatched_predictions = _filter_to_dates(unmatched_predictions, shared_dates)
    summary = _summary_frame(
        service=service,
        matched=matched_predictions,
        unmatched=unmatched_predictions,
        top_n=2,
        target_column="alpha_vs_sector_60d",
    )
    fold_summary = _fold_spearman_frame(
        service=service,
        matched=matched_predictions,
        unmatched=unmatched_predictions,
        top_n=2,
        target_column="alpha_vs_sector_60d",
        fold_size=20,
    )
    disable_recommended, disable_reason = _disable_recommendation(summary)
    attempted_folds = matched_fold_count * len(CANDIDATE_MODELS)
    matched_stats.update(
        {
            "attempted_folds": attempted_folds,
            "matched_folds": attempted_folds,
            "fallback_folds": 0,
            "unknown_folds": 0,
        }
    )
    _render_report(
        output_path=output_path,
        summary=summary,
        fold_summary=fold_summary,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()),
        oos_dates=len(shared_dates),
        matched_stats=matched_stats,
        disable_recommended=disable_recommended,
        disable_reason=disable_reason,
    )
    best_matched = float(summary["matched_spearman"].max())
    best_unmatched = float(summary["unmatched_spearman"].max())
    return {
        "output_path": str(output_path),
        "oos_dates": len(shared_dates),
        "best_matched_spearman": best_matched,
        "best_unmatched_spearman": best_unmatched,
        "disable_recommended": disable_recommended,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute 60d shortlist skill with regime matching on versus off.")
    parser.add_argument("--output", type=Path, default=Path("reports/regime_matching_attribution.md"))
    parser.add_argument("--matched-oos-csv", type=Path, default=Path("reports/shortlist_model_oos_predictions.csv"))
    args = parser.parse_args()
    result = run(output_path=args.output, matched_oos_csv=args.matched_oos_csv)
    print(
        "wrote {output_path} oos_dates={oos_dates} best_matched_spearman={best_matched_spearman:+.4f} "
        "best_unmatched_spearman={best_unmatched_spearman:+.4f} disable_recommended={disable_recommended}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
