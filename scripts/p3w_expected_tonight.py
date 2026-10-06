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

from src.research.shortlist_bakeoff_service import (
    expand_model_feature_columns,
    model_feature_columns_for_profile,
    normalize_shortlist_feature_profile,
)
from src.research.shortlist_model_service import PROMOTION_BASKET_SIZE, ShortlistModelService
from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.settings import get_settings, load_feature_config
from src.utils.db_manager import DatabaseManager


TARGET_COLUMN = "alpha_vs_sector_60d"
AUDIT_MODEL = "ridge_adaptive"


@dataclass(frozen=True)
class GateCheck:
    window: str
    metric: str
    value: float
    threshold: float
    comparator: str
    passed: bool


def _fmt(value: object, *, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+.{places}f}"


def _shortlist_config() -> dict[str, object]:
    payload = load_feature_config().get("scan_policy", {}).get("shortlist_model", {})
    horizon = int(payload.get("horizon_days", 60))
    return {
        "eligible_universe_mode": normalize_eligible_universe_mode(
            str(payload.get("production_eligible_universe_mode", payload.get("eligible_universe_mode", "passed_only")))
        ),
        "label_horizon_dates": horizon,
        "min_train_dates": int(payload.get("min_train_dates", 252)),
        "max_train_dates": int(payload.get("max_train_dates", payload.get("min_train_dates", 252))),
        "test_window_dates": int(payload.get("test_window_dates", 20)),
        "evaluation_stride_dates": int(payload.get("oos_evaluation_stride_dates", horizon)),
        "model_scope": normalize_model_scope(str(payload.get("production_model_scope", "global"))),
        "xgboost_config": str(payload.get("production_xgboost_config", "balanced_depth4")),
        "feature_profile": normalize_shortlist_feature_profile(str(payload.get("production_feature_profile", "full"))),
        "regime_matching_mode": str(payload.get("regime_matching", "off")),
    }


def _read_snapshots(*, duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        return connection.execute(
            f"""
            SELECT *
            FROM universe_daily_snapshots
            WHERE {TARGET_COLUMN} IS NOT NULL
            ORDER BY snapshot_date, ticker
            """
        ).fetchdf()


def _read_calendar_dates(*, duckdb_path: Path) -> list[pd.Timestamp]:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT DISTINCT snapshot_date FROM universe_daily_snapshots ORDER BY snapshot_date ASC"
        ).fetchall()
    return sorted(pd.to_datetime(pd.Series([row[0] for row in rows]), errors="coerce").dropna().dt.normalize().tolist())


def gate_checks_for_model(
    *,
    service: ShortlistModelService,
    acceptance_summaries: pd.DataFrame,
    promotion_gate: dict[str, float | int],
    model_name: str,
    required_fold_windows: tuple[int, ...],
) -> list[GateCheck]:
    checks: list[GateCheck] = []
    for folds in required_fold_windows:
        row_name = service._fold_window_model_name(model_name=model_name, fold_count=int(folds))
        row = acceptance_summaries[acceptance_summaries["model"].astype(str).eq(row_name)]
        summary = row.iloc[0] if not row.empty else {}
        label = service._fold_window_label(int(folds))
        for metric, key in (
            ("hit_rate_excess", f"min_recent_{folds}fold_hit_rate_excess"),
            ("beat_universe_rate", f"min_recent_{folds}fold_beat_universe_rate"),
            ("mean_target_excess", f"min_recent_{folds}fold_mean_target_excess"),
            ("spearman", f"min_recent_{folds}fold_spearman"),
        ):
            value = float(summary.get(metric, float("nan"))) if hasattr(summary, "get") else float("nan")
            threshold = float(promotion_gate.get(key, 0.0))
            checks.append(
                GateCheck(
                    window=label,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    comparator=">=",
                    passed=math.isfinite(value) and value >= threshold,
                )
            )
        metric = "top_ticker_date_rate"
        key = f"max_recent_{folds}fold_top_ticker_date_rate"
        value = float(summary.get(metric, float("nan"))) if hasattr(summary, "get") else float("nan")
        threshold = float(promotion_gate.get(key, 0.40))
        checks.append(
            GateCheck(
                window=label,
                metric=metric,
                value=value,
                threshold=threshold,
                comparator="<=",
                passed=math.isfinite(value) and value <= threshold,
            )
        )
    full_row = acceptance_summaries[acceptance_summaries["model"].astype(str).eq(f"{model_name}_full_oos")]
    full_summary = full_row.iloc[0] if not full_row.empty else {}
    value = float(full_summary.get("spearman", float("nan"))) if hasattr(full_summary, "get") else float("nan")
    threshold = float(promotion_gate.get("min_full_oos_spearman", 0.0))
    checks.append(
        GateCheck(
            window="full_oos",
            metric="spearman",
            value=value,
            threshold=threshold,
            comparator=">=",
            passed=math.isfinite(value) and value >= threshold,
        )
    )
    return checks


def gate_verdict(checks: list[GateCheck]) -> tuple[bool, list[str]]:
    failures = [f"{check.window} {check.metric}" for check in checks if not check.passed]
    return not failures, failures


def _production_predictions(
    *,
    service: ShortlistModelService,
    eligible: pd.DataFrame,
    config: dict[str, object],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    xgboost_params = service._xgboost_params_for_config(str(config["xgboost_config"]))
    candidate_models = service._candidate_model_roster()
    min_feature_ic = service._load_min_feature_ic()
    min_feature_ic_observation_fraction = service._load_min_feature_ic_observation_fraction()
    min_regime_train_dates = service._load_min_regime_train_dates()
    regime_transition_purge_mode = service._load_regime_transition_purge_mode()
    regime_matching_stats = {
        "attempted_folds": 0,
        "matched_folds": 0,
        "fallback_folds": 0,
        "unknown_folds": 0,
        "live_matched": 0,
        "live_fallback": 0,
    }
    regime_feature_stats: dict[str, dict[str, object]] = {}
    model_predictions: dict[str, pd.DataFrame] = {}
    for model_name in candidate_models:
        model_feature_columns = service._feature_columns_for_candidate(
            model_name=model_name,
            default_feature_columns=feature_columns,
        )
        predictions = service._walk_forward_predictions(
            eligible,
            target_column=TARGET_COLUMN,
            evaluation_target_column=TARGET_COLUMN,
            model_name=model_name,
            min_train_dates=int(config["min_train_dates"]),
            max_train_dates=service._candidate_max_train_dates(
                model_name=model_name,
                default_max_train_dates=int(config["max_train_dates"]),
            ),
            test_window_dates=int(config["test_window_dates"]),
            evaluation_stride_dates=int(config["evaluation_stride_dates"]),
            label_horizon_dates=int(config["label_horizon_dates"]),
            model_scope=str(config["model_scope"]),
            xgboost_params=xgboost_params if model_name == "xgboost_model" else None,
            feature_columns_override=model_feature_columns,
            min_feature_ic=min_feature_ic,
            min_feature_ic_observation_fraction=min_feature_ic_observation_fraction,
            regime_matching_mode=str(config["regime_matching_mode"]),
            min_regime_train_dates=min_regime_train_dates,
            regime_transition_purge_mode=regime_transition_purge_mode,
            regime_matching_stats=regime_matching_stats,
            regime_feature_stats=regime_feature_stats,
        )
        if predictions is None or predictions.empty:
            continue
        if str(config["regime_matching_mode"]) == "train_and_flip":
            predictions = service._apply_regime_conditional_score_flip(
                predictions,
                horizon_sessions=int(config["label_horizon_dates"]),
                fallback_only=True,
            )
        model_predictions[model_name] = predictions
    model_predictions = service._align_model_predictions_to_common_oos_grid(model_predictions)
    ensemble_predictions = service._build_ensemble_predictions(
        service._legacy_grid_model_predictions(model_predictions)
    )
    if ensemble_predictions is not None and not ensemble_predictions.empty:
        model_predictions["ensemble_model"] = ensemble_predictions
    return model_predictions


def _acceptance_summaries(
    *,
    service: ShortlistModelService,
    predictions: dict[str, pd.DataFrame],
    calendar_dates: list[pd.Timestamp],
    config: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    evaluation_predictions = {
        model_name: service._non_overlapping_oos_predictions(
            frame,
            horizon_days=int(config["label_horizon_dates"]),
            calendar_dates=calendar_dates,
        )
        for model_name, frame in predictions.items()
    }
    fold_windows = service._promotion_fold_windows(horizon_days=int(config["label_horizon_dates"]))
    rows = [
        row
        for model_name, frame in evaluation_predictions.items()
        for row in service._rolling_window_summaries(
            predictions=frame,
            target_column=TARGET_COLUMN,
            model_name=model_name,
            top_n=PROMOTION_BASKET_SIZE,
            windows=(),
            fold_windows=fold_windows,
            fold_size=int(config["test_window_dates"]),
            include_full_oos=True,
        ).to_dict(orient="records")
    ]
    return evaluation_predictions, pd.DataFrame(rows)


def _summary_values(acceptance_summaries: pd.DataFrame, row_name: str) -> dict[str, object]:
    row = acceptance_summaries[acceptance_summaries["model"].astype(str).eq(row_name)]
    return row.iloc[0].to_dict() if not row.empty else {}


def _render_report(
    *,
    output_path: Path,
    service: ShortlistModelService,
    acceptance_summaries: pd.DataFrame,
    evaluation_predictions: dict[str, pd.DataFrame],
    config: dict[str, object],
    checks: list[GateCheck],
    eligible_rows: int,
    eligible_dates: int,
) -> None:
    passed, failures = gate_verdict(checks)
    fold_windows = service._promotion_fold_windows(horizon_days=int(config["label_horizon_dates"]))
    lines = [
        "# P3-W Expected Tonight",
        "",
        "- idea: P3-LIVE read-only expected promotion verdict for ridge_adaptive on the production grid",
        "- data_access: DuckDB read_only=True; no `./sq` writes; no data/ mutation",
        f"- target_column: {TARGET_COLUMN}",
        f"- audit_model: {AUDIT_MODEL}",
        f"- model_scope: {config['model_scope']}",
        f"- eligible_universe_mode: {config['eligible_universe_mode']}",
        f"- xgboost_config: {config['xgboost_config']}",
        f"- feature_profile: {config['feature_profile']}",
        f"- regime_matching: {config['regime_matching_mode']}",
        f"- min_train_dates: {int(config['min_train_dates'])}",
        f"- default_max_train_dates: {int(config['max_train_dates'])}",
        "- ridge_adaptive_max_train_dates: 126",
        f"- test_window_dates: {int(config['test_window_dates'])}",
        f"- evaluation_stride_dates: {int(config['evaluation_stride_dates'])}",
        f"- label_horizon_dates: {int(config['label_horizon_dates'])}",
        f"- promotion_basket_size: {PROMOTION_BASKET_SIZE}",
        f"- eligible_rows: {eligible_rows}",
        f"- eligible_dates: {eligible_dates}",
        f"- verdict: ridge_adaptive {'passes on paper' if passed else 'fails on paper'}"
        + ("" if passed else f" ({', '.join(failures)})"),
        "",
        "## Acceptance Windows",
        "",
        "| model | rows | dates | full_oos_spearman | last_fold_spearman | trailing_3fold_spearman | hit_excess_last | beat_last | mean_excess_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_models = [AUDIT_MODEL] + [
        model for model in sorted(evaluation_predictions) if model != AUDIT_MODEL
    ]
    for model_name in ordered_models:
        frame = evaluation_predictions.get(model_name, pd.DataFrame())
        full = _summary_values(acceptance_summaries, f"{model_name}_full_oos")
        last = _summary_values(
            acceptance_summaries,
            service._fold_window_model_name(model_name=model_name, fold_count=1),
        )
        trailing = _summary_values(
            acceptance_summaries,
            service._fold_window_model_name(model_name=model_name, fold_count=3),
        )
        lines.append(
            f"| {model_name} | {len(frame.index)} | {int(frame['snapshot_date'].nunique()) if not frame.empty else 0} | "
            f"{_fmt(full.get('spearman'))} | {_fmt(last.get('spearman'))} | {_fmt(trailing.get('spearman'))} | "
            f"{_fmt(last.get('hit_rate_excess'))} | {_fmt(last.get('beat_universe_rate'))} | {_fmt(last.get('mean_target_excess'))} |"
        )
    lines.extend(
        [
            "",
            "## Ridge Adaptive Gate Floors",
            "",
            "| window | metric | value | floor | result |",
            "|---|---|---:|---:|---|",
        ]
    )
    for check in checks:
        lines.append(
            f"| {check.window} | {check.metric} | {_fmt(check.value)} | "
            f"{check.comparator} {_fmt(check.threshold)} | {'PASS' if check.passed else 'FAIL'} |"
        )
    active_folds = ", ".join(service._fold_window_label(int(folds)) for folds in fold_windows)
    lines.extend(
        [
            "",
            "## Verification Notes",
            "",
            f"- active_fold_windows: {active_folds}",
            "- active_recent_windows: none",
            "- active_full_window: full_oos",
            "- critic_verify_tomorrow: compare this table to tonight's `reports/shortlist_model.md` acceptance windows; floor verdict should match within measurement noise.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, output_path: Path) -> dict[str, object]:
    settings = get_settings()
    service = ShortlistModelService(DatabaseManager(paths=settings.paths))
    config = _shortlist_config()
    snapshots = _read_snapshots(duckdb_path=settings.paths.duckdb_path)
    prepared = service._prepare_snapshot_frame(snapshots)
    prepared = service._add_a4_regime_interaction_features(
        prepared,
        horizon_sessions=int(config["label_horizon_dates"]),
    )
    eligible = service._build_matured_eligible_universe(
        prepared,
        target_column=TARGET_COLUMN,
        eligible_universe_mode=str(config["eligible_universe_mode"]),
    )
    base_features = model_feature_columns_for_profile(str(config["feature_profile"]))
    feature_columns = service._filter_model_feature_columns(expand_model_feature_columns(base_features))
    predictions = _production_predictions(
        service=service,
        eligible=eligible,
        config=config,
        feature_columns=feature_columns,
    )
    calendar_dates = _read_calendar_dates(duckdb_path=settings.paths.duckdb_path)
    evaluation_predictions, acceptance_summaries = _acceptance_summaries(
        service=service,
        predictions=predictions,
        calendar_dates=calendar_dates,
        config=config,
    )
    promotion_gate = service._load_promotion_gate()
    fold_windows = service._promotion_fold_windows(horizon_days=int(config["label_horizon_dates"]))
    checks = gate_checks_for_model(
        service=service,
        acceptance_summaries=acceptance_summaries,
        promotion_gate=promotion_gate,
        model_name=AUDIT_MODEL,
        required_fold_windows=fold_windows,
    )
    _render_report(
        output_path=output_path,
        service=service,
        acceptance_summaries=acceptance_summaries,
        evaluation_predictions=evaluation_predictions,
        config=config,
        checks=checks,
        eligible_rows=len(eligible.index),
        eligible_dates=int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
    )
    passed, failures = gate_verdict(checks)
    return {
        "output_path": str(output_path),
        "eligible_dates": int(eligible["snapshot_date"].nunique()) if not eligible.empty else 0,
        "verdict": "passes" if passed else "fails",
        "failures": ", ".join(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render P3-W expected production-grid verdict.")
    parser.add_argument("--output", type=Path, default=Path("reports/p3w_expected_tonight.md"))
    args = parser.parse_args()
    result = run(output_path=args.output)
    print(
        "wrote {output_path} eligible_dates={eligible_dates} verdict={verdict} failures={failures}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
