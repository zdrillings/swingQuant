from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
import math
import warnings

import numpy as np
import pandas as pd

from src.research.shortlist_bakeoff_service import (
    MODEL_FEATURE_COLUMNS,
    build_rank_augmented_feature_frame,
    expand_model_feature_columns,
    model_feature_columns_for_profile,
    normalize_shortlist_feature_profile,
)
from src.research.shortlist_universe import (
    eligible_universe_mode_description,
    filter_eligible_universe,
    normalize_eligible_universe_mode,
    normalize_model_scope,
)
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.performance_metrics import annualized_sharpe, newey_west_t_stat, years_required_for_tstat

PROMOTION_BASKET_SIZE = 2
REGIME_MATCHING_MODES = {"off", "train_only", "train_and_flip"}
REGIME_TRANSITION_PURGE_MODES = {"off", "majority", "strict"}
SHORTLIST_HEURISTIC_MODELS = {
    "signal_proxy",
    "reversal_rules",
    "event_signal",
    "structure_factor_signal",
    "structure_factor_event_signal",
}
SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES = {
    "analyst_snapshot_age_days",
    "analyst_revision_snapshot_age_days",
}
REVERSAL_RULE_FEATURES = (
    "rsi_2",
    "rsi_14",
    "ret_1d",
    "ret_5d",
    "close_vs_20d_low",
    "distance_above_20d_high",
    "base_volume_dryup_ratio_20",
    "distance_from_52w_high",
    "sma_50_dist",
    "roc_63",
    "roc_126",
)
REVERSAL_CORE_FEATURES = {"rsi_2", "ret_1d", "close_vs_20d_low"}
EVENT_DRIVEN_FEATURES = (
    "analyst_target_upside",
    "analyst_target_range_pct",
    "analyst_count",
    "analyst_recommendation_score",
    "analyst_eps_revision_breadth",
    "analyst_upgrade_downgrade_score",
    "days_since_last_earnings",
    "days_to_next_earnings",
    "last_earnings_gap_pct",
    "last_earnings_volume_ratio_20",
    "last_earnings_open_vs_20d_high",
    "close_vs_last_earnings_close",
)
EVENT_REACTION_FEATURES = (
    "last_earnings_gap_pct",
    "last_earnings_volume_ratio_20",
    "last_earnings_open_vs_20d_high",
    "close_vs_last_earnings_close",
)
EVENT_ANALYST_FEATURES = (
    "analyst_target_upside",
    "analyst_target_range_pct",
    "analyst_count",
    "analyst_recommendation_score",
    "analyst_eps_revision_breadth",
    "analyst_upgrade_downgrade_score",
)
STRUCTURE_FACTOR_COMPONENTS = (
    ("distance_from_52w_high", 1.0),
    ("distance_above_20d_high", 1.0),
    ("sector_pct_above_50", 1.0),
    ("atr_pct_14", -1.0),
    ("base_range_pct_20", -1.0),
    ("avg_abs_gap_pct_20", -1.0),
    ("roc_126", -1.0),
    ("days_since_last_earnings", -1.0),
    ("days_to_next_earnings", 1.0),
)
STRUCTURE_FACTOR_EVENT_MODEL = "structure_factor_event_signal"
STRUCTURE_FACTOR_BASE_MODEL = "structure_factor_signal"
SHORTLIST_ENSEMBLE_EXCLUDED_MODELS = {STRUCTURE_FACTOR_EVENT_MODEL}


@dataclass(frozen=True)
class ShortlistModelReport:
    output_path: str
    target_column: str
    champion_model: str
    oos_dates: int
    live_candidates: int


class ShortlistModelService:
    XGBOOST_CONFIGS = {
        "baseline": {},
        "balanced_depth4": {
            "max_depth": 4,
            "min_child_weight": 3.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        "shallower_regularized": {
            "max_depth": 3,
            "min_child_weight": 3.0,
            "reg_lambda": 2.0,
        },
        "faster_shallow": {
            "max_depth": 3,
            "learning_rate": 0.07,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    }

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("shortlist_model")

    def run(
        self,
        *,
        top_n: int = 10,
        horizon_days: int = 20,
        min_train_dates: int = 252,
        test_window_dates: int = 20,
        recent_dates: int = 60,
        eligible_universe_mode: str = "passed_only",
        model_scope: str = "global",
        xgboost_config: str = "baseline",
        feature_profile: str = "full",
        target_type: str = "regression",
        oos_stride_dates: int | None = None,
        max_train_dates: int | None = None,
        persist: bool = True,
    ) -> ShortlistModelReport:
        self.db_manager.initialize()
        if target_type == "classification":
            target_column = f"alpha_vs_sector_{int(horizon_days)}d_pos"
        elif target_type == "path":
            target_column = f"path_alpha_vs_sector_{int(horizon_days)}d"
        else:
            target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        evaluation_target_column = (
            target_column if target_type == "path" else f"alpha_vs_sector_{int(horizon_days)}d"
        )
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        feature_profile = normalize_shortlist_feature_profile(feature_profile)
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")
        if target_column not in frame.columns and target_type == "classification":
            source_col = f"alpha_vs_sector_{int(horizon_days)}d"
            if source_col in frame.columns:
                alpha_vals = pd.to_numeric(frame[source_col], errors="coerce")
                frame[target_column] = np.where(
                    alpha_vals.notna(),
                    (alpha_vals > 0.02).astype(float),
                    np.nan,
                )
                self.logger.info("Derived %s from %s on-the-fly (column not yet backfilled).", target_column, source_col)
        if target_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")
        if evaluation_target_column not in frame.columns:
            evaluation_target_column = target_column

        all_snapshots = self._prepare_snapshot_frame(frame)
        matured = self._build_matured_eligible_universe(
            all_snapshots,
            target_column=target_column,
            eligible_universe_mode=eligible_universe_mode,
        )
        if matured.empty:
            raise ValueError("No matured eligible universe rows found for shortlist model.")

        unique_dates = sorted(matured["snapshot_date"].drop_duplicates().tolist())
        if len(unique_dates) <= int(min_train_dates):
            raise ValueError("Not enough eligible snapshot dates for walk-forward shortlist modeling.")

        xgboost_config = self._normalize_xgboost_config(xgboost_config)
        xgboost_params = self._xgboost_params_for_config(xgboost_config)
        resolved_oos_stride_dates = max(
            int(oos_stride_dates) if oos_stride_dates is not None else int(horizon_days),
            1,
        )
        resolved_max_train_dates = self._resolve_max_train_dates(
            max_train_dates=max_train_dates,
            min_train_dates=int(min_train_dates),
        )
        base_feature_columns = model_feature_columns_for_profile(feature_profile)
        expanded_feature_columns = self._filter_model_feature_columns(expand_model_feature_columns(base_feature_columns))
        candidate_models = (
            "signal_proxy",
            "reversal_rules",
            "event_signal",
            "event_ic_model",
            "structure_factor_signal",
            "structure_factor_event_signal",
            "ridge_model",
            "lasso_model",
            "elastic_net_model",
            "ic_sign_model",
            "xgboost_model",
        )
        min_feature_ic = self._load_min_feature_ic()
        min_feature_ic_observation_fraction = self._load_min_feature_ic_observation_fraction()
        regime_matching_mode = self._load_regime_matching_mode()
        min_regime_train_dates = self._load_min_regime_train_dates()
        regime_transition_purge_mode = self._load_regime_transition_purge_mode()
        regime_matching_stats = {
            "attempted_folds": 0,
            "matched_folds": 0,
            "fallback_folds": 0,
            "unknown_folds": 0,
            "live_matched": 0,
            "live_fallback": 0,
        }
        regime_transition_purge_counts = self._regime_transition_purge_dry_run_counts(
            matured,
            horizon_sessions=max(int(horizon_days), 1),
        )
        regime_feature_stats: dict[str, dict[str, object]] = {}
        feature_ic_report = self._feature_ic_report(
            matured,
            target_column=evaluation_target_column,
            min_train_dates=int(min_train_dates),
            test_window_dates=int(test_window_dates),
            evaluation_stride_dates=resolved_oos_stride_dates,
            label_horizon_dates=max(int(horizon_days), 1),
            min_feature_ic=min_feature_ic,
            min_observation_fraction=min_feature_ic_observation_fraction,
            feature_columns_override=expanded_feature_columns,
        )
        feature_columns_override = feature_ic_report["surviving_features"]
        if not feature_columns_override:
            self.logger.warning(
                "No features survived the diagnostic full-OOS feature IC report at min_feature_ic=%.4f; "
                "fold-local training screens will still be evaluated.",
                float(min_feature_ic),
            )

        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name in candidate_models:
            predicted = self._walk_forward_predictions(
                matured,
                target_column=target_column,
                evaluation_target_column=evaluation_target_column,
                model_name=model_name,
                min_train_dates=int(min_train_dates),
                max_train_dates=resolved_max_train_dates,
                test_window_dates=int(test_window_dates),
                evaluation_stride_dates=resolved_oos_stride_dates,
                label_horizon_dates=max(int(horizon_days), 1),
                model_scope=model_scope,
                xgboost_params=xgboost_params if model_name == "xgboost_model" else None,
                feature_columns_override=expanded_feature_columns,
                min_feature_ic=min_feature_ic,
                min_feature_ic_observation_fraction=min_feature_ic_observation_fraction,
                regime_matching_mode=regime_matching_mode,
                min_regime_train_dates=min_regime_train_dates,
                regime_transition_purge_mode=regime_transition_purge_mode if target_type != "path" else "off",
                regime_matching_stats=regime_matching_stats,
                regime_feature_stats=regime_feature_stats,
            )
            if predicted is not None and not predicted.empty:
                if regime_matching_mode == "train_and_flip":
                    predicted = self._apply_regime_conditional_score_flip(
                        predicted,
                        horizon_sessions=max(int(horizon_days), 1),
                        fallback_only=True,
                    )
                model_predictions[model_name] = predicted
        ensemble_predictions = self._build_ensemble_predictions(model_predictions)
        if ensemble_predictions is not None:
            model_predictions["ensemble_model"] = ensemble_predictions

        if not model_predictions:
            raise ValueError("No shortlist models produced out-of-sample predictions.")

        report_path = self.db_manager.paths.reports_dir / "shortlist_model.md"
        oos_path = self.db_manager.paths.reports_dir / "shortlist_model_oos_predictions.csv"
        live_path = self.db_manager.paths.reports_dir / "shortlist_model_live_predictions.csv"
        promotion_top_n = PROMOTION_BASKET_SIZE
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        combined_predictions = pd.concat(
            [
                predictions.assign(model_name=model_name, dataset_split="oos")
                for model_name, predictions in model_predictions.items()
            ],
            axis=0,
            ignore_index=True,
        )
        combined_predictions = self._annotate_oos_artifact_ranks(combined_predictions)
        combined_predictions = self._annotate_calibrated_probabilities(
            combined_predictions,
            target_column=target_column,
        )
        for model_name, predictions in list(model_predictions.items()):
            annotated = self._annotate_oos_artifact_ranks(predictions.assign(model_name=model_name))
            annotated = self._annotate_calibrated_probabilities(
                annotated,
                target_column=target_column,
            )
            model_predictions[model_name] = annotated.drop(columns=["model_name"], errors="ignore")
        full_summaries = pd.DataFrame(
            [
                self._evaluate_predictions(
                    predictions=predictions,
                    top_n=promotion_top_n,
                    target_column=evaluation_target_column,
                    model_name=model_name,
                )
                for model_name, predictions in model_predictions.items()
            ]
        )
        recent_summary_rows: list[dict[str, object]] = []
        for model_name, predictions in model_predictions.items():
            recent_prediction_dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
            recent_predictions = predictions[predictions["snapshot_date"].isin(recent_prediction_dates)].copy()
            recent_summary_rows.append(
                self._evaluate_predictions(
                    predictions=recent_predictions,
                    top_n=promotion_top_n,
                    target_column=evaluation_target_column,
                    model_name=model_name,
                )
            )
        recent_summaries = pd.DataFrame(recent_summary_rows)
        promotion_gate = self._load_promotion_gate()
        required_recent_windows = self._promotion_recent_windows(horizon_days=int(horizon_days))
        required_fold_windows = self._promotion_fold_windows(horizon_days=int(horizon_days))
        acceptance_summaries = pd.DataFrame(
            [
                row
                for model_name, predictions in model_predictions.items()
                for row in self._rolling_window_summaries(
                    predictions=predictions,
                    target_column=evaluation_target_column,
                    model_name=model_name,
                    top_n=promotion_top_n,
                    windows=required_recent_windows,
                    fold_windows=required_fold_windows,
                    fold_size=int(test_window_dates),
                    include_full_oos=True,
                ).to_dict(orient="records")
            ]
        )
        try:
            champion_model, champion_gate_passed = self._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
                required_recent_windows=required_recent_windows,
                required_fold_windows=required_fold_windows,
            )
        except ValueError as exc:
            if persist:
                self._decommission_active_champions(
                    generated_at=generated_at,
                    horizon_days=int(horizon_days),
                    eligible_universe_mode=eligible_universe_mode,
                    model_scope=model_scope,
                    xgboost_config=xgboost_config,
                    feature_profile=feature_profile,
                    reason=str(exc),
                )
            combined_predictions.to_csv(oos_path, index=False)
            lines = self._build_report_lines(
                target_column=target_column,
                evaluation_target_column=evaluation_target_column,
                top_n=int(top_n),
                promotion_top_n=promotion_top_n,
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                candidate_models=tuple(model_predictions.keys()),
                selected_model="n/a",
                selected_model_gate_passed=False,
                xgboost_config=xgboost_config,
                feature_profile=feature_profile,
                min_train_dates=int(min_train_dates),
                max_train_dates=resolved_max_train_dates,
                test_window_dates=int(test_window_dates),
                evaluation_stride_dates=resolved_oos_stride_dates,
                label_horizon_dates=max(int(horizon_days), 1),
                feature_ic_path=self.db_manager.paths.reports_dir / "feature_ic_report.md",
                min_feature_ic=min_feature_ic,
                surviving_features=feature_columns_override,
                min_feature_ic_observation_fraction=min_feature_ic_observation_fraction,
                regime_matching_mode=regime_matching_mode,
                regime_transition_purge_mode=regime_transition_purge_mode,
                regime_transition_purge_counts=regime_transition_purge_counts,
                regime_matching_stats=regime_matching_stats,
                regime_feature_stats=regime_feature_stats,
                eligible_rows=len(matured.index),
                eligible_dates=int(matured["snapshot_date"].nunique()),
                oos_prediction_dates=int(combined_predictions["snapshot_date"].nunique()),
                oos_path=oos_path,
                live_path=live_path,
                generated_at=generated_at,
                full_summaries=full_summaries,
                recent_summaries=recent_summaries,
                recent_dates=min(int(recent_dates), int(combined_predictions["snapshot_date"].nunique())),
                promotion_gate=promotion_gate,
                required_recent_windows=required_recent_windows,
                required_fold_windows=required_fold_windows,
                acceptance_summaries=acceptance_summaries,
                oos_predictions=combined_predictions,
                failure_reason=str(exc),
                days_since_last_champion=self._days_since_last_champion(
                    generated_at=generated_at,
                    horizon_days=int(horizon_days),
                ),
            )
            report_path.write_text("\n".join(lines), encoding="utf-8")
            raise

        live_base_predictions: dict[str, pd.DataFrame] = {}
        for model_name in candidate_models:
            if model_name not in model_predictions:
                continue
            scored = self._score_live_snapshot(
                all_snapshots=all_snapshots,
                matured=matured,
                model_name=model_name,
                target_column=target_column,
                feature_ic_target_column=evaluation_target_column,
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                xgboost_params=xgboost_params if model_name == "xgboost_model" else None,
                feature_columns_override=expanded_feature_columns,
                min_feature_ic=min_feature_ic,
                min_feature_ic_observation_fraction=min_feature_ic_observation_fraction,
                max_train_dates=resolved_max_train_dates,
                regime_matching_mode=regime_matching_mode,
                min_regime_train_dates=min_regime_train_dates,
                regime_matching_stats=regime_matching_stats,
                regime_feature_stats=regime_feature_stats,
            )
            if scored is not None and not scored.empty:
                if regime_matching_mode == "train_and_flip":
                    scored = self._apply_regime_conditional_score_flip(
                        scored,
                        horizon_sessions=max(int(horizon_days), 1),
                        fallback_only=True,
                    )
                scored = self._apply_calibration_from_oos(
                    scored,
                    model_predictions.get(model_name, pd.DataFrame()),
                    target_column=target_column,
                )
                live_base_predictions[model_name] = scored
        live_ensemble_predictions = self._build_ensemble_predictions(live_base_predictions)
        if live_ensemble_predictions is not None:
            live_ensemble_predictions = self._apply_calibration_from_oos(
                live_ensemble_predictions,
                model_predictions.get("ensemble_model", pd.DataFrame()),
                target_column=target_column,
            )
            live_base_predictions["ensemble_model"] = live_ensemble_predictions
        live_predictions_all = live_base_predictions.get(champion_model)
        if live_predictions_all is None:
            raise ValueError(f"No live predictions available for champion_model={champion_model}.")
        live_predictions = live_predictions_all.sort_values(
            ["predicted_alpha", "ticker"],
            ascending=[False, True],
        ).head(int(top_n)).reset_index(drop=True)
        combined_predictions.to_csv(oos_path, index=False)
        live_predictions_all.to_csv(live_path, index=False)

        lines = self._build_report_lines(
            target_column=target_column,
            evaluation_target_column=evaluation_target_column,
            top_n=int(top_n),
            promotion_top_n=promotion_top_n,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            candidate_models=tuple(model_predictions.keys()),
            selected_model=champion_model,
            selected_model_gate_passed=bool(champion_gate_passed),
            xgboost_config=xgboost_config,
            feature_profile=feature_profile,
            min_train_dates=int(min_train_dates),
            max_train_dates=resolved_max_train_dates,
            test_window_dates=int(test_window_dates),
            evaluation_stride_dates=resolved_oos_stride_dates,
            label_horizon_dates=max(int(horizon_days), 1),
            feature_ic_path=self.db_manager.paths.reports_dir / "feature_ic_report.md",
            min_feature_ic=min_feature_ic,
            surviving_features=feature_columns_override,
            min_feature_ic_observation_fraction=min_feature_ic_observation_fraction,
            regime_matching_mode=regime_matching_mode,
            regime_transition_purge_mode=regime_transition_purge_mode,
            regime_transition_purge_counts=regime_transition_purge_counts,
            regime_matching_stats=regime_matching_stats,
            regime_feature_stats=regime_feature_stats,
            eligible_rows=len(matured.index),
            eligible_dates=int(matured["snapshot_date"].nunique()),
            oos_prediction_dates=int(combined_predictions["snapshot_date"].nunique()),
            oos_path=oos_path,
            live_path=live_path,
            generated_at=generated_at,
            full_summaries=full_summaries,
            recent_summaries=recent_summaries,
            recent_dates=min(int(recent_dates), int(combined_predictions["snapshot_date"].nunique())),
            promotion_gate=promotion_gate,
            required_recent_windows=required_recent_windows,
            required_fold_windows=required_fold_windows,
            acceptance_summaries=acceptance_summaries,
            oos_predictions=combined_predictions,
        )
        lines.extend(
            self._render_summary_table(
                self._rolling_window_summaries(
                    predictions=model_predictions[champion_model],
                    target_column=evaluation_target_column,
                    model_name=champion_model,
                    top_n=promotion_top_n,
                    windows=required_recent_windows,
                    fold_windows=required_fold_windows,
                    fold_size=int(test_window_dates),
                    include_full_oos=True,
                ),
                heading="## Champion Rolling Acceptance Windows",
            )
        )
        lines.extend(
            self._render_sector_contribution(
                predictions=model_predictions[champion_model],
                target_column=evaluation_target_column,
                top_n=int(top_n),
                heading="## Champion Sector Contribution",
            )
        )
        lines.extend(self._render_live_candidates(champion_model=champion_model, frame=live_predictions))
        report_path.write_text("\n".join(lines), encoding="utf-8")

        if persist:
            self.db_manager.insert_shortlist_model_run(
                row={
                    "generated_at": generated_at,
                    "horizon_days": int(horizon_days),
                    "eligible_universe_mode": eligible_universe_mode,
                    "model_scope": model_scope,
                    "xgboost_config": xgboost_config,
                    "feature_profile": feature_profile,
                    "top_n": int(top_n),
                    "min_train_dates": int(min_train_dates),
                    "test_window_dates": int(test_window_dates),
                    "recent_dates": int(recent_dates),
                    "champion_model": champion_model,
                    "target_column": target_column,
                    "eligible_rows": len(matured.index),
                    "eligible_dates": int(matured["snapshot_date"].nunique()),
                    "oos_dates": int(combined_predictions["snapshot_date"].nunique()),
                    "live_snapshot_date": str(live_predictions_all["snapshot_date"].max().date()) if not live_predictions_all.empty else None,
                    "report_path": str(report_path),
                }
            )
            persistence_rows = [
                {
                    "model_name": row["model_name"],
                    "dataset_split": "oos",
                    "snapshot_date": str(pd.Timestamp(row["snapshot_date"]).date()),
                    "ticker": row["ticker"],
                    "sector": row.get("sector"),
                    "eligible_universe_mode": eligible_universe_mode,
                    "model_scope": model_scope,
                    "md_volume_30d": row.get("md_volume_30d"),
                    "predicted_alpha": row.get("predicted_alpha"),
                    "actual_alpha_vs_sector": row.get(evaluation_target_column),
                    "details": {
                        "model_top_reasons": self._ensure_reason_list(row.get("model_top_reasons")),
                        "model_reason_summary": row.get("model_reason_summary"),
                        "calibrated_p_beat_sector": row.get("calibrated_p_beat_sector"),
                        "model_rank": row.get("model_rank"),
                        "raw_predicted_alpha": row.get("raw_predicted_alpha"),
                        "regime_classification": row.get("regime_classification"),
                        "regime_flip_applied": row.get("regime_flip_applied"),
                    },
                }
                for row in combined_predictions.to_dict(orient="records")
            ]
            for model_name, live_frame in live_base_predictions.items():
                persistence_rows.extend(
                    [
                        {
                            "model_name": model_name,
                            "dataset_split": "live",
                            "snapshot_date": str(pd.Timestamp(row["snapshot_date"]).date()),
                            "ticker": row["ticker"],
                            "sector": row.get("sector"),
                            "eligible_universe_mode": eligible_universe_mode,
                            "model_scope": model_scope,
                            "md_volume_30d": row.get("md_volume_30d"),
                            "predicted_alpha": row.get("predicted_alpha"),
                            "actual_alpha_vs_sector": None,
                            "details": {
                                "model_top_reasons": self._ensure_reason_list(row.get("model_top_reasons")),
                                "model_reason_summary": row.get("model_reason_summary"),
                                "calibrated_p_beat_sector": row.get("calibrated_p_beat_sector"),
                                "model_rank": row.get("model_rank"),
                                "raw_predicted_alpha": row.get("raw_predicted_alpha"),
                                "regime_classification": row.get("regime_classification"),
                                "regime_flip_applied": row.get("regime_flip_applied"),
                            },
                        }
                        for row in live_frame.to_dict(orient="records")
                    ]
                )
            self.db_manager.replace_shortlist_model_predictions(
                generated_at=generated_at,
                horizon_days=int(horizon_days),
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                rows=persistence_rows,
            )

        return ShortlistModelReport(
            output_path=str(report_path),
            target_column=target_column,
            champion_model=champion_model,
            oos_dates=int(combined_predictions["snapshot_date"].nunique()),
            live_candidates=len(live_predictions.index),
        )

    def _prepare_snapshot_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        working["md_volume_30d"] = pd.to_numeric(working["md_volume_30d"], errors="coerce")
        working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
        working["passed_any_strategy"] = working["passed_any_strategy"].astype(bool)
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _normalize_xgboost_config(self, xgboost_config: str) -> str:
        normalized = str(xgboost_config or "baseline").strip().lower()
        if normalized not in self.XGBOOST_CONFIGS:
            valid = ", ".join(sorted(self.XGBOOST_CONFIGS))
            raise ValueError(f"Unsupported xgboost_config '{xgboost_config}'. Valid choices: {valid}.")
        return normalized

    def _xgboost_params_for_config(self, xgboost_config: str) -> dict[str, float | int]:
        return dict(self.XGBOOST_CONFIGS.get(str(xgboost_config), {}))

    def _build_matured_eligible_universe(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        feature_ic_target_column: str | None = None,
        eligible_universe_mode: str,
    ) -> pd.DataFrame:
        working = frame.copy()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=["snapshot_date", target_column]).copy()
        matured_rows = working
        working = filter_eligible_universe(
            matured_rows,
            eligible_universe_mode=eligible_universe_mode,
        )
        working = self._extend_reversal_eligible_universe(
            base_frame=matured_rows,
            eligible_frame=working,
            horizon_sessions=self._target_horizon_dates(target_column),
        )
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _build_live_eligible_universe(
        self,
        frame: pd.DataFrame,
        *,
        eligible_universe_mode: str,
        horizon_sessions: int | None = None,
    ) -> pd.DataFrame:
        working = filter_eligible_universe(
            frame.copy(),
            eligible_universe_mode=eligible_universe_mode,
        )
        working = self._extend_reversal_eligible_universe(
            base_frame=frame,
            eligible_frame=working,
            horizon_sessions=int(horizon_sessions or 20),
        )
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _extend_reversal_eligible_universe(
        self,
        *,
        base_frame: pd.DataFrame,
        eligible_frame: pd.DataFrame,
        horizon_sessions: int,
    ) -> pd.DataFrame:
        required = {"snapshot_date", "ticker", "md_volume_30d", "adj_close", "sma_200_dist", "roc_63", "rsi_14"}
        if base_frame.empty or not required.issubset(base_frame.columns):
            return eligible_frame.copy()
        regime_by_date = self._regime_classifications_by_prediction_date(
            base_frame["snapshot_date"].dropna().drop_duplicates().tolist(),
            horizon_sessions=horizon_sessions,
        )
        if not regime_by_date:
            return eligible_frame.copy()
        working = base_frame.copy()
        normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
        working["_lagged_regime"] = normalized_dates.map(regime_by_date).fillna("unknown")
        liquidity = pd.to_numeric(working["md_volume_30d"], errors="coerce")
        close = pd.to_numeric(working["adj_close"], errors="coerce")
        sma_200 = pd.to_numeric(working["sma_200_dist"], errors="coerce")
        roc_63 = pd.to_numeric(working["roc_63"], errors="coerce")
        rsi_14 = pd.to_numeric(working["rsi_14"], errors="coerce")
        pullback_mask = (
            working["_lagged_regime"].astype(str).eq("reversal")
            & liquidity.ge(20_000_000.0)
            & close.gt(0.0)
            & sma_200.ge(-0.10)
            & roc_63.le(0.0)
            & rsi_14.lt(60.0)
        )
        extension = working.loc[pullback_mask].drop(columns=["_lagged_regime"]).copy()
        if extension.empty:
            return eligible_frame.copy()
        combined = pd.concat([eligible_frame.copy(), extension], axis=0, ignore_index=True)
        return combined.drop_duplicates(subset=["snapshot_date", "ticker"], keep="first").copy()

    def _walk_forward_predictions(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        evaluation_target_column: str | None = None,
        model_name: str,
        min_train_dates: int,
        test_window_dates: int,
        model_scope: str,
        evaluation_stride_dates: int | None = None,
        label_horizon_dates: int | None = None,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
        min_feature_ic: float | None = None,
        min_feature_ic_observation_fraction: float = 0.20,
        max_train_dates: int | None = None,
        regime_matching_mode: str = "off",
        min_regime_train_dates: int = 120,
        regime_transition_purge_mode: str = "off",
        regime_matching_stats: dict[str, int] | None = None,
        regime_feature_stats: dict[str, dict[str, object]] | None = None,
    ) -> pd.DataFrame | None:
        dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
        evaluation_target_column = evaluation_target_column or target_column
        folds: list[pd.DataFrame] = []
        start_index = int(min_train_dates)
        stride = max(int(evaluation_stride_dates or test_window_dates), 1)
        label_embargo = max(int(label_horizon_dates or 0), 0)
        normalized_regime_mode = self._normalize_regime_matching_mode(regime_matching_mode)
        normalized_transition_purge_mode = self._normalize_regime_transition_purge_mode(regime_transition_purge_mode)
        regime_by_date = (
            self._regime_classifications_by_prediction_date(dates, horizon_sessions=label_embargo)
            if normalized_regime_mode != "off" or normalized_transition_purge_mode != "off"
            else {}
        )
        while start_index < len(dates):
            test_dates = dates[start_index : start_index + max(int(test_window_dates), 1)]
            if not test_dates:
                break
            train_end_index = max(0, start_index - label_embargo)
            train_pool_dates = dates[:train_end_index]
            if len(train_pool_dates) < int(min_train_dates):
                start_index += stride
                continue
            train_date_window, matched_training, test_regime, feature_screen_dates = self._regime_matched_train_dates(
                train_date_window=train_pool_dates,
                test_dates=test_dates,
                min_train_dates=int(min_regime_train_dates),
                max_train_dates=max_train_dates,
                regime_by_date=regime_by_date,
                mode=normalized_regime_mode,
                stats=regime_matching_stats,
            )
            if normalized_regime_mode == "off" and max_train_dates is not None:
                train_date_window = train_date_window[-max(int(max_train_dates), 1):]
            if normalized_transition_purge_mode != "off":
                train_date_window = self._purge_regime_transition_dates(
                    train_date_window,
                    all_dates=dates,
                    horizon_sessions=label_embargo,
                    regime_by_date=regime_by_date,
                    mode=normalized_transition_purge_mode,
                )
                if len(train_date_window) < int(min_train_dates):
                    self.logger.info(
                        "Walk-forward fold %s skipped %s after regime transition purge left %d train dates.",
                        pd.Timestamp(test_dates[0]).date(),
                        model_name,
                        len(train_date_window),
                    )
                    start_index += stride
                    continue
            train_dates = set(train_date_window)
            train_frame = frame[frame["snapshot_date"].isin(train_dates)].copy()
            raw_train_rows = len(train_frame.index)
            train_frame = self._stride_training_labels(
                train_frame,
                dates=dates,
                anchor_index=start_index,
                label_horizon_dates=label_embargo,
            )
            self.logger.info(
                "Walk-forward fold %s train_rows=%d after label stride from raw_train_rows=%d",
                pd.Timestamp(test_dates[0]).date(),
                len(train_frame.index),
                raw_train_rows,
            )
            test_frame = frame[frame["snapshot_date"].isin(test_dates)].copy()
            test_frame["regime_matched_training_applied"] = bool(matched_training)
            test_frame["regime_matching_test_regime"] = test_regime or "unknown"
            fold_feature_columns = feature_columns_override
            if min_feature_ic is not None and model_name not in SHORTLIST_HEURISTIC_MODELS:
                feature_screen_frame = train_frame
                if feature_screen_dates and str(test_regime or "unknown") != "neutral":
                    feature_screen_dates_set = set(feature_screen_dates)
                    scoped = frame[frame["snapshot_date"].isin(feature_screen_dates_set)].copy()
                    if not scoped.empty:
                        feature_screen_frame = self._stride_training_labels(
                            scoped,
                            dates=dates,
                            anchor_index=start_index,
                            label_horizon_dates=label_embargo,
                        )
                ic_survivors = self._feature_ic_survivors_from_frame(
                    feature_screen_frame,
                    target_column=evaluation_target_column,
                    min_feature_ic=float(min_feature_ic),
                    min_observation_fraction=float(min_feature_ic_observation_fraction),
                )
                self._record_regime_feature_survivors(
                    regime_feature_stats,
                    regime=test_regime,
                    survivors=ic_survivors,
                )
                if feature_columns_override is None:
                    fold_feature_columns = ic_survivors
                else:
                    allowed_features = set(feature_columns_override)
                    fold_feature_columns = [
                        feature for feature in ic_survivors if feature in allowed_features
                    ]
                if not fold_feature_columns:
                    self.logger.info(
                        "Walk-forward fold %s skipped %s because no train-only features cleared min_feature_ic=%.4f",
                        pd.Timestamp(test_dates[0]).date(),
                        model_name,
                        float(min_feature_ic),
                    )
                    start_index += stride
                    continue
            scored = self._score_model(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                model_scope=model_scope,
                xgboost_params=xgboost_params,
                feature_columns_override=fold_feature_columns,
            )
            if scored is not None and not scored.empty:
                output_columns = [
                    "snapshot_date",
                    "ticker",
                    "sector",
                    "md_volume_30d",
                    target_column,
                    "predicted_alpha",
                    "model_top_reasons",
                    "model_reason_summary",
                    "regime_matched_training_applied",
                    "regime_matching_test_regime",
                ]
                if evaluation_target_column != target_column and evaluation_target_column in scored.columns:
                    output_columns.insert(5, evaluation_target_column)
                folds.append(
                    scored[output_columns].copy()
                )
            start_index += stride
        if not folds:
            return None
        return pd.concat(folds, axis=0, ignore_index=True)

    def _stride_training_labels(
        self,
        train_frame: pd.DataFrame,
        *,
        dates: list,
        anchor_index: int,
        label_horizon_dates: int,
    ) -> pd.DataFrame:
        horizon = max(int(label_horizon_dates or 0), 0)
        if horizon <= 1 or train_frame.empty:
            return train_frame
        date_index = {date_value: index for index, date_value in enumerate(dates)}
        working = train_frame.copy()
        working["_snapshot_date_index"] = working["snapshot_date"].map(date_index)
        keep_mask = (
            working["_snapshot_date_index"].notna()
            & ((((int(anchor_index) - 1) - working["_snapshot_date_index"].astype(int)) % horizon) == 0)
        )
        return working.loc[keep_mask].drop(columns=["_snapshot_date_index"]).copy()

    def _score_live_snapshot(
        self,
        *,
        all_snapshots: pd.DataFrame,
        matured: pd.DataFrame,
        model_name: str,
        target_column: str,
        eligible_universe_mode: str,
        model_scope: str,
        feature_ic_target_column: str | None = None,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
        min_feature_ic: float | None = None,
        min_feature_ic_observation_fraction: float = 0.20,
        max_train_dates: int | None = None,
        regime_matching_mode: str = "off",
        min_regime_train_dates: int = 120,
        regime_matching_stats: dict[str, int] | None = None,
        regime_feature_stats: dict[str, dict[str, object]] | None = None,
    ) -> pd.DataFrame:
        latest_date = all_snapshots["snapshot_date"].max()
        feature_ic_target_column = feature_ic_target_column or target_column
        live_snapshot = all_snapshots[all_snapshots["snapshot_date"] == latest_date].copy()
        live_snapshot = self._build_live_eligible_universe(
            live_snapshot,
            eligible_universe_mode=eligible_universe_mode,
            horizon_sessions=self._target_horizon_dates(target_column),
        )
        if live_snapshot.empty:
            return live_snapshot.assign(predicted_alpha=pd.Series(dtype=float))
        safe_train = matured[matured["snapshot_date"] < latest_date].copy()
        if safe_train.empty:
            safe_train = matured
        safe_train_dates = sorted(safe_train["snapshot_date"].drop_duplicates().tolist())
        normalized_regime_mode = self._normalize_regime_matching_mode(regime_matching_mode)
        test_regime = "unknown"
        feature_screen_dates: list = []
        if normalized_regime_mode != "off":
            regime_by_date = self._regime_classifications_by_prediction_date(
                safe_train_dates + [latest_date],
                horizon_sessions=self._target_horizon_dates(target_column),
            )
            safe_train_dates, matched_training, resolved_test_regime, feature_screen_dates = self._regime_matched_train_dates(
                train_date_window=safe_train_dates,
                test_dates=[latest_date],
                min_train_dates=int(min_regime_train_dates),
                max_train_dates=max_train_dates,
                regime_by_date=regime_by_date,
                mode=normalized_regime_mode,
                stats=regime_matching_stats,
                live=True,
            )
            test_regime = resolved_test_regime or "unknown"
            safe_train = safe_train[safe_train["snapshot_date"].isin(set(safe_train_dates))].copy()
            live_snapshot["regime_matched_training_applied"] = bool(matched_training)
        else:
            if max_train_dates is not None and len(safe_train_dates) > int(max_train_dates):
                safe_train_dates = safe_train_dates[-max(int(max_train_dates), 1):]
                safe_train = safe_train[safe_train["snapshot_date"].isin(safe_train_dates)].copy()
            live_snapshot["regime_matched_training_applied"] = False
        live_snapshot["regime_matching_test_regime"] = test_regime
        if len(safe_train_dates) > 1:
            safe_train = self._stride_training_labels(
                safe_train,
                dates=safe_train_dates,
                anchor_index=len(safe_train_dates),
                label_horizon_dates=self._target_horizon_dates(target_column),
            )
        live_feature_columns = feature_columns_override
        if min_feature_ic is not None and model_name not in SHORTLIST_HEURISTIC_MODELS:
            feature_screen_frame = safe_train
            if feature_screen_dates and str(test_regime or "unknown") != "neutral":
                scoped = matured[matured["snapshot_date"].isin(set(feature_screen_dates))].copy()
                if not scoped.empty:
                    feature_screen_frame = scoped
            ic_survivors = self._feature_ic_survivors_from_frame(
                feature_screen_frame,
                target_column=feature_ic_target_column,
                min_feature_ic=float(min_feature_ic),
                min_observation_fraction=float(min_feature_ic_observation_fraction),
            )
            self._record_regime_feature_survivors(
                regime_feature_stats,
                regime=test_regime,
                survivors=ic_survivors,
            )
            if feature_columns_override is None:
                live_feature_columns = ic_survivors
            else:
                allowed_features = set(feature_columns_override)
                live_feature_columns = [feature for feature in ic_survivors if feature in allowed_features]
        scored = self._score_model(
            model_name=model_name,
            train_frame=safe_train,
            test_frame=live_snapshot,
            target_column=target_column,
            model_scope=model_scope,
            xgboost_params=xgboost_params,
            feature_columns_override=live_feature_columns,
        )
        if scored is None or scored.empty:
            return live_snapshot.assign(predicted_alpha=pd.Series(dtype=float))
        return scored

    def _apply_regime_conditional_score_flip(
        self,
        predictions: pd.DataFrame,
        *,
        horizon_sessions: int,
        fallback_only: bool = False,
    ) -> pd.DataFrame:
        if predictions.empty or "snapshot_date" not in predictions.columns or "predicted_alpha" not in predictions.columns:
            return predictions.copy()
        working = predictions.copy()
        score = pd.to_numeric(working["predicted_alpha"], errors="coerce")
        if "raw_predicted_alpha" not in working.columns:
            working["raw_predicted_alpha"] = score
        classifications = self._regime_classifications_by_prediction_date(
            working["snapshot_date"].dropna().tolist(),
            horizon_sessions=horizon_sessions,
        )
        if not classifications:
            working["regime_classification"] = working.get("regime_classification", pd.Series("unknown", index=working.index))
            working["regime_flip_applied"] = False
            return working
        normalized_dates = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
        working["regime_classification"] = normalized_dates.map(classifications).fillna("unknown")
        flip_mask = working["regime_classification"].astype(str).eq("reversal") & score.notna()
        if fallback_only and "regime_matched_training_applied" in working.columns:
            matched = working["regime_matched_training_applied"].fillna(False).astype(bool)
            flip_mask = flip_mask & ~matched
        working["regime_flip_applied"] = flip_mask
        working.loc[flip_mask, "predicted_alpha"] = -score.loc[flip_mask]
        return working

    def _regime_matched_train_dates(
        self,
        *,
        train_date_window: list,
        test_dates: list,
        min_train_dates: int,
        max_train_dates: int | None = None,
        regime_by_date: dict[pd.Timestamp, str],
        mode: str,
        stats: dict[str, int] | None = None,
        live: bool = False,
    ) -> tuple[list, bool, str | None, list]:
        if mode == "off":
            capped = list(train_date_window)
            if max_train_dates is not None:
                capped = capped[-max(int(max_train_dates), 1):]
            return capped, False, None, []
        if stats is not None:
            stats["attempted_folds"] = int(stats.get("attempted_folds", 0)) + (0 if live else 1)
        test_regime = self._majority_regime_for_dates(test_dates, regime_by_date)
        if test_regime is None:
            if stats is not None:
                key = "live_fallback" if live else "unknown_folds"
                stats[key] = int(stats.get(key, 0)) + 1
            fallback = list(train_date_window)
            if max_train_dates is not None:
                fallback = fallback[-max(int(max_train_dates), 1):]
            return fallback, False, None, []
        matched_dates = [
            date_value
            for date_value in train_date_window
            if regime_by_date.get(pd.Timestamp(date_value).normalize()) == test_regime
        ]
        required_matched_dates = max(60, int(math.ceil(len(matched_dates) * 0.80)))
        if len(matched_dates) >= required_matched_dates:
            if stats is not None:
                key = "live_matched" if live else "matched_folds"
                stats[key] = int(stats.get(key, 0)) + 1
            selected_dates = list(matched_dates)
            if max_train_dates is not None:
                selected_dates = selected_dates[-max(int(max_train_dates), 1):]
            return selected_dates, True, str(test_regime), list(matched_dates)
        if stats is not None:
            key = "live_fallback" if live else "fallback_folds"
            stats[key] = int(stats.get(key, 0)) + 1
        fallback = list(train_date_window)
        if max_train_dates is not None:
            fallback = fallback[-max(int(max_train_dates), 1):]
        return fallback, False, str(test_regime), list(matched_dates)

    def _majority_regime_for_dates(
        self,
        dates: list,
        regime_by_date: dict[pd.Timestamp, str],
    ) -> str | None:
        counts: dict[str, int] = {}
        for raw_date in dates:
            regime = regime_by_date.get(pd.Timestamp(raw_date).normalize())
            if regime in (None, "", "unknown"):
                continue
            counts[str(regime)] = counts.get(str(regime), 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _regime_transition_purge_dry_run_counts(
        self,
        frame: pd.DataFrame,
        *,
        horizon_sessions: int,
    ) -> dict[str, int]:
        if frame.empty or "snapshot_date" not in frame.columns:
            return {"rows": 0, "majority": 0, "strict": 0}
        dates = sorted(pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.normalize().drop_duplicates().tolist())
        regime_by_date = self._regime_classifications_by_prediction_date(
            dates,
            horizon_sessions=max(int(horizon_sessions), 1),
        )
        if not dates or not regime_by_date:
            return {"rows": int(len(frame.index)), "majority": 0, "strict": 0}
        transition_by_date = {
            pd.Timestamp(date_value).normalize(): {
                "majority": self._date_crosses_regime_boundary(
                    date_value,
                    all_dates=dates,
                    horizon_sessions=horizon_sessions,
                    regime_by_date=regime_by_date,
                    mode="majority",
                ),
                "strict": self._date_crosses_regime_boundary(
                    date_value,
                    all_dates=dates,
                    horizon_sessions=horizon_sessions,
                    regime_by_date=regime_by_date,
                    mode="strict",
                ),
            }
            for date_value in dates
        }
        normalized_dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
        majority = normalized_dates.map(lambda value: bool(transition_by_date.get(pd.Timestamp(value).normalize(), {}).get("majority")) if pd.notna(value) else False)
        strict = normalized_dates.map(lambda value: bool(transition_by_date.get(pd.Timestamp(value).normalize(), {}).get("strict")) if pd.notna(value) else False)
        return {
            "rows": int(len(frame.index)),
            "majority": int(majority.sum()),
            "strict": int(strict.sum()),
        }

    def _purge_regime_transition_dates(
        self,
        train_dates: list,
        *,
        all_dates: list,
        horizon_sessions: int,
        regime_by_date: dict[pd.Timestamp, str],
        mode: str,
    ) -> list:
        normalized_mode = self._normalize_regime_transition_purge_mode(mode)
        if normalized_mode == "off" or not train_dates or not regime_by_date:
            return list(train_dates)
        return [
            date_value
            for date_value in train_dates
            if not self._date_crosses_regime_boundary(
                date_value,
                all_dates=all_dates,
                horizon_sessions=horizon_sessions,
                regime_by_date=regime_by_date,
                mode=normalized_mode,
            )
        ]

    def _date_crosses_regime_boundary(
        self,
        date_value,
        *,
        all_dates: list,
        horizon_sessions: int,
        regime_by_date: dict[pd.Timestamp, str],
        mode: str,
    ) -> bool:
        entry_date = pd.Timestamp(date_value).normalize()
        entry_regime = regime_by_date.get(entry_date)
        if entry_regime in (None, "", "unknown"):
            return False
        normalized_dates = [pd.Timestamp(raw).normalize() for raw in all_dates]
        try:
            entry_index = normalized_dates.index(entry_date)
        except ValueError:
            return False
        horizon = max(int(horizon_sessions), 1)
        forward_dates = normalized_dates[entry_index + 1 : entry_index + 1 + horizon]
        forward_regimes = [
            str(regime_by_date[forward_date])
            for forward_date in forward_dates
            if regime_by_date.get(forward_date) not in (None, "", "unknown")
        ]
        if not forward_regimes:
            return False
        mismatches = sum(1 for regime in forward_regimes if regime != str(entry_regime))
        if mode == "strict":
            return mismatches > 0
        if mode == "majority":
            return mismatches > (len(forward_regimes) / 2.0)
        return False

    def _forward_regime_share_for_dates(
        self,
        entry_dates: list,
        *,
        all_dates: list,
        horizon_sessions: int,
        regime_by_date: dict[pd.Timestamp, str],
    ) -> dict[str, float]:
        counts = {"neutral": 0, "trending": 0, "reversal": 0}
        total = 0
        normalized_dates = [pd.Timestamp(raw).normalize() for raw in all_dates]
        date_index = {date_value: index for index, date_value in enumerate(normalized_dates)}
        horizon = max(int(horizon_sessions), 1)
        for raw_date in entry_dates:
            entry_date = pd.Timestamp(raw_date).normalize()
            index = date_index.get(entry_date)
            if index is None:
                continue
            for forward_date in normalized_dates[index + 1 : index + 1 + horizon]:
                regime = regime_by_date.get(forward_date)
                if regime not in counts:
                    continue
                counts[str(regime)] += 1
                total += 1
        if total <= 0:
            return {key: float("nan") for key in counts}
        return {key: value / total for key, value in counts.items()}

    def _regime_classifications_by_prediction_date(
        self,
        prediction_dates: list,
        *,
        horizon_sessions: int,
    ) -> dict[pd.Timestamp, str]:
        regime_loader = getattr(self.db_manager, "load_regime_meter", None)
        if not callable(regime_loader):
            return {}
        try:
            regime = regime_loader()
        except Exception as exc:
            self.logger.warning("Unable to load regime meter for shortlist score flip: %s", exc)
            return {}
        if regime is None or regime.empty or not {"snapshot_date", "classification"}.issubset(regime.columns):
            return {}
        regime = regime.copy()
        regime["snapshot_date"] = pd.to_datetime(regime["snapshot_date"], errors="coerce").dt.normalize()
        regime = regime.dropna(subset=["snapshot_date"]).sort_values("snapshot_date").reset_index(drop=True)
        if regime.empty:
            return {}
        universe_dates = self._universe_snapshot_dates_for_regime_lookup(regime)
        if not universe_dates:
            return {}
        regime_dates = regime["snapshot_date"].tolist()
        regime_classes = regime["classification"].astype(str).tolist()
        horizon = max(int(horizon_sessions or 0), 0)
        output: dict[pd.Timestamp, str] = {}
        for raw_date in prediction_dates:
            prediction_date = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(prediction_date):
                continue
            prediction_date = prediction_date.normalize()
            source_index = bisect_right(universe_dates, prediction_date) - horizon - 1
            if source_index < 0:
                continue
            cutoff = universe_dates[source_index]
            regime_index = bisect_right(regime_dates, cutoff) - 1
            if regime_index < 0:
                continue
            output[prediction_date] = regime_classes[regime_index]
        return output

    def _universe_snapshot_dates_for_regime_lookup(self, regime: pd.DataFrame) -> list[pd.Timestamp]:
        loader = getattr(self.db_manager, "list_universe_daily_snapshot_dates", None)
        if callable(loader):
            try:
                dates = loader()
            except Exception as exc:
                self.logger.warning("Unable to load universe dates for regime score flip: %s", exc)
                dates = []
            parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
            if not parsed.empty:
                return sorted(parsed.dt.normalize().drop_duplicates().tolist())
        return sorted(regime["snapshot_date"].drop_duplicates().tolist())

    def _target_horizon_dates(self, target_column: str) -> int:
        for piece in str(target_column).split("_"):
            if piece.endswith("d") and piece[:-1].isdigit():
                return max(int(piece[:-1]), 1)
        return 1

    def _score_model(
        self,
        *,
        model_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        target_column: str,
        model_scope: str = "global",
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame | None:
        if train_frame.empty or test_frame.empty:
            return None
        if model_scope == "sector_specific" and model_name not in SHORTLIST_HEURISTIC_MODELS:
            return self._score_model_by_sector(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
        if model_scope == "regime_specific" and model_name not in SHORTLIST_HEURISTIC_MODELS:
            return self._score_model_by_regime(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "signal_proxy":
            return self._score_signal_proxy(test_frame)
        if model_name == "reversal_rules":
            return self._score_reversal_rules(test_frame)
        if model_name == "event_signal":
            return self._score_event_signal(test_frame)
        if model_name == "event_ic_model":
            return self._score_ic_sign_model(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=self._event_feature_columns_for_frame(train_frame),
            )
        if model_name == "structure_factor_signal":
            return self._score_structure_factor_signal(test_frame)
        if model_name == "structure_factor_event_signal":
            return self._score_structure_factor_event_signal(test_frame)
        if model_name == "ridge_model":
            return self._score_ridge_closed_form(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "lasso_model":
            return self._score_lasso_model(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "elastic_net_model":
            return self._score_elastic_net_model(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "ic_sign_model":
            return self._score_ic_sign_model(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "xgboost_model":
            return self._score_xgboost_model(
                train_frame,
                test_frame,
                target_column=target_column,
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
        raise ValueError(f"Unsupported model_name={model_name}")

    def _score_model_by_sector(
        self,
        *,
        model_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        target_column: str,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for sector, sector_test in test_frame.groupby("sector", sort=False):
            sector_train = train_frame[train_frame["sector"] == sector].copy()
            scoped_train = sector_train
            if len(sector_train.index) < 120 or sector_train["snapshot_date"].nunique() < 40:
                scoped_train = train_frame
            scored = self._score_model(
                model_name=model_name,
                train_frame=scoped_train,
                test_frame=sector_test.copy(),
                target_column=target_column,
                model_scope="global",
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
            if scored is not None and not scored.empty:
                frames.append(scored)
        if not frames:
            return None
        return pd.concat(frames, axis=0, ignore_index=True)

    def _classify_regime(self, frame: pd.DataFrame) -> pd.Series:
        if "spy_roc_20" in frame.columns and frame["spy_roc_20"].notna().any():
            spy_roc = pd.to_numeric(frame["spy_roc_20"], errors="coerce")
            regimes = pd.Series("choppy", index=frame.index)
            regimes[spy_roc > 0.01] = "trending"
            return regimes
        if "regime_green" in frame.columns:
            green = frame["regime_green"].astype(bool)
            return green.map({True: "trending", False: "choppy"})
        return pd.Series("choppy", index=frame.index)

    def _score_model_by_regime(
        self,
        *,
        model_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        target_column: str,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame | None:
        train_regimes = self._classify_regime(train_frame)
        frames: list[pd.DataFrame] = []
        for regime_label in ("trending", "choppy"):
            regime_train = train_frame[train_regimes == regime_label].copy()
            regime_test = test_frame[self._classify_regime(test_frame) == regime_label].copy()
            if regime_train.empty or regime_test.empty:
                continue
            if regime_train["snapshot_date"].nunique() < 40:
                regime_train = train_frame
            scored = self._score_model(
                model_name=model_name,
                train_frame=regime_train,
                test_frame=regime_test,
                target_column=target_column,
                model_scope="global",
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
            if scored is not None and not scored.empty:
                frames.append(scored)
        if not frames:
            return self._score_model(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                model_scope="global",
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    def _score_signal_proxy(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        components = [
            "relative_strength_index_vs_spy",
            "roc_63",
            "sma_200_dist",
            "vol_alpha",
        ]
        for component in components:
            values = pd.to_numeric(working[component], errors="coerce")
            if len(working["snapshot_date"].drop_duplicates()) > 1:
                working[f"{component}_rank"] = values.groupby(working["snapshot_date"]).rank(method="average", pct=True)
            else:
                working[f"{component}_rank"] = values.rank(method="average", pct=True)
        working["predicted_alpha"] = working[[f"{component}_rank" for component in components]].mean(axis=1, skipna=True)
        working["model_top_reasons"] = working.apply(
            lambda row: self._top_reason_names(
                {
                    component: row.get(f"{component}_rank")
                    for component in components
                }
            ),
            axis=1,
        )
        working["model_reason_summary"] = working["model_top_reasons"].apply(self._format_reason_summary)
        return working

    def _score_event_signal(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        for column in EVENT_DRIVEN_FEATURES:
            if column not in working.columns:
                working[column] = np.nan

        def rank_component(component: str, *, ascending: bool = True) -> pd.Series:
            values = pd.to_numeric(working[component], errors="coerce")
            if working["snapshot_date"].nunique() > 1:
                return values.groupby(working["snapshot_date"]).rank(method="average", pct=True, ascending=ascending)
            return values.rank(method="average", pct=True, ascending=ascending)

        for component in EVENT_ANALYST_FEATURES + EVENT_REACTION_FEATURES:
            working[f"{component}_event_rank"] = rank_component(component)

        days_since = pd.to_numeric(working["days_since_last_earnings"], errors="coerce")
        post_earnings_mask = (days_since >= 0.0) & (days_since <= 45.0)
        earnings_score = working[
            [f"{component}_event_rank" for component in EVENT_REACTION_FEATURES]
        ].mean(axis=1, skipna=True)
        earnings_score = earnings_score.where(post_earnings_mask, 0.5).fillna(0.5)

        analyst_score = working[
            [f"{component}_event_rank" for component in EVENT_ANALYST_FEATURES]
        ].mean(axis=1, skipna=True).fillna(0.5)

        days_to_next = pd.to_numeric(working["days_to_next_earnings"], errors="coerce")
        earnings_safety = pd.Series(
            np.where((days_to_next >= 0.0) & (days_to_next <= 10.0), 0.0, 1.0),
            index=working.index,
        )
        if working["snapshot_date"].nunique() > 1:
            earnings_safety = earnings_safety.groupby(working["snapshot_date"]).rank(method="average", pct=True)
        else:
            earnings_safety = earnings_safety.rank(method="average", pct=True)

        working["event_analyst_score"] = analyst_score
        working["event_earnings_score"] = earnings_score
        working["event_earnings_safety"] = earnings_safety.fillna(0.5)
        working["predicted_alpha"] = (
            (0.50 * working["event_analyst_score"])
            + (0.40 * working["event_earnings_score"])
            + (0.10 * working["event_earnings_safety"])
        )
        working["model_top_reasons"] = working.apply(
            lambda row: self._top_reason_names(
                {
                    "analyst_target_upside": row.get("analyst_target_upside_event_rank"),
                    "analyst_eps_revision_breadth": row.get("analyst_eps_revision_breadth_event_rank"),
                    "analyst_upgrade_downgrade_score": row.get("analyst_upgrade_downgrade_score_event_rank"),
                    "last_earnings_gap_pct": row.get("last_earnings_gap_pct_event_rank"),
                    "last_earnings_volume_ratio_20": row.get("last_earnings_volume_ratio_20_event_rank"),
                    "close_vs_last_earnings_close": row.get("close_vs_last_earnings_close_event_rank"),
                    "days_to_next_earnings": row.get("event_earnings_safety"),
                }
            ),
            axis=1,
        )
        working["model_reason_summary"] = working["model_top_reasons"].apply(self._format_reason_summary)
        return working

    def _score_structure_factor_signal(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        rank_columns: list[str] = []
        for component, direction in STRUCTURE_FACTOR_COMPONENTS:
            if component not in working.columns:
                working[component] = np.nan
            values = pd.to_numeric(working[component], errors="coerce")
            rank_column = f"{component}_structure_rank"
            ascending = bool(float(direction) > 0.0)
            if working["snapshot_date"].nunique() > 1:
                ranks = values.groupby(working["snapshot_date"]).rank(method="average", pct=True, ascending=ascending)
            else:
                ranks = values.rank(method="average", pct=True, ascending=ascending)
            working[rank_column] = ranks.fillna(0.5)
            rank_columns.append(rank_column)

        working["predicted_alpha"] = working[rank_columns].mean(axis=1, skipna=True).fillna(0.5)
        working["model_top_reasons"] = working.apply(
            lambda row: self._top_reason_names(
                {
                    component: row.get(f"{component}_structure_rank")
                    for component, _ in STRUCTURE_FACTOR_COMPONENTS
                }
            ),
            axis=1,
        )
        working["model_reason_summary"] = working["model_top_reasons"].apply(self._format_reason_summary)
        return working

    def _score_structure_factor_event_signal(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = self._score_structure_factor_signal(frame)
        event_mask = self._event_proximity_mask(scored)
        filtered = scored[event_mask].copy()
        if filtered.empty:
            return filtered
        filtered["event_condition_reason"] = self._event_condition_reason(filtered)
        return filtered

    def _event_proximity_mask(self, frame: pd.DataFrame) -> pd.Series:
        analyst_age = self._numeric_frame_column(frame, "analyst_snapshot_age_days")
        revision_age = self._numeric_frame_column(frame, "analyst_revision_snapshot_age_days")
        days_since_earnings = self._numeric_frame_column(frame, "days_since_last_earnings")
        days_to_earnings = self._numeric_frame_column(frame, "days_to_next_earnings")
        mask = (
            ((analyst_age >= 0.0) & (analyst_age <= 5.0))
            | ((revision_age >= 0.0) & (revision_age <= 5.0))
            | ((days_since_earnings >= 0.0) & (days_since_earnings <= 10.0))
            | ((days_to_earnings >= 0.0) & (days_to_earnings <= 10.0))
        )
        return mask.fillna(False)

    def _event_condition_reason(self, frame: pd.DataFrame) -> pd.Series:
        reasons: list[str] = []
        analyst_age = self._numeric_frame_column(frame, "analyst_snapshot_age_days")
        revision_age = self._numeric_frame_column(frame, "analyst_revision_snapshot_age_days")
        days_since_earnings = self._numeric_frame_column(frame, "days_since_last_earnings")
        days_to_earnings = self._numeric_frame_column(frame, "days_to_next_earnings")
        for index in frame.index:
            row_reasons: list[str] = []
            if pd.notna(analyst_age.loc[index]) and 0.0 <= float(analyst_age.loc[index]) <= 5.0:
                row_reasons.append("fresh analyst snapshot")
            if pd.notna(revision_age.loc[index]) and 0.0 <= float(revision_age.loc[index]) <= 5.0:
                row_reasons.append("fresh analyst revision")
            if pd.notna(days_since_earnings.loc[index]) and 0.0 <= float(days_since_earnings.loc[index]) <= 10.0:
                row_reasons.append("recent earnings")
            if pd.notna(days_to_earnings.loc[index]) and 0.0 <= float(days_to_earnings.loc[index]) <= 10.0:
                row_reasons.append("near earnings")
            reasons.append(", ".join(row_reasons) if row_reasons else "event proximity")
        return pd.Series(reasons, index=frame.index)

    def _numeric_frame_column(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    def _score_reversal_rules(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "regime_matching_test_regime" not in working.columns:
            working["regime_matching_test_regime"] = "unknown"
        momentum_components = [
            "relative_strength_index_vs_spy",
            "roc_63",
            "sma_200_dist",
            "vol_alpha",
        ]
        reversal_components = [
            "roc_63",
            "rsi_14",
            "close_vs_20d_low",
            "sma_50_dist",
        ]
        for component in set(momentum_components + reversal_components):
            if component not in working.columns:
                working[component] = np.nan

        def rank_component(component: str, *, ascending: bool) -> pd.Series:
            values = pd.to_numeric(working[component], errors="coerce")
            if working["snapshot_date"].nunique() > 1:
                return values.groupby(working["snapshot_date"]).rank(method="average", pct=True, ascending=ascending)
            return values.rank(method="average", pct=True, ascending=ascending)

        for component in momentum_components:
            working[f"momentum_{component}_rank"] = rank_component(component, ascending=True)
        for component in reversal_components:
            working[f"reversal_{component}_rank"] = rank_component(component, ascending=False)

        momentum_score = working[[f"momentum_{component}_rank" for component in momentum_components]].mean(axis=1, skipna=True)
        reversal_score = working[[f"reversal_{component}_rank" for component in reversal_components]].mean(axis=1, skipna=True)
        reversal_mask = working["regime_matching_test_regime"].astype(str).eq("reversal")
        working["predicted_alpha"] = momentum_score
        working.loc[reversal_mask, "predicted_alpha"] = reversal_score.loc[reversal_mask]
        working["model_top_reasons"] = working.apply(
            lambda row: self._top_reason_names(
                {
                    **{
                        component: row.get(f"momentum_{component}_rank")
                        for component in momentum_components
                    },
                    **(
                        {
                            f"reversal_{component}": row.get(f"reversal_{component}_rank")
                            for component in reversal_components
                        }
                        if str(row.get("regime_matching_test_regime")) == "reversal"
                        else {}
                    ),
                }
            ),
            axis=1,
        )
        working["model_reason_summary"] = working["model_top_reasons"].apply(self._format_reason_summary)
        return working

    def _score_lasso_model(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame:
        is_classification = str(target_column).endswith("_pos")
        train_matrix, test_matrix, feature_names, standardized_test = self._prepare_model_matrices(
            train_frame,
            test_frame,
            feature_columns_override=feature_columns_override,
        )
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(train_target)
        if not finite_mask.all():
            train_matrix = train_matrix[finite_mask]
            train_target = train_target[finite_mask]
        train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if is_classification:
            try:
                from sklearn.linear_model import LogisticRegression
            except ModuleNotFoundError:
                self.logger.warning("scikit-learn unavailable for classification; falling back.")
                return test_frame.assign(predicted_alpha=0.0)
            unique_classes = np.unique(train_target)
            if len(unique_classes) < 2:
                scored = test_frame.copy()
                scored["predicted_alpha"] = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
                scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
                scored["model_reason_summary"] = None
                return scored
            model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=500, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=l1.*", category=UserWarning)
                model.fit(train_matrix, train_target)
            nonzero = np.abs(model.coef_[0]) > 1e-8
            if not nonzero.all():
                dropped = [str(feature_names[i]) for i in range(len(feature_names)) if not nonzero[i]]
                self.logger.info("Purging %d zero-coef features: %s", len(dropped), ", ".join(dropped[:10]))
                model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=500, random_state=42)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
                    warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=l1.*", category=UserWarning)
                    model.fit(train_matrix[:, nonzero], train_target)
                full_weights = np.zeros(len(feature_names))
                full_weights[nonzero] = model.coef_[0]
                test_matrix_used = test_matrix[:, nonzero]
            else:
                full_weights = model.coef_[0]
                test_matrix_used = test_matrix
            weights = full_weights
            scored = test_frame.copy()
            scored["predicted_alpha"] = model.predict_proba(test_matrix_used)[:, 1]
            contribution_frame = standardized_test.mul(weights, axis=1)
        else:
            try:
                from sklearn.linear_model import Lasso
            except ModuleNotFoundError:
                self.logger.warning("scikit-learn unavailable; falling back to closed-form ridge.")
                return self._score_ridge_closed_form(
                    train_frame, test_frame,
                    target_column=target_column,
                    feature_columns_override=feature_columns_override,
                )
            model = Lasso(alpha=0.001, max_iter=2000, random_state=42, selection="cyclic")
            model.fit(train_matrix, train_target)
            weights = model.coef_
            nonzero = np.abs(weights) > 1e-8
            if not nonzero.all():
                dropped = [str(feature_names[i]) for i in range(len(feature_names)) if not nonzero[i]]
                self.logger.info("Purging %d zero-coef features: %s", len(dropped), ", ".join(dropped[:10]))
                model = Lasso(alpha=0.001, max_iter=2000, random_state=42, selection="cyclic")
                model.fit(train_matrix[:, nonzero], train_target)
                full_weights = np.zeros(len(feature_names))
                full_weights[nonzero] = model.coef_
                test_matrix_used = test_matrix[:, nonzero]
                weights = full_weights
            else:
                test_matrix_used = test_matrix
            scored = test_frame.copy()
            if not nonzero.all():
                scored["predicted_alpha"] = test_matrix_used @ model.coef_
            else:
                scored["predicted_alpha"] = test_matrix @ weights
            contribution_frame = standardized_test.mul(weights, axis=1)

        scored["model_top_reasons"] = [
            self._top_reason_names(contribution_frame.iloc[index].to_dict())
            for index in range(len(contribution_frame.index))
        ]
        scored["model_reason_summary"] = scored["model_top_reasons"].apply(self._format_reason_summary)
        return scored

    def _score_elastic_net_model(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            from sklearn.linear_model import ElasticNet
        except ModuleNotFoundError:
            self.logger.warning("scikit-learn unavailable; falling back to closed-form ridge.")
            return self._score_ridge_closed_form(
                train_frame,
                test_frame,
                target_column=target_column,
                feature_columns_override=feature_columns_override,
            )
        train_matrix, test_matrix, feature_names, standardized_test = self._prepare_model_matrices(
            train_frame,
            test_frame,
            feature_columns_override=feature_columns_override,
        )
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(train_target)
        if not finite_mask.all():
            train_matrix = train_matrix[finite_mask]
            train_target = train_target[finite_mask]
        train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        model = ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=10000, random_state=42, selection="cyclic")
        model.fit(train_matrix, train_target)
        weights = model.coef_
        scored = test_frame.copy()
        scored["predicted_alpha"] = test_matrix @ weights
        contribution_frame = standardized_test.mul(weights, axis=1)
        scored["model_top_reasons"] = [
            self._top_reason_names(contribution_frame.iloc[index].to_dict())
            for index in range(len(contribution_frame.index))
        ]
        scored["model_reason_summary"] = scored["model_top_reasons"].apply(self._format_reason_summary)
        return scored

    def _score_ic_sign_model(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame:
        train_matrix, test_matrix, feature_names, standardized_test = self._prepare_model_matrices(
            train_frame,
            test_frame,
            feature_columns_override=feature_columns_override,
        )
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        finite_target = np.isfinite(train_target)
        train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.zeros(len(feature_names), dtype=float)
        for index in range(len(feature_names)):
            values = train_matrix[:, index]
            valid = finite_target & np.isfinite(values)
            if int(valid.sum()) < 40 or np.nanstd(values[valid]) == 0.0:
                continue
            ic = pd.Series(values[valid]).corr(pd.Series(train_target[valid]), method="spearman")
            if pd.notna(ic) and math.isfinite(float(ic)) and abs(float(ic)) >= 0.01:
                weights[index] = float(ic)
        norm = float(np.sum(np.abs(weights)))
        if norm > 0.0:
            weights = weights / norm
        scored = test_frame.copy()
        scored["predicted_alpha"] = test_matrix @ weights
        contribution_frame = standardized_test.mul(weights, axis=1)
        scored["model_top_reasons"] = [
            self._top_reason_names(contribution_frame.iloc[index].to_dict())
            for index in range(len(contribution_frame.index))
        ]
        scored["model_reason_summary"] = scored["model_top_reasons"].apply(self._format_reason_summary)
        return scored

    def _score_ridge_closed_form(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame:
        train_matrix, test_matrix, feature_names, standardized_test = self._prepare_model_matrices(
            train_frame,
            test_frame,
            feature_columns_override=feature_columns_override,
        )
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(train_target)
        if not finite_mask.all():
            train_matrix = train_matrix[finite_mask]
            train_target = train_target[finite_mask]
        train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        ridge_penalty = 1.0
        xtx = train_matrix.T @ train_matrix
        identity = np.eye(xtx.shape[0], dtype=float)
        weights = np.linalg.solve(xtx + ridge_penalty * identity, train_matrix.T @ train_target)
        scored = test_frame.copy()
        scored["predicted_alpha"] = test_matrix @ weights
        contribution_frame = standardized_test.mul(weights, axis=1)
        scored["model_top_reasons"] = [
            self._top_reason_names(contribution_frame.iloc[index].to_dict())
            for index in range(len(contribution_frame.index))
        ]
        scored["model_reason_summary"] = scored["model_top_reasons"].apply(self._format_reason_summary)
        return scored

    def _score_xgboost_model(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame | None:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError:
            self.logger.warning("xgboost unavailable; skipping xgboost_model in shortlist model.")
            return None
        train_matrix, test_matrix, feature_names, _ = self._prepare_model_matrices(
            train_frame,
            test_frame,
            feature_columns_override=feature_columns_override,
        )
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "objective": "reg:squarederror",
        }
        if xgboost_params:
            params.update(xgboost_params)
        model = XGBRegressor(**params)
        model.fit(train_matrix, train_target, verbose=False)
        scored = test_frame.copy()
        scored["predicted_alpha"] = model.predict(test_matrix)
        try:
            from xgboost import DMatrix

            contribution_matrix = model.get_booster().predict(
                DMatrix(test_matrix, feature_names=feature_names),
                pred_contribs=True,
            )
            contribution_frame = pd.DataFrame(
                contribution_matrix[:, :-1],
                columns=feature_names,
            )
            scored["model_top_reasons"] = [
                self._top_reason_names(contribution_frame.iloc[index].to_dict())
                for index in range(len(contribution_frame.index))
            ]
        except Exception:
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
        scored["model_reason_summary"] = scored["model_top_reasons"].apply(self._format_reason_summary)
        return scored

    def _build_ensemble_predictions(self, predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        usable = {
            model_name: frame.copy()
            for model_name, frame in predictions_by_model.items()
            if frame is not None and not frame.empty
            and model_name not in SHORTLIST_ENSEMBLE_EXCLUDED_MODELS
        }
        if len(usable) < 2:
            return None
        base_model_name = next(iter(usable))
        merged = usable[base_model_name].copy()
        merged = merged.rename(
            columns={
                "predicted_alpha": f"{base_model_name}_predicted_alpha",
                "model_top_reasons": f"{base_model_name}_model_top_reasons",
            }
        )
        for model_name, frame in usable.items():
            if model_name == base_model_name:
                continue
            scoped = frame[["snapshot_date", "ticker", "predicted_alpha", "model_top_reasons"]].copy()
            scoped = scoped.rename(
                columns={
                    "predicted_alpha": f"{model_name}_predicted_alpha",
                    "model_top_reasons": f"{model_name}_model_top_reasons",
                }
            )
            merged = merged.merge(scoped, on=["snapshot_date", "ticker"], how="inner")
        if merged is None or merged.empty:
            return None
        rank_columns: list[str] = []
        for model_name in usable:
            source_column = f"{model_name}_predicted_alpha"
            rank_column = f"{model_name}_rank"
            if merged["snapshot_date"].nunique() > 1:
                merged[rank_column] = pd.to_numeric(merged[source_column], errors="coerce").groupby(
                    merged["snapshot_date"]
                ).rank(method="average", pct=True)
            else:
                merged[rank_column] = pd.to_numeric(merged[source_column], errors="coerce").rank(method="average", pct=True)
            rank_columns.append(rank_column)
        merged["predicted_alpha"] = merged[rank_columns].mean(axis=1, skipna=True)
        merged["model_top_reasons"] = merged.apply(
            lambda row: self._merge_reason_lists(
                [
                    self._ensure_reason_list(row.get(f"{model_name}_model_top_reasons"))
                    for model_name in usable
                ]
            ),
            axis=1,
        )
        merged["model_reason_summary"] = merged["model_top_reasons"].apply(self._format_reason_summary)
        return merged

    def _annotate_oos_artifact_ranks(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not {"snapshot_date", "predicted_alpha"}.issubset(frame.columns):
            return frame.copy()
        working = frame.copy()
        if "model_name" not in working.columns:
            working["model_name"] = "model"
        working["model_rank"] = (
            pd.to_numeric(working["predicted_alpha"], errors="coerce")
            .groupby([working["model_name"].astype(str), working["snapshot_date"]])
            .rank(method="first", ascending=False)
        )
        for model_name, index in working.groupby("model_name", sort=False).groups.items():
            rank_column = f"{model_name}_rank"
            working.loc[index, rank_column] = working.loc[index, "model_rank"]
        return working

    def _filter_model_feature_columns(self, feature_columns: list[str] | tuple[str, ...]) -> list[str]:
        return [
            str(feature_name)
            for feature_name in feature_columns
            if self._base_feature_name(str(feature_name)) not in SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES
        ]

    def _prepare_model_matrices(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        feature_columns_override: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
        available_columns = ["snapshot_date", "sector"] + [
            col
            for col in MODEL_FEATURE_COLUMNS
            if col in train_frame.columns and col not in SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES
        ]
        feature_frame = pd.concat(
            [
                train_frame[available_columns].copy(),
                test_frame[available_columns].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        for col in MODEL_FEATURE_COLUMNS:
            if col not in feature_frame.columns:
                feature_frame[col] = np.nan
        feature_frame, feature_columns = build_rank_augmented_feature_frame(feature_frame)
        feature_columns = self._filter_model_feature_columns(feature_columns)
        if feature_columns_override is not None:
            feature_columns = [
                column
                for column in self._filter_model_feature_columns(feature_columns_override)
                if column in feature_frame.columns
            ]
            if not feature_columns:
                feature_columns = self._filter_model_feature_columns(expand_model_feature_columns(MODEL_FEATURE_COLUMNS))
        feature_frame = feature_frame[feature_columns + ["sector"]].copy()
        feature_frame = pd.get_dummies(feature_frame, columns=["sector"], dummy_na=False)
        train_features = feature_frame.iloc[: len(train_frame.index)].copy()
        test_features = feature_frame.iloc[len(train_frame.index) :].copy()
        train_medians = train_features.median(numeric_only=True)
        train_features = train_features.fillna(train_medians)
        test_features = test_features.fillna(train_medians)
        means = train_features.mean(axis=0)
        stds = train_features.std(axis=0).replace(0.0, 1.0)
        standardized_train = (train_features - means) / stds
        standardized_test = (test_features - means) / stds
        train_matrix = standardized_train.to_numpy(dtype=float)
        test_matrix = standardized_test.to_numpy(dtype=float)
        return train_matrix, test_matrix, list(train_features.columns), standardized_test

    def _event_feature_columns_for_frame(self, frame: pd.DataFrame) -> list[str]:
        available = [
            column
            for column in EVENT_DRIVEN_FEATURES
            if column in frame.columns and column not in SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES
        ]
        return expand_model_feature_columns(available)

    def _load_min_feature_ic(self) -> float:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            if isinstance(config, dict)
            else {}
        )
        return float(payload.get("min_feature_ic", 0.03))

    def _load_min_feature_ic_observation_fraction(self) -> float:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            if isinstance(config, dict)
            else {}
        )
        return max(0.0, min(float(payload.get("min_feature_ic_observation_fraction", 0.20)), 1.0))

    def _load_regime_matching_mode(self) -> str:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            if isinstance(config, dict)
            else {}
        )
        enabled = payload.get("regime_matching_enabled")
        if enabled is not None and not bool(enabled):
            return "off"
        if "regime_matching" not in payload:
            self.logger.warning(
                "scan_policy.shortlist_model.regime_matching missing; using train_and_flip."
            )
        return self._normalize_regime_matching_mode(payload.get("regime_matching", "train_and_flip"))

    def _load_min_regime_train_dates(self) -> int:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            if isinstance(config, dict)
            else {}
        )
        return max(2, int(payload.get("min_regime_train_dates", 120)))

    def _load_regime_transition_purge_mode(self) -> str:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            if isinstance(config, dict)
            else {}
        )
        return self._normalize_regime_transition_purge_mode(payload.get("regime_transition_purge", "off"))

    def _normalize_regime_matching_mode(self, mode: object) -> str:
        normalized = str(mode or "off").strip().lower()
        if normalized not in REGIME_MATCHING_MODES:
            self.logger.warning(
                "Unknown shortlist_model.regime_matching=%r; using train_and_flip.",
                mode,
            )
            return "train_and_flip"
        return normalized

    def _normalize_regime_transition_purge_mode(self, mode: object) -> str:
        normalized = str(mode or "off").strip().lower()
        if normalized not in REGIME_TRANSITION_PURGE_MODES:
            self.logger.warning(
                "Unknown shortlist_model.regime_transition_purge=%r; using off.",
                mode,
            )
            return "off"
        return normalized

    def _resolve_max_train_dates(self, *, max_train_dates: int | None, min_train_dates: int) -> int | None:
        if max_train_dates is None:
            config = load_feature_config()
            payload = (
                config.get("scan_policy", {})
                .get("shortlist_model", {})
                if isinstance(config, dict)
                else {}
            )
            configured = payload.get("max_train_dates")
            if configured in (None, ""):
                return None
            max_train_dates = int(configured)
        resolved = max(int(max_train_dates), int(min_train_dates))
        return resolved

    def _feature_ic_survivors_from_frame(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        min_feature_ic: float,
        min_observation_fraction: float = 0.20,
    ) -> list[str]:
        if frame.empty or target_column not in frame.columns:
            return []
        available_columns = ["snapshot_date", "sector"] + [
            col
            for col in MODEL_FEATURE_COLUMNS
            if col in frame.columns and col not in SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES
        ]
        feature_frame = frame[available_columns].copy()
        for col in MODEL_FEATURE_COLUMNS:
            if col not in feature_frame.columns:
                feature_frame[col] = np.nan
        feature_frame, feature_columns = build_rank_augmented_feature_frame(feature_frame)
        feature_columns = self._filter_model_feature_columns(feature_columns)
        target = pd.to_numeric(frame[target_column], errors="coerce")
        survivors: list[str] = []
        survivor_signatures: set[tuple[tuple[int, float | None], ...]] = set()
        min_observations = self._feature_ic_min_observations(
            len(frame.index),
            min_observation_fraction=min_observation_fraction,
        )
        for feature_name in feature_columns:
            values = pd.to_numeric(feature_frame[feature_name], errors="coerce")
            valid = values.notna() & target.notna()
            if int(valid.sum()) < min_observations:
                continue
            base_feature_name = self._base_feature_name(feature_name)
            if not self._has_cross_sectional_variation(
                feature_frame=feature_frame,
                feature_name=base_feature_name,
                valid=valid,
            ):
                continue
            if values[valid].nunique(dropna=True) < 2 or target[valid].nunique(dropna=True) < 2:
                continue
            ic = values[valid].corr(target[valid], method="spearman")
            if pd.notna(ic) and math.isfinite(float(ic)) and abs(float(ic)) >= float(min_feature_ic):
                signature = self._feature_value_signature(values=values, valid=valid)
                if signature in survivor_signatures:
                    continue
                survivor_signatures.add(signature)
                survivors.append(str(feature_name))
        return survivors

    def _feature_ic_min_observations(self, row_count: int, *, min_observation_fraction: float) -> int:
        if row_count <= 0:
            return 0
        fraction = max(0.0, min(float(min_observation_fraction), 1.0))
        return max(2, int(math.ceil(float(row_count) * fraction)))

    def _base_feature_name(self, feature_name: str) -> str:
        for suffix in ("__rank_all", "__rank_sector"):
            if str(feature_name).endswith(suffix):
                return str(feature_name)[: -len(suffix)]
        return str(feature_name)

    def _has_cross_sectional_variation(
        self,
        *,
        feature_frame: pd.DataFrame,
        feature_name: str,
        valid: pd.Series,
    ) -> bool:
        if feature_name not in feature_frame.columns:
            return False
        base_values = pd.to_numeric(feature_frame[feature_name], errors="coerce")
        base_valid = valid & base_values.notna()
        if int(base_valid.sum()) < 2:
            return False
        varied_by_date = base_values[base_valid].groupby(feature_frame.loc[base_valid, "snapshot_date"]).nunique(dropna=True)
        min_varied_dates = min(5, max(1, int(varied_by_date.index.nunique() // 4)))
        return int((varied_by_date > 1).sum()) >= min_varied_dates

    def _feature_value_signature(self, *, values: pd.Series, valid: pd.Series) -> tuple[tuple[int, float | None], ...]:
        payload: list[tuple[int, float | None]] = []
        numeric = pd.to_numeric(values, errors="coerce")
        for index, value in numeric[valid].items():
            if pd.isna(value) or not math.isfinite(float(value)):
                payload.append((int(index), None))
            else:
                payload.append((int(index), float(value)))
        return tuple(payload)

    def _record_regime_feature_survivors(
        self,
        stats: dict[str, dict[str, object]] | None,
        *,
        regime: str | None,
        survivors: list[str],
    ) -> None:
        if stats is None:
            return
        key = str(regime or "unknown")
        bucket = stats.setdefault(key, {"folds": 0, "features": {}})
        bucket["folds"] = int(bucket.get("folds", 0)) + 1
        feature_counts = bucket.setdefault("features", {})
        if not isinstance(feature_counts, dict):
            feature_counts = {}
            bucket["features"] = feature_counts
        for feature in survivors:
            feature_name = str(feature)
            feature_counts[feature_name] = int(feature_counts.get(feature_name, 0)) + 1

    def _render_regime_feature_survivors(self, stats: dict[str, dict[str, object]] | None) -> list[str]:
        lines = ["## Surviving Features By Fold Regime", ""]
        if not stats:
            lines.extend(["No fold-local regime feature survivor data available.", ""])
            return lines
        for regime in ("reversal", "trending", "neutral", "unknown"):
            bucket = stats.get(regime)
            if not bucket:
                continue
            feature_counts = bucket.get("features", {})
            if not isinstance(feature_counts, dict):
                feature_counts = {}
            ordered = sorted(feature_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
            feature_list = ", ".join(f"{feature} ({count})" for feature, count in ordered[:20]) or "none"
            core_hits = sorted(
                {
                    self._base_feature_name(feature)
                    for feature, _count in ordered
                    if self._base_feature_name(feature) in REVERSAL_CORE_FEATURES
                }
            )
            lines.append(f"### {regime}")
            lines.append(f"- folds_screened: {int(bucket.get('folds', 0))}")
            lines.append(f"- unique_surviving_features: {len(feature_counts)}")
            lines.append(f"- top_surviving_features: {feature_list}")
            if regime == "reversal":
                lines.append(f"- reversal_core_survivors: {', '.join(core_hits) if core_hits else 'none'}")
            lines.append("")
        return lines

    def _decommission_active_champions(
        self,
        *,
        generated_at: str,
        horizon_days: int,
        eligible_universe_mode: str,
        model_scope: str,
        xgboost_config: str,
        feature_profile: str,
        reason: str,
    ) -> int:
        decommissioner = getattr(self.db_manager, "decommission_shortlist_model_runs", None)
        if decommissioner is None:
            return 0
        try:
            return int(
                decommissioner(
                    horizon_days=int(horizon_days),
                    eligible_universe_mode=eligible_universe_mode,
                    model_scope=model_scope,
                    xgboost_config=xgboost_config,
                    feature_profile=feature_profile,
                    reason=reason,
                    decommissioned_at=generated_at,
                )
            )
        except TypeError:
            return 0

    def _days_since_last_champion(self, *, generated_at: str, horizon_days: int) -> int | None:
        loader = getattr(self.db_manager, "load_shortlist_model_runs", None)
        if loader is None:
            return None
        try:
            runs = loader(horizon_days=int(horizon_days), limit=1)
        except TypeError:
            try:
                runs = loader(horizon_days=int(horizon_days), eligible_universe_mode=None, model_scope=None, limit=1)
            except Exception:
                return None
        except Exception:
            return None
        if runs is None or runs.empty or "generated_at" not in runs.columns:
            return None
        last_generated = pd.to_datetime(runs.iloc[0].get("generated_at"), errors="coerce", utc=True)
        current_generated = pd.to_datetime(generated_at, errors="coerce", utc=True)
        if pd.isna(last_generated) or pd.isna(current_generated):
            return None
        return max(int((current_generated.date() - last_generated.date()).days), 0)

    def _feature_ic_report(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        min_train_dates: int,
        test_window_dates: int,
        evaluation_stride_dates: int,
        label_horizon_dates: int,
        min_feature_ic: float,
        min_observation_fraction: float = 0.20,
        feature_columns_override: list[str] | None = None,
    ) -> dict[str, object]:
        dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
        oos_dates: list = []
        start_index = int(min_train_dates)
        stride = max(int(evaluation_stride_dates or test_window_dates), 1)
        label_embargo = max(int(label_horizon_dates or 0), 0)
        while start_index < len(dates):
            test_dates = dates[start_index : start_index + max(int(test_window_dates), 1)]
            train_end_index = max(0, start_index - label_embargo)
            if len(dates[:train_end_index]) >= int(min_train_dates):
                oos_dates.extend(test_dates)
            start_index += stride

        report_path = self.db_manager.paths.reports_dir / "feature_ic_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not oos_dates:
            report_path.write_text(
                "\n".join(
                    [
                        "# Feature IC Report",
                        "",
                        f"- target_column: {target_column}",
                        f"- min_feature_ic: {float(min_feature_ic):.4f}",
                        f"- min_feature_ic_observation_fraction: {float(min_observation_fraction):.4f}",
                        "- oos_dates: 0",
                        "- surviving_features: 0",
                        "",
                        "No embargoed OOS feature rows were available.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            return {"surviving_features": [], "report": pd.DataFrame()}

        oos_frame = frame[frame["snapshot_date"].isin(oos_dates)].copy()
        available_columns = ["snapshot_date", "sector"] + [
            col
            for col in MODEL_FEATURE_COLUMNS
            if col in oos_frame.columns and col not in SHORTLIST_MODEL_EXCLUDED_BASE_FEATURES
        ]
        feature_frame = oos_frame[available_columns].copy()
        for col in MODEL_FEATURE_COLUMNS:
            if col not in feature_frame.columns:
                feature_frame[col] = np.nan
        feature_frame, feature_columns = build_rank_augmented_feature_frame(feature_frame)
        feature_columns = self._filter_model_feature_columns(feature_columns)
        if feature_columns_override is not None:
            feature_columns = [
                column
                for column in self._filter_model_feature_columns(feature_columns_override)
                if column in feature_frame.columns
            ]
        target = pd.to_numeric(oos_frame[target_column], errors="coerce")
        min_observations = self._feature_ic_min_observations(
            len(oos_frame.index),
            min_observation_fraction=min_observation_fraction,
        )
        rows: list[dict[str, object]] = []
        survivor_signatures: dict[tuple[tuple[int, float | None], ...], str] = {}
        for feature_name in feature_columns:
            values = pd.to_numeric(feature_frame[feature_name], errors="coerce")
            valid = values.notna() & target.notna()
            has_cross_sectional_variation = self._has_cross_sectional_variation(
                feature_frame=feature_frame,
                feature_name=self._base_feature_name(feature_name),
                valid=valid,
            )
            if (
                int(valid.sum()) < min_observations
                or not has_cross_sectional_variation
                or values[valid].nunique(dropna=True) < 2
                or target[valid].nunique(dropna=True) < 2
            ):
                ic = float("nan")
            else:
                ic = float(values[valid].corr(target[valid], method="spearman"))
            duplicate_of = None
            abs_ic = abs(ic) if math.isfinite(ic) else float("nan")
            if pd.notna(abs_ic) and float(abs_ic) >= float(min_feature_ic):
                signature = self._feature_value_signature(values=values, valid=valid)
                duplicate_of = survivor_signatures.get(signature)
                if duplicate_of is None:
                    survivor_signatures[signature] = str(feature_name)
            rows.append(
                {
                    "feature": feature_name,
                    "rank_ic": ic,
                    "abs_rank_ic": abs_ic,
                    "observations": int(valid.sum()),
                    "duplicate_of": duplicate_of,
                }
            )
        report = pd.DataFrame(rows)
        report = report.sort_values(
            ["abs_rank_ic", "feature"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        surviving_features = [
            str(row.feature)
            for row in report.itertuples(index=False)
            if pd.notna(row.abs_rank_ic) and float(row.abs_rank_ic) >= float(min_feature_ic)
            and int(row.observations) >= min_observations
            and pd.isna(row.duplicate_of)
        ]
        lines = [
            "# Feature IC Report",
            "",
            f"- target_column: {target_column}",
            f"- min_feature_ic: {float(min_feature_ic):.4f}",
            f"- min_feature_ic_observation_fraction: {float(min_observation_fraction):.4f}",
            f"- min_feature_ic_observations: {int(min_observations)}",
            f"- oos_dates: {len(set(oos_dates))}",
            f"- oos_rows: {len(oos_frame.index)}",
            f"- surviving_features: {len(surviving_features)}",
            f"- surviving_feature_names: {', '.join(surviving_features) if surviving_features else 'none'}",
            "",
            "| feature | rank_ic | abs_rank_ic | observations | duplicate_of | survives |",
            "|---|---:|---:|---:|---|---|",
        ]
        for row in report.itertuples(index=False):
            abs_ic = float(row.abs_rank_ic) if pd.notna(row.abs_rank_ic) else float("nan")
            survives = (abs_ic >= float(min_feature_ic) and pd.isna(row.duplicate_of)) if math.isfinite(abs_ic) else False
            survives = survives and int(row.observations) >= min_observations
            lines.append(
                "| "
                f"{row.feature} | "
                f"{self._fmt(row.rank_ic)} | "
                f"{self._fmt(row.abs_rank_ic)} | "
                f"{int(row.observations)} | "
                f"{row.duplicate_of if pd.notna(row.duplicate_of) else ''} | "
                f"{str(survives).lower()} |"
            )
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(
            "Feature IC screen kept %d/%d features at min_feature_ic=%.4f",
            len(surviving_features),
            len(feature_columns),
            float(min_feature_ic),
        )
        return {"surviving_features": surviving_features, "report": report}

    def _annotate_calibrated_probabilities(self, frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
        if frame.empty or "predicted_alpha" not in frame.columns or target_column not in frame.columns:
            return frame.copy()
        working = frame.copy()
        calibrated = pd.Series(np.nan, index=working.index, dtype=float)
        for _model_name, model_frame in working.groupby("model_name", sort=False) if "model_name" in working.columns else [(None, working)]:
            scores = pd.to_numeric(model_frame["predicted_alpha"], errors="coerce")
            actual = pd.to_numeric(model_frame[target_column], errors="coerce")
            valid = scores.notna() & actual.notna()
            if valid.sum() < 20:
                calibrated.loc[model_frame.index] = np.nan
                continue
            calibrated.loc[model_frame.index] = self._calibrate_scores_to_probability(
                fit_scores=scores[valid],
                fit_actual=actual[valid],
                transform_scores=scores,
            )
        working["calibrated_p_beat_sector"] = calibrated
        return working

    def _apply_calibration_from_oos(
        self,
        live_frame: pd.DataFrame,
        oos_frame: pd.DataFrame,
        *,
        target_column: str,
    ) -> pd.DataFrame:
        working = live_frame.copy()
        if working.empty or oos_frame.empty:
            working["calibrated_p_beat_sector"] = np.nan
            return working
        fit_scores = pd.to_numeric(oos_frame.get("predicted_alpha"), errors="coerce")
        fit_actual = pd.to_numeric(oos_frame.get(target_column), errors="coerce")
        transform_scores = pd.to_numeric(working.get("predicted_alpha"), errors="coerce")
        valid = fit_scores.notna() & fit_actual.notna()
        if valid.sum() < 20:
            working["calibrated_p_beat_sector"] = np.nan
            return working
        working["calibrated_p_beat_sector"] = self._calibrate_scores_to_probability(
            fit_scores=fit_scores[valid],
            fit_actual=fit_actual[valid],
            transform_scores=transform_scores,
        )
        return working

    def _calibrate_scores_to_probability(
        self,
        *,
        fit_scores: pd.Series,
        fit_actual: pd.Series,
        transform_scores: pd.Series,
    ) -> np.ndarray:
        y = (pd.to_numeric(fit_actual, errors="coerce") > 0.0).astype(float).to_numpy(dtype=float)
        x = pd.to_numeric(fit_scores, errors="coerce").to_numpy(dtype=float)
        target_x = pd.to_numeric(transform_scores, errors="coerce").to_numpy(dtype=float)
        try:
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(x, y)
            return model.predict(target_x)
        except Exception:
            ordered = pd.DataFrame({"score": x, "actual": y}).sort_values("score").reset_index(drop=True)
            if ordered.empty:
                return np.full(len(target_x), np.nan, dtype=float)
            quantile_count = min(10, max(2, int(len(ordered.index) // 20)))
            ordered["bucket"] = pd.qcut(ordered.index, q=quantile_count, labels=False, duplicates="drop")
            bucket_stats = ordered.groupby("bucket", sort=True).agg(score=("score", "mean"), prob=("actual", "mean")).dropna()
            if bucket_stats.empty:
                return np.full(len(target_x), float(np.mean(y)) if len(y) else np.nan, dtype=float)
            return np.interp(
                target_x,
                bucket_stats["score"].to_numpy(dtype=float),
                bucket_stats["prob"].to_numpy(dtype=float),
                left=float(bucket_stats["prob"].iloc[0]),
                right=float(bucket_stats["prob"].iloc[-1]),
            )

    def _evaluate_predictions(
        self,
        *,
        predictions: pd.DataFrame,
        top_n: int,
        target_column: str,
        model_name: str,
    ) -> dict[str, object]:
        if predictions.empty:
            return self._empty_summary(model_name)
        cost_fraction = self._round_trip_cost_fraction()
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
            picks = ordered.head(int(top_n)).copy()
            target = pd.to_numeric(picks[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
            universe_target = pd.to_numeric(day_frame[target_column], errors="coerce").clip(lower=-1.0, upper=1.0).dropna()
            if target.empty or universe_target.empty:
                continue
            net_target = target - cost_fraction
            net_universe_target = universe_target - cost_fraction
            full_target = pd.to_numeric(ordered[target_column], errors="coerce")
            full_score = pd.to_numeric(ordered["predicted_alpha"], errors="coerce")
            spearman = float("nan")
            full_valid = full_target.notna() & full_score.notna()
            if int(full_valid.sum()) >= 3 and full_target[full_valid].nunique(dropna=True) > 1 and full_score[full_valid].nunique(dropna=True) > 1:
                corr = full_score[full_valid].corr(full_target[full_valid], method="spearman")
                if pd.notna(corr) and math.isfinite(float(corr)):
                    spearman = float(corr)
            rows.append(
                {
                    "date": pd.Timestamp(snapshot_date),
                    "pick_count": len(picks.index),
                    "gross_mean_target": float(target.mean()),
                    "mean_target": float(net_target.mean()),
                    "hit_rate": float((net_target > 0.0).mean()),
                    "universe_mean_target": float(net_universe_target.mean()),
                    "universe_hit_rate": float((net_universe_target > 0.0).mean()),
                    "spearman": spearman,
                }
            )
        if not rows:
            return self._empty_summary(model_name)
        frame = pd.DataFrame(rows)
        net_targets = pd.to_numeric(frame["mean_target"], errors="coerce").dropna()
        gross_targets = pd.to_numeric(frame["gross_mean_target"], errors="coerce").dropna()
        horizon = self._target_horizon_dates(target_column)
        concentration = self._top_ticker_concentration(
            predictions=predictions,
            target_column=target_column,
            top_n=top_n,
        )
        sharpe = annualized_sharpe(net_targets, periods_per_year=max(252.0 / float(horizon), 1.0))
        nw_t = newey_west_t_stat(net_targets, lag=horizon)
        return {
            "model": model_name,
            "dates": len(frame.index),
            "avg_pick_count": float(frame["pick_count"].mean()),
            "gross_mean_target": float(gross_targets.mean()) if not gross_targets.empty else float("nan"),
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "universe_mean_target": float(frame["universe_mean_target"].mean()),
            "universe_hit_rate": float(frame["universe_hit_rate"].mean()),
            "mean_target_excess": float((frame["mean_target"] - frame["universe_mean_target"]).mean()),
            "hit_rate_excess": float((frame["hit_rate"] - frame["universe_hit_rate"]).mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "spearman": float(frame["spearman"].dropna().mean()) if frame["spearman"].notna().any() else float("nan"),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
            "top_ticker": concentration["top_ticker"],
            "top_ticker_date_rate": concentration["top_ticker_date_rate"],
            "top_ticker_pick_share": concentration["top_ticker_pick_share"],
            "net_sharpe": sharpe,
            "newey_west_t": nw_t,
            "years_for_t_1_96": years_required_for_tstat(sharpe),
            "round_trip_cost": cost_fraction,
        }

    def _top_ticker_concentration(
        self,
        *,
        predictions: pd.DataFrame,
        target_column: str,
        top_n: int,
    ) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            ordered = day_frame.dropna(subset=[target_column]).sort_values(["predicted_alpha", "ticker"], ascending=[False, True])
            picks = ordered.head(int(top_n)).copy()
            if picks.empty:
                continue
            for ticker in picks["ticker"].astype(str).tolist():
                rows.append({"snapshot_date": pd.Timestamp(snapshot_date), "ticker": ticker})
        if not rows:
            return {"top_ticker": None, "top_ticker_date_rate": float("nan"), "top_ticker_pick_share": float("nan")}
        frame = pd.DataFrame(rows)
        ticker_pick_counts = frame["ticker"].value_counts()
        top_ticker = str(ticker_pick_counts.index[0])
        total_dates = max(int(frame["snapshot_date"].nunique()), 1)
        ticker_dates = int(frame.loc[frame["ticker"] == top_ticker, "snapshot_date"].nunique())
        return {
            "top_ticker": top_ticker,
            "top_ticker_date_rate": float(ticker_dates) / float(total_dates),
            "top_ticker_pick_share": float(ticker_pick_counts.iloc[0]) / float(len(frame.index)),
        }

    def _round_trip_cost_fraction(self) -> float:
        config = load_feature_config()
        payload = config.get("backtest_costs", {}) if isinstance(config, dict) else {}
        slippage = float(payload.get("slippage_bps_per_side", 0.0) or 0.0)
        commission = float(payload.get("commission_bps_per_side", 0.0) or 0.0)
        return ((slippage + commission) * 2.0) / 10_000.0

    def _rolling_window_summaries(
        self,
        *,
        predictions: pd.DataFrame,
        target_column: str,
        model_name: str,
        top_n: int,
        windows: tuple[int, ...],
        fold_windows: tuple[int, ...] = (),
        fold_size: int | None = None,
        include_full_oos: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        unique_dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())
        for window in windows:
            selected_dates = unique_dates[-min(int(window), len(unique_dates)) :]
            scoped = predictions[predictions["snapshot_date"].isin(selected_dates)].copy()
            summary = self._evaluate_predictions(
                predictions=scoped,
                top_n=top_n,
                target_column=target_column,
                model_name=f"{model_name}_{int(window)}d",
            )
            rows.append(summary)
        for fold_count in fold_windows:
            date_count = max(int(fold_count), 1) * max(int(fold_size or 1), 1)
            selected_dates = unique_dates[-min(date_count, len(unique_dates)) :]
            scoped = predictions[predictions["snapshot_date"].isin(selected_dates)].copy()
            summary = self._evaluate_predictions(
                predictions=scoped,
                top_n=top_n,
                target_column=target_column,
                model_name=self._fold_window_model_name(
                    model_name=model_name,
                    fold_count=int(fold_count),
                ),
            )
            rows.append(summary)
        if include_full_oos:
            rows.append(
                self._evaluate_predictions(
                    predictions=predictions,
                    top_n=top_n,
                    target_column=target_column,
                    model_name=f"{model_name}_full_oos",
                )
            )
        return pd.DataFrame(rows)

    def _fold_window_label(self, fold_count: int) -> str:
        folds = max(int(fold_count), 1)
        if folds == 1:
            return "last_fold"
        return f"trailing_{folds}folds"

    def _fold_window_model_name(self, *, model_name: str, fold_count: int) -> str:
        return f"{model_name}_{self._fold_window_label(fold_count)}"

    def _render_sector_contribution(
        self,
        *,
        predictions: pd.DataFrame,
        target_column: str,
        top_n: int,
        heading: str,
    ) -> list[str]:
        lines = [heading, ""]
        rows: list[dict[str, object]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            picks = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).head(int(top_n)).copy()
            if picks.empty:
                continue
            for sector, sector_frame in picks.groupby("sector", sort=True):
                target = pd.to_numeric(sector_frame[target_column], errors="coerce").dropna()
                if target.empty:
                    continue
                rows.append(
                    {
                        "snapshot_date": pd.Timestamp(snapshot_date),
                        "sector": sector,
                        "pick_count": len(sector_frame.index),
                        "mean_target": float(target.mean()),
                        "hit_rate": float((target > 0.0).mean()),
                    }
                )
        if not rows:
            lines.append("No sector contribution results available.")
            lines.append("")
            return lines
        frame = pd.DataFrame(rows)
        aggregated = (
            frame.groupby("sector", as_index=False)
            .agg(
                dates=("snapshot_date", "nunique"),
                avg_pick_count=("pick_count", "mean"),
                mean_target=("mean_target", "mean"),
                hit_rate=("hit_rate", "mean"),
            )
            .sort_values(["mean_target", "hit_rate", "sector"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        for row in aggregated.itertuples(index=False):
            lines.append(f"### {row.sector}")
            lines.append(f"- dates: {int(row.dates)}")
            lines.append(f"- avg_pick_count: {self._fmt(row.avg_pick_count)}")
            lines.append(f"- mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append("")
        return lines

    def _top_reason_names(self, contributions: dict[str, object], *, limit: int = 3) -> list[str]:
        ranked: list[tuple[float, str]] = []
        for feature_name, raw_value in contributions.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            ranked.append((value, self._humanize_model_reason(str(feature_name), value)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        output: list[str] = []
        for _, name in ranked:
            if name not in output:
                output.append(name)
            if len(output) >= int(limit):
                break
        return output

    def _merge_reason_lists(self, reason_lists: list[list[str]], *, limit: int = 3) -> list[str]:
        merged: list[str] = []
        for reasons in reason_lists:
            for reason in reasons:
                if reason not in merged:
                    merged.append(reason)
                if len(merged) >= int(limit):
                    return merged
        return merged

    def _ensure_reason_list(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _format_reason_summary(self, reasons: list[str]) -> str | None:
        clean = [str(reason) for reason in reasons if str(reason).strip()]
        if not clean:
            return None
        return ", ".join(clean[:3])

    def _humanize_model_reason(self, feature_name: str, value: float) -> str:
        base = str(feature_name)
        if base.endswith("__rank_all"):
            return self._rank_reason_phrase(base[:-10], value, scope="cross")
        if base.endswith("__rank_sector"):
            return self._rank_reason_phrase(base[:-13], value, scope="sector")

        labels = {
            "relative_strength_index_vs_spy": "strong RS vs SPY",
            "relative_strength_index_vs_qqq": "strong RS vs QQQ",
            "relative_strength_index_vs_xlk": "strong RS vs XLK",
            "relative_strength_index_vs_subindustry": "strong RS vs group ETF",
            "rs_vs_spy_5d_change": "improving RS vs SPY",
            "rs_vs_qqq_5d_change": "improving RS vs QQQ",
            "rs_vs_xlk_5d_change": "improving RS vs XLK",
            "rs_vs_subindustry_5d_change": "improving RS vs group ETF",
            "rs_vs_subindustry_10d_change": "improving 10d RS vs group ETF",
            "roc_63": "strong 63d momentum",
            "roc_126": "strong 126d momentum",
            "vol_alpha": "strong volume confirmation",
            "sma_200_dist": "well above 200d trend",
            "sma_50_dist": "well above 50d trend",
            "rsi_14": "healthy RSI 14",
            "atr_14": "constructive ATR profile",
            "days_to_next_earnings": "clear of near-term earnings",
            "days_since_last_earnings": "timely post-earnings setup",
            "last_earnings_gap_pct": "strong earnings gap",
            "last_earnings_volume_ratio_20": "strong earnings volume",
            "last_earnings_open_vs_20d_high": "earnings breakout open",
            "close_vs_last_earnings_close": "holding above earnings close",
            "avg_abs_gap_pct_20": "active price discovery",
            "max_gap_down_pct_60": "limited recent downside gap risk",
            "distance_above_20d_high": "holding above recent breakout",
            "base_range_pct_20": "tight recent base",
            "base_atr_contraction_20": "recent ATR contraction",
            "base_volume_dryup_ratio_20": "volume dry-up into setup",
            "breakout_volume_ratio_50": "breakout volume expansion",
            "sector_pct_above_50": "healthy 50d sector breadth",
            "sector_pct_above_200": "healthy 200d sector breadth",
            "sector_median_roc_63": "strong sector momentum backdrop",
            "analyst_target_upside": "strong analyst target upside",
            "analyst_target_range_pct": "wide analyst upside range",
            "analyst_count": "broad analyst coverage",
            "analyst_recommendation_score": "constructive analyst recommendation",
            "analyst_eps_revision_breadth": "positive analyst EPS revisions",
            "analyst_upgrade_downgrade_score": "positive analyst revision flow",
        }
        return labels.get(base, self._humanize_model_feature_name(base))

    def _rank_reason_phrase(self, feature_name: str, value: float, *, scope: str) -> str:
        label = self._humanize_model_feature_name(feature_name)
        if value >= 0.9:
            strength = "top-tier"
        elif value >= 0.75:
            strength = "strong"
        else:
            strength = "supportive"
        if scope == "sector":
            return f"{strength} within-sector {label.lower()}"
        return f"{strength} {label.lower()}"

    def _humanize_model_feature_name(self, feature_name: str) -> str:
        base = str(feature_name)
        suffix = ""
        if base.endswith("__rank_all"):
            base = base[:-10]
            suffix = " rank"
        elif base.endswith("__rank_sector"):
            base = base[:-13]
            suffix = " sector rank"
        labels = {
            "relative_strength_index_vs_spy": "RS vs SPY",
            "relative_strength_index_vs_qqq": "RS vs QQQ",
            "relative_strength_index_vs_xlk": "RS vs XLK",
            "relative_strength_index_vs_subindustry": "RS vs group ETF",
            "rs_vs_spy_5d_change": "5d RS change vs SPY",
            "rs_vs_qqq_5d_change": "5d RS change vs QQQ",
            "rs_vs_xlk_5d_change": "5d RS change vs XLK",
            "rs_vs_subindustry_5d_change": "5d RS change vs group ETF",
            "rs_vs_subindustry_10d_change": "10d RS change vs group ETF",
            "roc_63": "63d momentum",
            "roc_126": "126d momentum",
            "vol_alpha": "volume confirmation",
            "sma_200_dist": "distance above 200d",
            "sma_50_dist": "distance above 50d",
            "rsi_14": "RSI 14",
            "atr_14": "ATR 14",
            "days_to_next_earnings": "days to earnings",
            "days_since_last_earnings": "days since earnings",
            "last_earnings_gap_pct": "earnings gap",
            "last_earnings_volume_ratio_20": "earnings volume",
            "last_earnings_open_vs_20d_high": "earnings open vs 20d high",
            "close_vs_last_earnings_close": "post-earnings hold",
            "avg_abs_gap_pct_20": "avg gap",
            "max_gap_down_pct_60": "max gap down",
            "distance_above_20d_high": "distance above 20d high",
            "base_range_pct_20": "base tightness",
            "base_atr_contraction_20": "ATR contraction",
            "base_volume_dryup_ratio_20": "volume dry-up",
            "breakout_volume_ratio_50": "breakout volume",
            "sector_pct_above_50": "sector breadth 50d",
            "sector_pct_above_200": "sector breadth 200d",
            "sector_median_roc_63": "sector median momentum",
            "analyst_target_upside": "analyst target upside",
            "analyst_target_range_pct": "analyst target range",
            "analyst_count": "analyst coverage",
            "analyst_recommendation_score": "analyst recommendation",
            "analyst_eps_revision_breadth": "analyst EPS revisions",
            "analyst_upgrade_downgrade_score": "analyst revision flow",
        }
        return f"{labels.get(base, base.replace('_', ' '))}{suffix}"

    def _load_promotion_gate(self) -> dict[str, float | int]:
        config = load_feature_config()
        payload = (
            config.get("scan_policy", {})
            .get("shortlist_model", {})
            .get("promotion_gate", {})
            if isinstance(config, dict)
            else {}
        )
        return {
            "enabled": bool(payload.get("enabled", True)),
            "min_recent_20d_hit_rate_excess": float(payload.get("min_recent_20d_hit_rate_excess", 0.02)),
            "min_recent_20d_beat_universe_rate": float(payload.get("min_recent_20d_beat_universe_rate", 0.50)),
            "min_recent_20d_mean_target_excess": float(payload.get("min_recent_20d_mean_target_excess", 0.0)),
            "min_recent_60d_hit_rate_excess": float(payload.get("min_recent_60d_hit_rate_excess", 0.02)),
            "min_recent_60d_beat_universe_rate": float(payload.get("min_recent_60d_beat_universe_rate", 0.50)),
            "min_recent_60d_mean_target_excess": float(payload.get("min_recent_60d_mean_target_excess", 0.0)),
            "min_recent_1fold_hit_rate_excess": float(payload.get("min_recent_1fold_hit_rate_excess", 0.02)),
            "min_recent_1fold_beat_universe_rate": float(payload.get("min_recent_1fold_beat_universe_rate", 0.50)),
            "min_recent_1fold_mean_target_excess": float(payload.get("min_recent_1fold_mean_target_excess", 0.0)),
            "min_recent_3fold_hit_rate_excess": float(payload.get("min_recent_3fold_hit_rate_excess", 0.02)),
            "min_recent_3fold_beat_universe_rate": float(payload.get("min_recent_3fold_beat_universe_rate", 0.50)),
            "min_recent_3fold_mean_target_excess": float(payload.get("min_recent_3fold_mean_target_excess", 0.0)),
            "min_recent_20d_spearman": float(payload.get("min_recent_20d_spearman", 0.0)),
            "min_recent_60d_spearman": float(payload.get("min_recent_60d_spearman", 0.0)),
            "min_recent_1fold_spearman": float(payload.get("min_recent_1fold_spearman", 0.0)),
            "min_recent_3fold_spearman": float(payload.get("min_recent_3fold_spearman", 0.0)),
            "max_recent_20d_top_ticker_date_rate": float(payload.get("max_recent_20d_top_ticker_date_rate", 0.40)),
            "max_recent_60d_top_ticker_date_rate": float(payload.get("max_recent_60d_top_ticker_date_rate", 0.40)),
            "max_recent_1fold_top_ticker_date_rate": float(payload.get("max_recent_1fold_top_ticker_date_rate", 0.40)),
            "max_recent_3fold_top_ticker_date_rate": float(payload.get("max_recent_3fold_top_ticker_date_rate", 0.40)),
        }

    def _promotion_recent_windows(self, *, horizon_days: int) -> tuple[int, ...]:
        return ()

    def _promotion_fold_windows(self, *, horizon_days: int) -> tuple[int, ...]:
        horizon = max(int(horizon_days), 1)
        return (1, 3)

    def _choose_champion_model(
        self,
        *,
        full_summaries: pd.DataFrame,
        acceptance_summaries: pd.DataFrame,
        promotion_gate: dict[str, float | int],
        required_recent_windows: tuple[int, ...] = (),
        required_fold_windows: tuple[int, ...] = (1, 3),
    ) -> tuple[str, bool]:
        ranked = self._rank_model_summaries(full_summaries)
        if ranked.empty:
            raise ValueError("No shortlist model summaries are available for champion selection.")
        if not bool(promotion_gate.get("enabled", True)):
            return str(ranked.iloc[0]["model"]), True

        passing_models = {
            str(model)
            for model in ranked["model"].astype(str).tolist()
            if self._model_passes_promotion_gate(
                model_name=str(model),
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
                required_recent_windows=required_recent_windows,
                required_fold_windows=required_fold_windows,
            )
        }
        passing_models = {
            model
            for model in passing_models
            if self._model_passes_variant_guard(
                model_name=model,
                acceptance_summaries=acceptance_summaries,
                required_fold_windows=required_fold_windows,
            )
        }
        if passing_models:
            passing_ranked = ranked[ranked["model"].astype(str).isin(passing_models)].copy()
            return str(passing_ranked.iloc[0]["model"]), True
        raise ValueError(
            "No shortlist model candidate passed the promotion gate; refusing to persist a failing champion. "
            "Inspect the Recent Acceptance Windows in reports/shortlist_model.md or relax scan_policy.shortlist_model.promotion_gate explicitly."
        )

    def _rank_model_summaries(self, summaries: pd.DataFrame) -> pd.DataFrame:
        if summaries.empty:
            return summaries.copy()
        return summaries.sort_values(
            ["mean_target", "beat_universe_rate", "positive_date_rate", "model"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def _model_passes_promotion_gate(
        self,
        *,
        model_name: str,
        acceptance_summaries: pd.DataFrame,
        promotion_gate: dict[str, float | int],
        required_recent_windows: tuple[int, ...] = (),
        required_fold_windows: tuple[int, ...] = (1, 3),
    ) -> bool:
        if acceptance_summaries.empty:
            return False
        for window in required_recent_windows:
            row = acceptance_summaries[
                acceptance_summaries["model"].astype(str) == f"{model_name}_{window}d"
            ]
            if row.empty:
                return False
            summary = row.iloc[0]
            if not self._finite_at_least(summary.get("hit_rate_excess"), promotion_gate.get(f"min_recent_{window}d_hit_rate_excess", 0.02)):
                return False
            if not self._finite_at_least(
                summary.get("beat_universe_rate"),
                promotion_gate.get(f"min_recent_{window}d_beat_universe_rate", 0.50),
            ):
                return False
            if not self._finite_at_least(summary.get("mean_target_excess"), promotion_gate.get(f"min_recent_{window}d_mean_target_excess", 0.0)):
                return False
            if not self._finite_at_least(summary.get("spearman"), promotion_gate.get(f"min_recent_{window}d_spearman", 0.0)):
                return False
            if self._finite_above(summary.get("top_ticker_date_rate"), promotion_gate.get(f"max_recent_{window}d_top_ticker_date_rate", 0.40)):
                return False
        for folds in required_fold_windows:
            row = acceptance_summaries[
                acceptance_summaries["model"].astype(str) == self._fold_window_model_name(
                    model_name=model_name,
                    fold_count=int(folds),
                )
            ]
            if row.empty:
                return False
            summary = row.iloc[0]
            if not self._finite_at_least(summary.get("hit_rate_excess"), promotion_gate.get(f"min_recent_{folds}fold_hit_rate_excess", 0.02)):
                return False
            if not self._finite_at_least(
                summary.get("beat_universe_rate"),
                promotion_gate.get(f"min_recent_{folds}fold_beat_universe_rate", 0.50),
            ):
                return False
            if not self._finite_at_least(summary.get("mean_target_excess"), promotion_gate.get(f"min_recent_{folds}fold_mean_target_excess", 0.0)):
                return False
            if not self._finite_at_least(summary.get("spearman"), promotion_gate.get(f"min_recent_{folds}fold_spearman", 0.0)):
                return False
            if self._finite_above(summary.get("top_ticker_date_rate"), promotion_gate.get(f"max_recent_{folds}fold_top_ticker_date_rate", 0.40)):
                return False
        full_row = acceptance_summaries[
            acceptance_summaries["model"].astype(str) == f"{model_name}_full_oos"
        ]
        if full_row.empty:
            return False
        return True

    def _model_passes_variant_guard(
        self,
        *,
        model_name: str,
        acceptance_summaries: pd.DataFrame,
        required_fold_windows: tuple[int, ...],
    ) -> bool:
        if model_name != STRUCTURE_FACTOR_EVENT_MODEL:
            return True
        if acceptance_summaries.empty:
            return False
        required_names = [f"{STRUCTURE_FACTOR_BASE_MODEL}_full_oos"]
        required_names.extend(
            self._fold_window_model_name(
                model_name=STRUCTURE_FACTOR_BASE_MODEL,
                fold_count=int(folds),
            )
            for folds in required_fold_windows
        )
        for summary_name in required_names:
            row = acceptance_summaries[acceptance_summaries["model"].astype(str) == summary_name]
            if row.empty:
                return False
            summary = row.iloc[0]
            if not self._finite_at_least(summary.get("mean_target_excess"), 0.0):
                return False
            if not self._finite_at_least(summary.get("hit_rate_excess"), 0.0):
                return False
            if not self._finite_at_least(summary.get("spearman"), 0.0):
                return False
        return True

    def _finite_at_least(self, value, threshold) -> bool:
        try:
            numeric = float(value)
            required = float(threshold)
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric) and numeric >= required

    def _finite_above(self, value, threshold) -> bool:
        try:
            numeric = float(value)
            maximum = float(threshold)
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric) and numeric > maximum

    def _render_promotion_gate(
        self,
        *,
        promotion_gate: dict[str, float | int],
        summaries: pd.DataFrame,
        required_recent_windows: tuple[int, ...] = (),
        required_fold_windows: tuple[int, ...] = (1, 3),
    ) -> list[str]:
        lines = ["## Promotion Gate", ""]
        if not bool(promotion_gate.get("enabled", True)):
            lines.append("- enabled: false")
            lines.append("- note: model selection uses full walk-forward ranking only.")
            lines.append("")
            return lines
        recent_label = ", ".join(f"{int(window)}d" for window in required_recent_windows) or "none"
        fold_label = ", ".join(self._fold_window_label(int(folds)) for folds in required_fold_windows)
        lines.extend(
            [
                "- enabled: true",
                "- gate_note: floors are excess-over-universe by design; calibrated 2026-09-21 to the original fixed-20d stringency intent.",
                f"- active_recent_windows: {recent_label}",
                f"- active_fold_windows: {fold_label}",
                "- active_full_window: full_oos",
                f"- min_recent_1fold_hit_rate_excess: {float(promotion_gate['min_recent_1fold_hit_rate_excess']):.2f}",
                f"- min_recent_1fold_beat_universe_rate: {float(promotion_gate['min_recent_1fold_beat_universe_rate']):.2f}",
                f"- min_recent_1fold_mean_target_excess: {float(promotion_gate['min_recent_1fold_mean_target_excess']):.4f}",
                f"- min_recent_3fold_hit_rate_excess: {float(promotion_gate['min_recent_3fold_hit_rate_excess']):.2f}",
                f"- min_recent_3fold_beat_universe_rate: {float(promotion_gate['min_recent_3fold_beat_universe_rate']):.2f}",
                f"- min_recent_3fold_mean_target_excess: {float(promotion_gate['min_recent_3fold_mean_target_excess']):.4f}",
                f"- min_recent_1fold_spearman: {float(promotion_gate['min_recent_1fold_spearman']):.4f}",
                f"- min_recent_3fold_spearman: {float(promotion_gate['min_recent_3fold_spearman']):.4f}",
                f"- max_recent_1fold_top_ticker_date_rate: {float(promotion_gate.get('max_recent_1fold_top_ticker_date_rate', 0.40)):.2f}",
                f"- max_recent_3fold_top_ticker_date_rate: {float(promotion_gate.get('max_recent_3fold_top_ticker_date_rate', 0.40)):.2f}",
                "- gate_metric: per-date cross-sectional Spearman over the full OOS slice",
                "- target_winsorization: acceptance-window basket targets clipped to [-1.0, 1.0] before mean, hit, and beat calculations",
                "",
            ]
        )
        lines.extend(self._render_summary_table(summaries, heading="### Recent Acceptance Windows"))
        return lines

    def _render_regime_contamination_decomposition(
        self,
        *,
        predictions: pd.DataFrame,
        summaries: pd.DataFrame,
        horizon_sessions: int,
        fold_size: int,
        required_recent_windows: tuple[int, ...],
        required_fold_windows: tuple[int, ...],
    ) -> list[str]:
        lines = [
            "## Regime Contamination Decomposition",
            "",
            "- note: contamination decomposition is diagnostic only and is not consumed by the promotion gate.",
            "",
        ]
        if predictions.empty or summaries.empty or "snapshot_date" not in predictions.columns:
            lines.append("No prediction windows available.")
            lines.append("")
            return lines
        working = predictions.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.normalize()
        working = working.dropna(subset=["snapshot_date"])
        all_dates = sorted(working["snapshot_date"].drop_duplicates().tolist())
        if not all_dates:
            lines.append("No dated prediction windows available.")
            lines.append("")
            return lines
        report_calendar_dates = self._regime_report_calendar_dates(fallback_dates=all_dates)
        regime_by_date = self._regime_meter_classifications_by_date(report_calendar_dates)
        lines.extend(
            [
                "| window | entry_regime | fwd_neutral | fwd_trending | fwd_reversal | net_mean_target | hit_rate | beat_universe_rate |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summaries.sort_values("model").itertuples(index=False):
            window_name = str(row.model)
            window_dates = self._dates_for_acceptance_window_name(
                window_name=window_name,
                predictions=working,
                recent_windows=required_recent_windows,
                fold_windows=required_fold_windows,
                fold_size=fold_size,
            )
            entry_regime = self._majority_regime_for_dates(window_dates, regime_by_date) or "unknown"
            shares = self._forward_regime_share_for_dates(
                window_dates,
                all_dates=report_calendar_dates,
                horizon_sessions=max(int(horizon_sessions), 1),
                regime_by_date=regime_by_date,
            )
            lines.append(
                f"| {window_name} | {entry_regime} | "
                f"{self._fmt(shares.get('neutral', float('nan')))} | "
                f"{self._fmt(shares.get('trending', float('nan')))} | "
                f"{self._fmt(shares.get('reversal', float('nan')))} | "
                f"{self._fmt(getattr(row, 'mean_target', float('nan')))} | "
                f"{self._fmt(getattr(row, 'hit_rate', float('nan')))} | "
                f"{self._fmt(getattr(row, 'beat_universe_rate', float('nan')))} |"
            )
        lines.append("")
        return lines

    def _dates_for_acceptance_window_name(
        self,
        *,
        window_name: str,
        predictions: pd.DataFrame,
        recent_windows: tuple[int, ...],
        fold_windows: tuple[int, ...],
        fold_size: int,
    ) -> list[pd.Timestamp]:
        if predictions.empty:
            return []
        model_name = self._base_model_name_from_window(window_name)
        scoped = predictions[predictions["model_name"].astype(str).eq(model_name)].copy() if "model_name" in predictions.columns else predictions.copy()
        if scoped.empty:
            scoped = predictions.copy()
        dates = sorted(scoped["snapshot_date"].dropna().drop_duplicates().tolist())
        if window_name.endswith("_full_oos"):
            return dates
        for window in recent_windows:
            suffix = f"_recent_{int(window)}d"
            if window_name.endswith(suffix):
                return dates[-max(int(window), 1):]
        for folds in fold_windows:
            expected = self._fold_window_model_name(model_name=model_name, fold_count=int(folds))
            if window_name == expected:
                return dates[-max(int(folds) * max(int(fold_size), 1), 1):]
        return dates

    def _base_model_name_from_window(self, window_name: str) -> str:
        for suffix in ("_full_oos",):
            if window_name.endswith(suffix):
                return window_name[: -len(suffix)]
        for marker in ("_recent_", "_trailing_"):
            if marker in window_name:
                return window_name.split(marker, 1)[0]
        if window_name.endswith("_last_fold"):
            return window_name[: -len("_last_fold")]
        return window_name

    def _regime_meter_classifications_by_date(self, dates: list) -> dict[pd.Timestamp, str]:
        regime_loader = getattr(self.db_manager, "load_regime_meter", None)
        if not callable(regime_loader):
            return {}
        try:
            regime = regime_loader()
        except Exception as exc:
            self.logger.warning("Unable to load regime meter for contamination report: %s", exc)
            return {}
        if regime is None or regime.empty or not {"snapshot_date", "classification"}.issubset(regime.columns):
            return {}
        regime = regime.copy()
        regime["snapshot_date"] = pd.to_datetime(regime["snapshot_date"], errors="coerce").dt.normalize()
        regime = regime.dropna(subset=["snapshot_date"]).sort_values("snapshot_date").reset_index(drop=True)
        if regime.empty:
            return {}
        regime_dates = regime["snapshot_date"].tolist()
        regime_classes = regime["classification"].astype(str).tolist()
        output: dict[pd.Timestamp, str] = {}
        for raw_date in dates:
            date_value = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(date_value):
                continue
            date_value = date_value.normalize()
            regime_index = bisect_right(regime_dates, date_value) - 1
            if regime_index < 0:
                continue
            output[date_value] = regime_classes[regime_index]
        return output

    def _regime_report_calendar_dates(self, *, fallback_dates: list) -> list[pd.Timestamp]:
        loader = getattr(self.db_manager, "list_universe_daily_snapshot_dates", None)
        if callable(loader):
            try:
                dates = loader()
            except Exception as exc:
                self.logger.warning("Unable to load universe dates for contamination report: %s", exc)
                dates = []
            parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
            if not parsed.empty:
                return sorted(parsed.dt.normalize().drop_duplicates().tolist())
        return sorted(pd.to_datetime(pd.Series(list(fallback_dates)), errors="coerce").dropna().dt.normalize().drop_duplicates().tolist())

    def _empty_summary(self, model_name: str) -> dict[str, object]:
        return {
            "model": model_name,
            "dates": 0,
            "avg_pick_count": float("nan"),
            "gross_mean_target": float("nan"),
            "mean_target": float("nan"),
            "hit_rate": float("nan"),
            "universe_mean_target": float("nan"),
            "universe_hit_rate": float("nan"),
            "mean_target_excess": float("nan"),
            "hit_rate_excess": float("nan"),
            "beat_universe_rate": float("nan"),
            "spearman": float("nan"),
            "positive_date_rate": float("nan"),
            "ge_2pct_rate": float("nan"),
            "ge_5pct_rate": float("nan"),
            "net_sharpe": float("nan"),
            "newey_west_t": float("nan"),
            "years_for_t_1_96": float("nan"),
            "round_trip_cost": float("nan"),
        }

    def _build_report_lines(
        self,
        *,
        target_column: str,
        evaluation_target_column: str | None = None,
        top_n: int,
        eligible_universe_mode: str,
        model_scope: str,
        candidate_models: tuple[str, ...],
        selected_model: str,
        selected_model_gate_passed: bool,
        xgboost_config: str,
        feature_profile: str,
        min_train_dates: int,
        max_train_dates: int | None,
        test_window_dates: int,
        evaluation_stride_dates: int,
        label_horizon_dates: int,
        feature_ic_path,
        min_feature_ic: float,
        min_feature_ic_observation_fraction: float,
        regime_matching_mode: str,
        regime_transition_purge_mode: str,
        regime_transition_purge_counts: dict[str, int],
        regime_matching_stats: dict[str, int],
        regime_feature_stats: dict[str, dict[str, object]] | None = None,
        surviving_features: list[str],
        eligible_rows: int,
        eligible_dates: int,
        oos_prediction_dates: int,
        oos_path,
        live_path,
        generated_at: str,
        full_summaries: pd.DataFrame,
        recent_summaries: pd.DataFrame,
        recent_dates: int,
        promotion_gate: dict[str, float | int],
        required_recent_windows: tuple[int, ...] = (),
        required_fold_windows: tuple[int, ...] = (1, 3),
        acceptance_summaries: pd.DataFrame,
        oos_predictions: pd.DataFrame,
        failure_reason: str | None = None,
        days_since_last_champion: int | None = None,
        promotion_top_n: int = PROMOTION_BASKET_SIZE,
    ) -> list[str]:
        lines = [
            "# Shortlist Model",
            "",
            f"- target_column: {target_column}",
            f"- evaluation_target_column: {evaluation_target_column or target_column}",
            f"- live_output_top_n: {int(top_n)}",
            f"- promotion_top_n: {int(promotion_top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {model_scope}",
            f"- candidate_models: {', '.join(candidate_models)}",
            f"- selected_model: {selected_model}",
            f"- selected_model_gate_passed: {str(bool(selected_model_gate_passed)).lower()}",
            f"- xgboost_config: {xgboost_config}",
            f"- feature_profile: {feature_profile}",
            f"- min_train_dates: {int(min_train_dates)}",
            f"- max_train_dates: {int(max_train_dates) if max_train_dates is not None else 'none'}",
            f"- test_window_dates: {int(test_window_dates)}",
            f"- oos_evaluation_stride_dates: {int(evaluation_stride_dates)}",
            f"- label_horizon_dates: {int(label_horizon_dates)}",
            "- training_label_policy: horizon-strided non-overlapping dates after label embargo",
            "- feature_selection_policy: fold-local train-only rank IC screen; feature_ic_report is diagnostic",
            f"- regime_matching_mode: {regime_matching_mode}",
            (
                f"- regime_transition_purge: {regime_transition_purge_mode} "
                f"(would purge {int(regime_transition_purge_counts.get('majority', 0))} rows of "
                f"{int(regime_transition_purge_counts.get('rows', 0))} under majority, "
                f"{int(regime_transition_purge_counts.get('strict', 0))} under strict)"
            ),
            (
                "- regime_matching_folds: "
                f"attempted={int(regime_matching_stats.get('attempted_folds', 0))}, "
                f"matched={int(regime_matching_stats.get('matched_folds', 0))}, "
                f"fallback={int(regime_matching_stats.get('fallback_folds', 0))}, "
                f"unknown={int(regime_matching_stats.get('unknown_folds', 0))}, "
                f"live_matched={int(regime_matching_stats.get('live_matched', 0))}, "
                f"live_fallback={int(regime_matching_stats.get('live_fallback', 0))}"
            ),
            "- regime_flip_applied: in train_and_flip mode, predicted_alpha is negated only when same-regime training falls back and the lagged regime is reversal",
            "- objective: walk-forward cross-sectional ranking of the eligible universe on forward sector-relative alpha",
            f"- universe: {eligible_universe_mode_description(eligible_universe_mode)}",
            "- feature_matrix: raw features plus date-wise cross-sectional ranks and sector-relative ranks",
            f"- feature_ic_report: {feature_ic_path}",
            f"- min_feature_ic: {float(min_feature_ic):.4f}",
            f"- min_feature_ic_observation_fraction: {float(min_feature_ic_observation_fraction):.4f}",
            f"- surviving_features: {len(surviving_features)}",
            "",
            f"- eligible_rows: {int(eligible_rows)}",
            f"- eligible_dates: {int(eligible_dates)}",
            f"- oos_prediction_dates: {int(oos_prediction_dates)}",
            f"- champion_model: {selected_model}",
            f"- oos_predictions_csv: {oos_path}",
            f"- live_predictions_csv: {live_path}",
            f"- generated_at: {generated_at}",
            "",
        ]
        lines.extend(
            self._render_regime_matching(
                mode=regime_matching_mode,
                stats=regime_matching_stats,
            )
        )
        lines.extend(self._render_regime_feature_survivors(regime_feature_stats))
        if failure_reason:
            lines.extend(
                [
                    "## Promotion Failure",
                    "",
                    f"- reason: {failure_reason}",
                    "- action: no champion was persisted and live scan must fail closed.",
                    f"- days_since_last_champion: {days_since_last_champion if days_since_last_champion is not None else 'n/a'}",
                    "",
                ]
            )
        lines.extend(self._render_summary_table(full_summaries, heading="## Full Walk-Forward Evaluation"))
        lines.extend(self._render_summary_table(recent_summaries, heading=f"## Recent {int(recent_dates)} Walk-Forward Dates"))
        lines.extend(
            self._render_promotion_gate(
                promotion_gate=promotion_gate,
                summaries=acceptance_summaries,
                required_recent_windows=required_recent_windows,
                required_fold_windows=required_fold_windows,
            )
        )
        lines.extend(
            self._render_regime_contamination_decomposition(
                predictions=oos_predictions,
                summaries=acceptance_summaries,
                horizon_sessions=int(label_horizon_dates),
                fold_size=int(test_window_dates),
                required_recent_windows=required_recent_windows,
                required_fold_windows=required_fold_windows,
            )
        )
        return lines

    def _render_regime_matching(self, *, mode: str, stats: dict[str, int]) -> list[str]:
        attempted = int(stats.get("attempted_folds", 0))
        matched = int(stats.get("matched_folds", 0))
        fallback = int(stats.get("fallback_folds", 0))
        unknown = int(stats.get("unknown_folds", 0))
        live_matched = int(stats.get("live_matched", 0))
        live_fallback = int(stats.get("live_fallback", 0))
        denominator = max(attempted, 1)
        return [
            "## Regime Matching",
            "",
            f"- mode: {mode}",
            f"- attempted_folds: {attempted}",
            f"- matched_folds: {matched}",
            f"- fallback_folds: {fallback}",
            f"- unknown_folds: {unknown}",
            f"- matched_fold_rate: {matched / denominator:.6f}",
            f"- fallback_fold_rate: {fallback / denominator:.6f}",
            f"- unknown_fold_rate: {unknown / denominator:.6f}",
            f"- live_matched: {live_matched}",
            f"- live_fallback: {live_fallback}",
            "",
        ]

    def _render_summary_table(self, frame: pd.DataFrame, *, heading: str) -> list[str]:
        lines = [heading, ""]
        if frame.empty:
            lines.append("No model results available.")
            lines.append("")
            return lines
        ordered = frame.sort_values(
            ["mean_target", "beat_universe_rate", "model"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.model}")
            lines.append(f"- dates: {int(row.dates)}")
            lines.append(f"- avg_pick_count: {self._fmt(row.avg_pick_count)}")
            lines.append(f"- gross_mean_target: {self._fmt(getattr(row, 'gross_mean_target', float('nan')))}")
            lines.append(f"- net_mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- universe_mean_target: {self._fmt(getattr(row, 'universe_mean_target', float('nan')))}")
            lines.append(f"- mean_target_excess: {self._fmt(getattr(row, 'mean_target_excess', float('nan')))}")
            lines.append(f"- round_trip_cost: {self._fmt(getattr(row, 'round_trip_cost', float('nan')))}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append(f"- universe_hit_rate: {self._fmt(getattr(row, 'universe_hit_rate', float('nan')))}")
            lines.append(f"- hit_rate_excess: {self._fmt(getattr(row, 'hit_rate_excess', float('nan')))}")
            lines.append(f"- beat_universe_rate: {self._fmt(row.beat_universe_rate)}")
            lines.append(f"- spearman: {self._fmt(getattr(row, 'spearman', float('nan')))}")
            lines.append(f"- net_sharpe_ann: {self._fmt(getattr(row, 'net_sharpe', float('nan')))}")
            lines.append(f"- newey_west_t_lag_horizon: {self._fmt(getattr(row, 'newey_west_t', float('nan')))}")
            lines.append(f"- years_for_t_1_96: {self._fmt(getattr(row, 'years_for_t_1_96', float('nan')))}")
            lines.append(f"- positive_date_rate: {self._fmt(row.positive_date_rate)}")
            lines.append(f"- ge_2pct_rate: {self._fmt(row.ge_2pct_rate)}")
            lines.append(f"- ge_5pct_rate: {self._fmt(row.ge_5pct_rate)}")
            lines.append(f"- top_ticker: {getattr(row, 'top_ticker', None) or 'n/a'}")
            lines.append(f"- top_ticker_date_rate: {self._fmt(getattr(row, 'top_ticker_date_rate', float('nan')))}")
            lines.append(f"- top_ticker_pick_share: {self._fmt(getattr(row, 'top_ticker_pick_share', float('nan')))}")
            lines.append("")
        return lines

    def _render_live_candidates(self, *, champion_model: str, frame: pd.DataFrame) -> list[str]:
        lines = ["## Live Top Candidates", ""]
        lines.append(f"- champion_model: {champion_model}")
        lines.append(f"- snapshot_date: {frame['snapshot_date'].max().date() if not frame.empty else 'n/a'}")
        lines.append("")
        if frame.empty:
            lines.append("No live candidates.")
            lines.append("")
            return lines
        for row in frame.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- predicted_alpha: {float(row.predicted_alpha):.6f}")
            calibrated = getattr(row, "calibrated_p_beat_sector", None)
            if calibrated is not None and pd.notna(calibrated) and math.isfinite(float(calibrated)):
                lines.append(f"- calibrated_p_beat_sector: {float(calibrated):.2%}")
            model_reason_summary = getattr(row, "model_reason_summary", None)
            if model_reason_summary:
                lines.append(f"- why: {model_reason_summary}")
            lines.append(f"- md_volume_30d: {float(row.md_volume_30d):.0f}")
            lines.append(f"- chart: https://www.tradingview.com/chart/?symbol={row.ticker}")
            lines.append("")
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
