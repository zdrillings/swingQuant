from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd

from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS, expand_model_feature_columns
from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope


@dataclass(frozen=True)
class ShortlistTuneReport:
    output_path: str
    tuned_candidate: str
    ablation_count: int


class ShortlistTuneService(ShortlistModelService):
    FEATURE_GROUPS = {
        "earnings": {
            "days_to_next_earnings",
            "days_since_last_earnings",
            "last_earnings_gap_pct",
            "last_earnings_volume_ratio_20",
            "last_earnings_open_vs_20d_high",
            "close_vs_last_earnings_close",
        },
        "gap_risk": {
            "avg_abs_gap_pct_20",
            "max_gap_down_pct_60",
        },
        "breadth": {
            "sector_pct_above_50",
            "sector_pct_above_200",
            "sector_median_roc_63",
        },
        "trend_distance": {
            "sma_200_dist",
            "sma_50_dist",
            "distance_above_20d_high",
            "atr_14",
        },
    }

    def run(
        self,
        *,
        top_n: int = 10,
        horizon_days: int = 20,
        min_train_dates: int = 252,
        test_window_dates: int = 20,
        recent_dates: int = 60,
        eligible_universe_mode: str = "passed_or_trend",
        model_scope: str = "sector_specific",
        mode: str = "full",
        tuning_profile: str = "focused",
        ablation_profile: str = "focused",
        ablation_params_candidate: str | None = None,
    ) -> ShortlistTuneReport:
        print("Shortlist tune: initializing databases.", flush=True)
        self.db_manager.initialize()
        target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        mode = self._normalize_mode(mode)
        tuning_profile = self._normalize_profile(tuning_profile)
        ablation_profile = self._normalize_profile(ablation_profile)

        print("Shortlist tune: loading universe snapshots.", flush=True)
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")
        if target_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")

        print("Shortlist tune: preparing snapshot frame.", flush=True)
        all_snapshots = self._prepare_snapshot_frame(frame)
        print("Shortlist tune: building matured eligible universe.", flush=True)
        matured = self._build_matured_eligible_universe(
            all_snapshots,
            target_column=target_column,
            eligible_universe_mode=eligible_universe_mode,
        )
        if matured.empty:
            raise ValueError("No matured eligible universe rows found for shortlist tuning.")

        snapshot_count = matured["snapshot_date"].nunique()
        ticker_count = matured["ticker"].nunique()
        print(
            f"Shortlist tune: {snapshot_count} matured dates, "
            f"{ticker_count} tickers, universe={eligible_universe_mode}, scope={model_scope}.",
            flush=True,
        )

        full_feature_columns = expand_model_feature_columns(MODEL_FEATURE_COLUMNS)
        param_rows: list[dict[str, object]] = []
        param_frame = pd.DataFrame()
        tuned_candidate = ""
        tuned_params: dict[str, object] = {}
        tuned_row: pd.Series | None = None

        if mode in {"full", "tune_only"}:
            tuning_candidates = self._xgboost_tuning_candidates(tuning_profile)
            for idx, candidate in enumerate(tuning_candidates, start=1):
                print(
                    f"Shortlist tune: tuning candidate {idx}/{len(tuning_candidates)} "
                    f"({candidate['name']}).",
                    flush=True,
                )
                predictions = self._walk_forward_predictions(
                    matured,
                    target_column=target_column,
                    model_name="xgboost_model",
                    min_train_dates=int(min_train_dates),
                    test_window_dates=int(test_window_dates),
                    model_scope=model_scope,
                    xgboost_params=candidate["params"],
                    feature_columns_override=full_feature_columns,
                )
                if predictions is None or predictions.empty:
                    continue
                param_rows.append(
                    self._summarize_experiment(
                        experiment_name=str(candidate["name"]),
                        predictions=predictions,
                        target_column=target_column,
                        top_n=int(top_n),
                        recent_dates=int(recent_dates),
                        params=candidate["params"],
                        excluded_feature_group=None,
                    )
                )
            if not param_rows:
                raise ValueError("No xgboost tuning results available. Verify xgboost is installed.")
            param_frame = pd.DataFrame(param_rows)
            tuned_row = self._choose_best_experiment(param_frame)
            tuned_candidate = str(tuned_row["experiment"])
            tuned_params = json.loads(str(tuned_row["params_json"])) if tuned_row.get("params_json") else {}
            print(
                f"Shortlist tune: tuned winner is {tuned_candidate} "
                f"with full beat-universe {self._fmt(tuned_row['full_beat_universe_rate'])}.",
                flush=True,
            )
        else:
            tuned_candidate, tuned_params = self._resolve_named_candidate(ablation_params_candidate)
            print(
                f"Shortlist tune: ablation-only mode using params from {tuned_candidate}.",
                flush=True,
            )

        ablation_rows: list[dict[str, object]] = []
        ablation_frame = pd.DataFrame()
        if mode in {"full", "ablation_only"}:
            ablation_candidates = self._feature_ablation_candidates(ablation_profile)
            for idx, (ablation_name, base_features) in enumerate(ablation_candidates.items(), start=1):
                print(
                    f"Shortlist tune: ablation {idx}/{len(ablation_candidates)} "
                    f"({ablation_name}).",
                    flush=True,
                )
                predictions = self._walk_forward_predictions(
                    matured,
                    target_column=target_column,
                    model_name="xgboost_model",
                    min_train_dates=int(min_train_dates),
                    test_window_dates=int(test_window_dates),
                    model_scope=model_scope,
                    xgboost_params=tuned_params,
                    feature_columns_override=expand_model_feature_columns(base_features),
                )
                if predictions is None or predictions.empty:
                    continue
                ablation_rows.append(
                    self._summarize_experiment(
                        experiment_name=ablation_name,
                        predictions=predictions,
                        target_column=target_column,
                        top_n=int(top_n),
                        recent_dates=int(recent_dates),
                        params=tuned_params,
                        excluded_feature_group=None if ablation_name == "full_features" else ablation_name.replace("no_", ""),
                    )
                )
            if not ablation_rows:
                raise ValueError("No feature ablation results available.")
            ablation_frame = pd.DataFrame(ablation_rows)

        report_path = self.db_manager.paths.reports_dir / "shortlist_tune.md"
        lines = [
            "# Shortlist Tune",
            "",
            f"- target_column: {target_column}",
            f"- top_n: {int(top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {model_scope}",
            f"- mode: {mode}",
            f"- tuning_profile: {tuning_profile}",
            f"- ablation_profile: {ablation_profile}",
            f"- min_train_dates: {int(min_train_dates)}",
            f"- test_window_dates: {int(test_window_dates)}",
            f"- tuned_model_family: xgboost_model",
            "",
            "## Tuned Winner",
            "",
            f"- experiment: {tuned_candidate or 'n/a'}",
            f"- params: {json.dumps(tuned_params, sort_keys=True)}",
            f"- full_mean_target: {self._fmt(tuned_row['full_mean_target']) if tuned_row is not None else 'n/a'}",
            f"- full_beat_universe_rate: {self._fmt(tuned_row['full_beat_universe_rate']) if tuned_row is not None else 'n/a'}",
            f"- recent_mean_target: {self._fmt(tuned_row['recent_mean_target']) if tuned_row is not None else 'n/a'}",
            f"- recent_beat_universe_rate: {self._fmt(tuned_row['recent_beat_universe_rate']) if tuned_row is not None else 'n/a'}",
            "",
        ]
        if not param_frame.empty:
            lines.extend(self._render_experiment_table(param_frame, heading="## XGBoost Parameter Grid"))
        else:
            lines.extend(["## XGBoost Parameter Grid", "", "- skipped in ablation_only mode", ""])
        if not ablation_frame.empty:
            lines.extend(self._render_experiment_table(ablation_frame, heading="## Feature Ablation"))
        else:
            lines.extend(["## Feature Ablation", "", "- skipped in tune_only mode", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Shortlist tune: wrote {report_path}.", flush=True)
        return ShortlistTuneReport(
            output_path=str(report_path),
            tuned_candidate=tuned_candidate,
            ablation_count=len(ablation_rows),
        )

    def _xgboost_tuning_candidates(self, profile: str = "focused") -> list[dict[str, object]]:
        candidates = [
            {"name": "baseline", "params": {}},
            {
                "name": "shallower_regularized",
                "params": {
                    "max_depth": 3,
                    "min_child_weight": 3.0,
                    "reg_lambda": 2.0,
                },
            },
            {
                "name": "balanced_depth4",
                "params": {
                    "max_depth": 4,
                    "min_child_weight": 3.0,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                },
            },
            {
                "name": "deeper_conservative",
                "params": {
                    "max_depth": 5,
                    "min_child_weight": 5.0,
                    "learning_rate": 0.04,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "reg_lambda": 2.0,
                },
            },
            {
                "name": "faster_shallow",
                "params": {
                    "max_depth": 3,
                    "learning_rate": 0.07,
                    "subsample": 0.9,
                    "colsample_bytree": 0.8,
                },
            },
        ]
        if profile == "focused":
            keep = {"baseline", "shallower_regularized", "balanced_depth4"}
            return [candidate for candidate in candidates if candidate["name"] in keep]
        return candidates

    def _feature_ablation_candidates(self, profile: str = "focused") -> dict[str, list[str]]:
        candidates = {"full_features": list(MODEL_FEATURE_COLUMNS)}
        for group_name, group_features in self.FEATURE_GROUPS.items():
            candidates[f"no_{group_name}"] = [
                feature
                for feature in MODEL_FEATURE_COLUMNS
                if feature not in group_features
            ]
        if profile == "focused":
            keep = {"full_features", "no_earnings", "no_breadth"}
            return {name: features for name, features in candidates.items() if name in keep}
        return candidates

    def _resolve_named_candidate(self, candidate_name: str | None) -> tuple[str, dict[str, object]]:
        if not candidate_name:
            raise ValueError("ablation_only mode requires --ablation-params-candidate.")
        for candidate in self._xgboost_tuning_candidates(profile="full"):
            if str(candidate["name"]) == str(candidate_name):
                return str(candidate["name"]), dict(candidate["params"])
        valid = ", ".join(str(candidate["name"]) for candidate in self._xgboost_tuning_candidates(profile="full"))
        raise ValueError(f"Unknown ablation params candidate '{candidate_name}'. Valid choices: {valid}.")

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"full", "tune_only", "ablation_only"}:
            raise ValueError(f"Unsupported shortlist tune mode '{mode}'.")
        return normalized

    def _normalize_profile(self, profile: str) -> str:
        normalized = str(profile).strip().lower()
        if normalized not in {"focused", "full"}:
            raise ValueError(f"Unsupported shortlist tune profile '{profile}'.")
        return normalized

    def _summarize_experiment(
        self,
        *,
        experiment_name: str,
        predictions: pd.DataFrame,
        target_column: str,
        top_n: int,
        recent_dates: int,
        params: dict[str, object],
        excluded_feature_group: str | None,
    ) -> dict[str, object]:
        full_summary = self._evaluate_predictions(
            predictions=predictions,
            top_n=int(top_n),
            target_column=target_column,
            model_name=experiment_name,
        )
        recent_prediction_dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
        recent_predictions = predictions[predictions["snapshot_date"].isin(recent_prediction_dates)].copy()
        recent_summary = self._evaluate_predictions(
            predictions=recent_predictions,
            top_n=int(top_n),
            target_column=target_column,
            model_name=experiment_name,
        )
        return {
            "experiment": experiment_name,
            "params_json": json.dumps(params, sort_keys=True),
            "excluded_feature_group": excluded_feature_group,
            "full_mean_target": float(full_summary["mean_target"]),
            "full_beat_universe_rate": float(full_summary["beat_universe_rate"]),
            "full_hit_rate": float(full_summary["hit_rate"]),
            "recent_mean_target": float(recent_summary["mean_target"]),
            "recent_beat_universe_rate": float(recent_summary["beat_universe_rate"]),
            "recent_hit_rate": float(recent_summary["hit_rate"]),
        }

    def _choose_best_experiment(self, frame: pd.DataFrame) -> pd.Series:
        ordered = frame.sort_values(
            [
                "full_beat_universe_rate",
                "recent_beat_universe_rate",
                "full_mean_target",
                "recent_mean_target",
                "experiment",
            ],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        return ordered.iloc[0]

    def _render_experiment_table(self, frame: pd.DataFrame, *, heading: str) -> list[str]:
        lines = [heading, ""]
        ordered = frame.sort_values(
            [
                "full_beat_universe_rate",
                "recent_beat_universe_rate",
                "full_mean_target",
                "recent_mean_target",
                "experiment",
            ],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.experiment}")
            if row.params_json:
                lines.append(f"- params: {row.params_json}")
            if isinstance(row.excluded_feature_group, str) and row.excluded_feature_group:
                lines.append(f"- excluded_feature_group: {row.excluded_feature_group}")
            lines.append(f"- full_mean_target: {self._fmt(row.full_mean_target)}")
            lines.append(f"- full_beat_universe_rate: {self._fmt(row.full_beat_universe_rate)}")
            lines.append(f"- full_hit_rate: {self._fmt(row.full_hit_rate)}")
            lines.append(f"- recent_mean_target: {self._fmt(row.recent_mean_target)}")
            lines.append(f"- recent_beat_universe_rate: {self._fmt(row.recent_beat_universe_rate)}")
            lines.append(f"- recent_hit_rate: {self._fmt(row.recent_hit_rate)}")
            lines.append("")
        return lines

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
