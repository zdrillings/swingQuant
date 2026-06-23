from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math

import numpy as np
import pandas as pd

from src.research.shortlist_bakeoff_service import (
    MODEL_FEATURE_COLUMNS,
    build_rank_augmented_feature_frame,
    expand_model_feature_columns,
)
from src.research.shortlist_universe import (
    eligible_universe_mode_description,
    filter_eligible_universe,
    normalize_eligible_universe_mode,
    normalize_model_scope,
)
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


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
    ) -> ShortlistModelReport:
        self.db_manager.initialize()
        target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        xgboost_config = self._normalize_xgboost_config(xgboost_config)
        xgboost_params = self._xgboost_params_for_config(xgboost_config)
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")
        if target_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")

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

        model_predictions: dict[str, pd.DataFrame] = {}
        for model_name in ("signal_proxy", "ridge_model", "xgboost_model"):
            predicted = self._walk_forward_predictions(
                matured,
                target_column=target_column,
                model_name=model_name,
                min_train_dates=int(min_train_dates),
                test_window_dates=int(test_window_dates),
                model_scope=model_scope,
                xgboost_params=xgboost_params,
            )
            if predicted is not None and not predicted.empty:
                model_predictions[model_name] = predicted
        ensemble_predictions = self._build_ensemble_predictions(model_predictions)
        if ensemble_predictions is not None:
            model_predictions["ensemble_model"] = ensemble_predictions

        if not model_predictions:
            raise ValueError("No shortlist models produced out-of-sample predictions.")

        full_summaries = pd.DataFrame(
            [
                self._evaluate_predictions(
                    predictions=predictions,
                    top_n=int(top_n),
                    target_column=target_column,
                    model_name=model_name,
                )
                for model_name, predictions in model_predictions.items()
            ]
        )
        champion_model = self._choose_champion_model(full_summaries)

        recent_summary_rows: list[dict[str, object]] = []
        for model_name, predictions in model_predictions.items():
            recent_prediction_dates = sorted(predictions["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
            recent_predictions = predictions[predictions["snapshot_date"].isin(recent_prediction_dates)].copy()
            recent_summary_rows.append(
                self._evaluate_predictions(
                    predictions=recent_predictions,
                    top_n=int(top_n),
                    target_column=target_column,
                    model_name=model_name,
                )
            )
        recent_summaries = pd.DataFrame(recent_summary_rows)

        live_base_predictions: dict[str, pd.DataFrame] = {}
        for model_name in ("signal_proxy", "ridge_model", "xgboost_model"):
            if model_name not in model_predictions:
                continue
            scored = self._score_live_snapshot(
                all_snapshots=all_snapshots,
                matured=matured,
                model_name=model_name,
                target_column=target_column,
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
                xgboost_params=xgboost_params,
            )
            if scored is not None and not scored.empty:
                live_base_predictions[model_name] = scored
        live_ensemble_predictions = self._build_ensemble_predictions(live_base_predictions)
        if live_ensemble_predictions is not None:
            live_base_predictions["ensemble_model"] = live_ensemble_predictions
        live_predictions_all = live_base_predictions.get(champion_model)
        if live_predictions_all is None:
            raise ValueError(f"No live predictions available for champion_model={champion_model}.")
        live_predictions = live_predictions_all.sort_values(
            ["predicted_alpha", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).head(int(top_n)).reset_index(drop=True)

        report_path = self.db_manager.paths.reports_dir / "shortlist_model.md"
        oos_path = self.db_manager.paths.reports_dir / "shortlist_model_oos_predictions.csv"
        live_path = self.db_manager.paths.reports_dir / "shortlist_model_live_predictions.csv"
        generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()

        combined_predictions = pd.concat(
            [
                predictions.assign(model_name=model_name, dataset_split="oos")
                for model_name, predictions in model_predictions.items()
            ],
            axis=0,
            ignore_index=True,
        )
        combined_predictions.to_csv(oos_path, index=False)
        live_predictions_all.to_csv(live_path, index=False)

        lines = [
            "# Shortlist Model",
            "",
            f"- target_column: {target_column}",
            f"- top_n: {int(top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            f"- model_scope: {model_scope}",
            f"- xgboost_config: {xgboost_config}",
            f"- min_train_dates: {int(min_train_dates)}",
            f"- test_window_dates: {int(test_window_dates)}",
            "- objective: walk-forward cross-sectional ranking of the eligible universe on forward sector-relative alpha",
            f"- universe: {eligible_universe_mode_description(eligible_universe_mode)}",
            "- feature_matrix: raw features plus date-wise cross-sectional ranks and sector-relative ranks",
            "",
            f"- eligible_rows: {len(matured.index)}",
            f"- eligible_dates: {int(matured['snapshot_date'].nunique())}",
            f"- oos_prediction_dates: {int(combined_predictions['snapshot_date'].nunique())}",
            f"- champion_model: {champion_model}",
            f"- oos_predictions_csv: {oos_path}",
            f"- live_predictions_csv: {live_path}",
            f"- generated_at: {generated_at}",
            "",
        ]
        lines.extend(self._render_summary_table(full_summaries, heading="## Full Walk-Forward Evaluation"))
        lines.extend(self._render_summary_table(recent_summaries, heading=f"## Recent {min(int(recent_dates), int(combined_predictions['snapshot_date'].nunique()))} Walk-Forward Dates"))
        lines.extend(
            self._render_summary_table(
                self._rolling_window_summaries(
                    predictions=model_predictions[champion_model],
                    target_column=target_column,
                    model_name=champion_model,
                    top_n=int(top_n),
                    windows=(20, 40, 60),
                ),
                heading="## Champion Rolling Acceptance Windows",
            )
        )
        lines.extend(
            self._render_sector_contribution(
                predictions=model_predictions[champion_model],
                target_column=target_column,
                top_n=int(top_n),
                heading="## Champion Sector Contribution",
            )
        )
        lines.extend(self._render_live_candidates(champion_model=champion_model, frame=live_predictions))
        report_path.write_text("\n".join(lines), encoding="utf-8")

        self.db_manager.insert_shortlist_model_run(
            row={
                "generated_at": generated_at,
                "horizon_days": int(horizon_days),
                "eligible_universe_mode": eligible_universe_mode,
                "model_scope": model_scope,
                "xgboost_config": xgboost_config,
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
                "actual_alpha_vs_sector": row.get(target_column),
                "details": {
                    "model_top_reasons": self._ensure_reason_list(row.get("model_top_reasons")),
                    "model_reason_summary": row.get("model_reason_summary"),
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
        eligible_universe_mode: str,
    ) -> pd.DataFrame:
        working = frame.copy()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=["snapshot_date", target_column]).copy()
        working = filter_eligible_universe(
            working,
            eligible_universe_mode=eligible_universe_mode,
        )
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _build_live_eligible_universe(self, frame: pd.DataFrame, *, eligible_universe_mode: str) -> pd.DataFrame:
        working = filter_eligible_universe(
            frame.copy(),
            eligible_universe_mode=eligible_universe_mode,
        )
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _walk_forward_predictions(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        model_name: str,
        min_train_dates: int,
        test_window_dates: int,
        model_scope: str,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame | None:
        dates = sorted(frame["snapshot_date"].drop_duplicates().tolist())
        folds: list[pd.DataFrame] = []
        start_index = int(min_train_dates)
        while start_index < len(dates):
            train_dates = set(dates[:start_index])
            test_dates = dates[start_index : start_index + max(int(test_window_dates), 1)]
            if not test_dates:
                break
            train_frame = frame[frame["snapshot_date"].isin(train_dates)].copy()
            test_frame = frame[frame["snapshot_date"].isin(test_dates)].copy()
            scored = self._score_model(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                model_scope=model_scope,
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
            if scored is not None and not scored.empty:
                folds.append(
                    scored[
                        [
                            "snapshot_date",
                            "ticker",
                            "sector",
                            "md_volume_30d",
                            target_column,
                            "predicted_alpha",
                            "model_top_reasons",
                            "model_reason_summary",
                        ]
                    ].copy()
                )
            start_index += max(int(test_window_dates), 1)
        if not folds:
            return None
        return pd.concat(folds, axis=0, ignore_index=True)

    def _score_live_snapshot(
        self,
        *,
        all_snapshots: pd.DataFrame,
        matured: pd.DataFrame,
        model_name: str,
        target_column: str,
        eligible_universe_mode: str,
        model_scope: str,
        xgboost_params: dict[str, float | int] | None = None,
        feature_columns_override: list[str] | None = None,
    ) -> pd.DataFrame:
        latest_date = all_snapshots["snapshot_date"].max()
        live_snapshot = all_snapshots[all_snapshots["snapshot_date"] == latest_date].copy()
        live_snapshot = self._build_live_eligible_universe(
            live_snapshot,
            eligible_universe_mode=eligible_universe_mode,
        )
        if live_snapshot.empty:
            return live_snapshot.assign(predicted_alpha=pd.Series(dtype=float))
        scored = self._score_model(
            model_name=model_name,
            train_frame=matured,
            test_frame=live_snapshot,
            target_column=target_column,
            model_scope=model_scope,
            xgboost_params=xgboost_params,
            feature_columns_override=feature_columns_override,
        )
        if scored is None or scored.empty:
            return live_snapshot.assign(predicted_alpha=pd.Series(dtype=float))
        return scored

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
        if model_scope == "sector_specific" and model_name in {"ridge_model", "xgboost_model"}:
            return self._score_model_by_sector(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_column=target_column,
                xgboost_params=xgboost_params,
                feature_columns_override=feature_columns_override,
            )
        if model_name == "signal_proxy":
            return self._score_signal_proxy(test_frame)
        if model_name == "ridge_model":
            return self._score_ridge_model(
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

    def _score_ridge_model(
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

    def _prepare_model_matrices(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        feature_columns_override: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
        feature_frame = pd.concat(
            [
                train_frame[["snapshot_date", "sector"] + MODEL_FEATURE_COLUMNS].copy(),
                test_frame[["snapshot_date", "sector"] + MODEL_FEATURE_COLUMNS].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        feature_frame, feature_columns = build_rank_augmented_feature_frame(feature_frame)
        if feature_columns_override is not None:
            feature_columns = [column for column in feature_columns_override if column in feature_frame.columns]
            if not feature_columns:
                feature_columns = expand_model_feature_columns(MODEL_FEATURE_COLUMNS)
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
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for snapshot_date, day_frame in predictions.groupby("snapshot_date", sort=True):
            ordered = day_frame.sort_values(["predicted_alpha", "ticker"], ascending=[False, True]).copy()
            picks = ordered.head(int(top_n)).copy()
            target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
            universe_target = pd.to_numeric(day_frame[target_column], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(snapshot_date),
                    "pick_count": len(picks.index),
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(universe_target.mean()),
                }
            )
        if not rows:
            return self._empty_summary(model_name)
        frame = pd.DataFrame(rows)
        return {
            "model": model_name,
            "dates": len(frame.index),
            "avg_pick_count": float(frame["pick_count"].mean()),
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
        }

    def _rolling_window_summaries(
        self,
        *,
        predictions: pd.DataFrame,
        target_column: str,
        model_name: str,
        top_n: int,
        windows: tuple[int, ...],
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
        return pd.DataFrame(rows)

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
        }
        return f"{labels.get(base, base.replace('_', ' '))}{suffix}"

    def _choose_champion_model(self, summaries: pd.DataFrame) -> str:
        ordered = summaries.sort_values(
            ["mean_target", "beat_universe_rate", "positive_date_rate", "model"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        return str(ordered.iloc[0]["model"])

    def _empty_summary(self, model_name: str) -> dict[str, object]:
        return {
            "model": model_name,
            "dates": 0,
            "avg_pick_count": float("nan"),
            "mean_target": float("nan"),
            "hit_rate": float("nan"),
            "beat_universe_rate": float("nan"),
            "positive_date_rate": float("nan"),
            "ge_2pct_rate": float("nan"),
            "ge_5pct_rate": float("nan"),
        }

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
            lines.append(f"- mean_target: {self._fmt(row.mean_target)}")
            lines.append(f"- hit_rate: {self._fmt(row.hit_rate)}")
            lines.append(f"- beat_universe_rate: {self._fmt(row.beat_universe_rate)}")
            lines.append(f"- positive_date_rate: {self._fmt(row.positive_date_rate)}")
            lines.append(f"- ge_2pct_rate: {self._fmt(row.ge_2pct_rate)}")
            lines.append(f"- ge_5pct_rate: {self._fmt(row.ge_5pct_rate)}")
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
