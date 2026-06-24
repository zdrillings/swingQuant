from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.research.shortlist_universe import (
    eligible_universe_mode_description,
    filter_eligible_universe,
    normalize_eligible_universe_mode,
)
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


MODEL_FEATURE_COLUMNS = [
    "relative_strength_index_vs_spy",
    "relative_strength_index_vs_qqq",
    "relative_strength_index_vs_xlk",
    "relative_strength_index_vs_subindustry",
    "roc_63",
    "roc_126",
    "vol_alpha",
    "sma_200_dist",
    "sma_50_dist",
    "rsi_14",
    "atr_14",
    "atr_pct_14",
    "atr_pct_14_percentile_252",
    "realized_vol_20_percentile_252",
    "days_to_next_earnings",
    "days_since_last_earnings",
    "last_earnings_gap_pct",
    "last_earnings_volume_ratio_20",
    "last_earnings_open_vs_20d_high",
    "close_vs_last_earnings_close",
    "avg_abs_gap_pct_20",
    "max_gap_down_pct_60",
    "distance_above_20d_high",
    "base_range_pct_20",
    "base_atr_contraction_20",
    "base_volume_dryup_ratio_20",
    "breakout_volume_ratio_50",
    "dollar_volume_ratio_20_60",
    "volume_percentile_60",
    "distance_from_52w_high",
    "days_since_52w_high",
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
]


def expand_model_feature_columns(base_features: list[str] | tuple[str, ...]) -> list[str]:
    ordered = [str(column) for column in base_features if str(column) in MODEL_FEATURE_COLUMNS]
    expanded = list(ordered)
    expanded.extend(f"{column}__rank_all" for column in ordered)
    expanded.extend(f"{column}__rank_sector" for column in ordered)
    return expanded


def build_rank_augmented_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    rank_columns: dict[str, pd.Series] = {}
    for column in MODEL_FEATURE_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
        values = pd.to_numeric(working[column], errors="coerce")
        working[column] = values
        rank_columns[f"{column}__rank_all"] = values.groupby(working["snapshot_date"]).rank(method="average", pct=True)
        if "sector" in working.columns:
            rank_columns[f"{column}__rank_sector"] = values.groupby(
                [working["snapshot_date"], working["sector"]]
            ).rank(method="average", pct=True)
    if rank_columns:
        working = pd.concat([working, pd.DataFrame(rank_columns, index=working.index)], axis=1)
    feature_columns = expand_model_feature_columns(MODEL_FEATURE_COLUMNS)
    return working, feature_columns


@dataclass(frozen=True)
class ShortlistBakeoffReport:
    output_path: str
    top_n: int
    target_column: str
    eligible_rows: int
    eligible_dates: int
    test_dates: int


class ShortlistBakeoffService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("shortlist_bakeoff")

    def run(
        self,
        *,
        top_n: int = 6,
        horizon_days: int = 20,
        recent_dates: int = 40,
        train_ratio: float = 0.7,
        eligible_universe_mode: str = "passed_only",
    ) -> ShortlistBakeoffReport:
        self.db_manager.initialize()
        target_column = f"alpha_vs_sector_{int(horizon_days)}d"
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")
        if target_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")

        eligible = self._build_eligible_universe(
            frame,
            target_column=target_column,
            eligible_universe_mode=eligible_universe_mode,
        )
        if eligible.empty:
            raise ValueError("No eligible universe rows found for shortlist bakeoff.")

        train_frame, test_frame = self._chronological_split(eligible, train_ratio=float(train_ratio))
        if train_frame.empty or test_frame.empty:
            raise ValueError("Not enough chronological history to create a bakeoff train/test split.")

        report_path = self.db_manager.paths.reports_dir / "shortlist_bakeoff.md"

        universe_sections = self._render_universe_model_bakeoff(
            train_frame=train_frame,
            test_frame=test_frame,
            target_column=target_column,
            top_n=int(top_n),
            recent_dates=int(recent_dates),
        )
        scan_sections = self._render_current_scan_policy_bakeoff(
            target_column=target_column,
            top_n=int(top_n),
            recent_dates=int(recent_dates),
        )

        lines = [
            "# Shortlist Bakeoff",
            "",
            f"- target_column: {target_column}",
            f"- top_n: {int(top_n)}",
            f"- eligible_universe_mode: {eligible_universe_mode}",
            "- objective: compare daily shortlist policies on forward sector-relative alpha",
            "- note: universe-model bakeoff uses chronological train/test split on `universe_daily_snapshots`",
            "- note: ridge/xgboost feature matrices use raw features plus date-wise cross-sectional ranks and sector-relative ranks",
            "- note: current scan policy bakeoff uses persisted `Scan_Candidates` rows where forward outcomes are already matured",
            f"- eligible_universe: {eligible_universe_mode_description(eligible_universe_mode)}",
            "",
            f"- eligible_rows: {len(eligible.index)}",
            f"- eligible_dates: {int(eligible['snapshot_date'].nunique())}",
            f"- train_dates: {int(train_frame['snapshot_date'].nunique())}",
            f"- test_dates: {int(test_frame['snapshot_date'].nunique())}",
            f"- test_date_range: {test_frame['snapshot_date'].min().date()} -> {test_frame['snapshot_date'].max().date()}",
            "",
        ]
        lines.extend(universe_sections)
        lines.extend(scan_sections)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return ShortlistBakeoffReport(
            output_path=str(report_path),
            top_n=int(top_n),
            target_column=target_column,
            eligible_rows=len(eligible.index),
            eligible_dates=int(eligible["snapshot_date"].nunique()),
            test_dates=int(test_frame["snapshot_date"].nunique()),
        )

    def _build_eligible_universe(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        eligible_universe_mode: str,
    ) -> pd.DataFrame:
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=[target_column, "snapshot_date"]).copy()
        working = filter_eligible_universe(
            working,
            eligible_universe_mode=eligible_universe_mode,
        )
        return working.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)

    def _chronological_split(self, frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = sorted(frame["snapshot_date"].dropna().unique().tolist())
        if len(dates) < 2:
            return pd.DataFrame(), pd.DataFrame()
        train_count = max(1, int(math.floor(len(dates) * train_ratio)))
        train_count = min(train_count, len(dates) - 1)
        train_dates = set(dates[:train_count])
        test_dates = set(dates[train_count:])
        return (
            frame[frame["snapshot_date"].isin(train_dates)].copy(),
            frame[frame["snapshot_date"].isin(test_dates)].copy(),
        )

    def _render_universe_model_bakeoff(
        self,
        *,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        target_column: str,
        top_n: int,
        recent_dates: int,
    ) -> list[str]:
        lines = ["## Universe Model Bakeoff", ""]
        lines.append("- policies: equal_weight_eligible, signal_proxy, ridge_model, xgboost_model, ensemble_model")
        lines.append("")

        scored_frames: dict[str, pd.DataFrame] = {
            "signal_proxy": self._score_signal_proxy(test_frame),
            "ridge_model": self._score_ridge_model(train_frame, test_frame, target_column=target_column),
        }
        tree_scored = self._score_xgboost_model(train_frame, test_frame, target_column=target_column)
        if tree_scored is not None:
            scored_frames["xgboost_model"] = tree_scored
        ensemble_scored = self._score_ensemble_model(scored_frames)
        if ensemble_scored is not None:
            scored_frames["ensemble_model"] = ensemble_scored

        summary_rows: list[dict[str, object]] = []
        for policy_name, scored in scored_frames.items():
            summary_rows.append(
                self._evaluate_policy(
                    frame=scored,
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name=policy_name,
                )
            )
        summary_rows.append(self._evaluate_equal_weight_policy(test_frame, target_column=target_column, policy_name="equal_weight_eligible"))

        summary_frame = pd.DataFrame(summary_rows)
        lines.extend(self._render_policy_window(summary_frame, heading="### Full Test Window"))

        recent_test_dates = sorted(test_frame["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
        recent_rows: list[dict[str, object]] = []
        for policy_name, scored in scored_frames.items():
            recent_rows.append(
                self._evaluate_policy(
                    frame=scored[scored["snapshot_date"].isin(recent_test_dates)].copy(),
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name=policy_name,
                )
            )
        recent_rows.append(
            self._evaluate_equal_weight_policy(
                test_frame[test_frame["snapshot_date"].isin(recent_test_dates)].copy(),
                target_column=target_column,
                policy_name="equal_weight_eligible",
            )
        )
        lines.extend(self._render_policy_window(pd.DataFrame(recent_rows), heading=f"### Recent {len(recent_test_dates)} Test Dates"))
        return lines

    def _render_current_scan_policy_bakeoff(
        self,
        *,
        target_column: str,
        top_n: int,
        recent_dates: int,
    ) -> list[str]:
        lines = ["## Current Scan Policy Bakeoff", ""]
        candidates = self.db_manager.load_scan_candidates()
        if candidates is None or candidates.empty:
            lines.append("No scan candidate history found.")
            lines.append("")
            return lines
        if target_column not in candidates.columns:
            lines.append(f"Scan candidate history does not include {target_column}.")
            lines.append("")
            return lines

        working = candidates.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working["selected"] = working["selected"].astype(int)
        working = working.dropna(subset=[target_column]).copy()
        working = working.sort_values(["scan_date", "ticker"]).reset_index(drop=True)
        if working.empty:
            lines.append("No matured scan candidate rows found for this target horizon.")
            lines.append("")
            return lines

        summaries = [
            self._evaluate_runtime_selected_policy(working, target_column=target_column, policy_name="runtime_selected"),
        ]
        if "opportunity_score" in working.columns:
            summaries.append(
                self._evaluate_policy(
                    frame=working.dropna(subset=["opportunity_score"]).rename(columns={"opportunity_score": "policy_score"}),
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name="opportunity_top_n",
                    date_column="scan_date",
                )
            )
        if "signal_score" in working.columns:
            summaries.append(
                self._evaluate_policy(
                    frame=working.dropna(subset=["signal_score"]).rename(columns={"signal_score": "policy_score"}),
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name="signal_top_n",
                    date_column="scan_date",
                )
            )
        lines.extend(self._render_policy_window(pd.DataFrame(summaries), heading="### Full Matured Scan Window"))

        recent_scan_dates = sorted(working["scan_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
        recent_working = working[working["scan_date"].isin(recent_scan_dates)].copy()
        recent_summaries = [
            self._evaluate_runtime_selected_policy(recent_working, target_column=target_column, policy_name="runtime_selected"),
        ]
        if "opportunity_score" in recent_working.columns:
            recent_summaries.append(
                self._evaluate_policy(
                    frame=recent_working.dropna(subset=["opportunity_score"]).rename(columns={"opportunity_score": "policy_score"}),
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name="opportunity_top_n",
                    date_column="scan_date",
                )
            )
        if "signal_score" in recent_working.columns:
            recent_summaries.append(
                self._evaluate_policy(
                    frame=recent_working.dropna(subset=["signal_score"]).rename(columns={"signal_score": "policy_score"}),
                    score_column="policy_score",
                    target_column=target_column,
                    top_n=top_n,
                    policy_name="signal_top_n",
                    date_column="scan_date",
                )
            )
        lines.extend(self._render_policy_window(pd.DataFrame(recent_summaries), heading=f"### Recent {len(recent_scan_dates)} Matured Scan Dates"))
        return lines

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
            working[f"{component}_rank"] = values.groupby(working["snapshot_date"]).rank(method="average", pct=True)
        working["policy_score"] = working[[f"{component}_rank" for component in components]].mean(axis=1, skipna=True)
        return working

    def _score_ridge_model(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
        train_matrix, test_matrix = self._prepare_model_matrices(train_frame, test_frame)
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        ridge_penalty = 1.0
        xtx = train_matrix.T @ train_matrix
        identity = np.eye(xtx.shape[0], dtype=float)
        weights = np.linalg.solve(xtx + ridge_penalty * identity, train_matrix.T @ train_target)
        scored = test_frame.copy()
        scored["policy_score"] = test_matrix @ weights
        return scored

    def _score_xgboost_model(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        *,
        target_column: str,
    ) -> pd.DataFrame | None:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError:
            self.logger.warning("xgboost unavailable; skipping xgboost_model in shortlist bakeoff.")
            return None
        train_matrix, test_matrix = self._prepare_model_matrices(train_frame, test_frame)
        train_target = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=float)
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(train_matrix, train_target, verbose=False)
        scored = test_frame.copy()
        scored["policy_score"] = model.predict(test_matrix)
        return scored

    def _score_ensemble_model(self, scored_frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        usable = {
            model_name: frame.copy()
            for model_name, frame in scored_frames.items()
            if frame is not None and not frame.empty
        }
        if len(usable) < 2:
            return None
        base_model_name = next(iter(usable))
        merged = usable[base_model_name].copy()
        merged = merged.rename(columns={"policy_score": f"{base_model_name}_score"})
        for model_name, frame in usable.items():
            if model_name == base_model_name:
                continue
            scoped = frame[["snapshot_date", "ticker", "policy_score"]].copy()
            scoped = scoped.rename(columns={"policy_score": f"{model_name}_score"})
            merged = merged.merge(scoped, on=["snapshot_date", "ticker"], how="inner")
        if merged is None or merged.empty:
            return None
        rank_columns: list[str] = []
        for model_name in usable:
            source_column = f"{model_name}_score"
            rank_column = f"{model_name}_rank"
            merged[rank_column] = pd.to_numeric(merged[source_column], errors="coerce").groupby(
                merged["snapshot_date"]
            ).rank(method="average", pct=True)
            rank_columns.append(rank_column)
        merged["policy_score"] = merged[rank_columns].mean(axis=1, skipna=True)
        return merged

    def _prepare_model_matrices(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feature_frame = pd.concat(
            [
                train_frame[["snapshot_date", "sector"] + MODEL_FEATURE_COLUMNS].copy(),
                test_frame[["snapshot_date", "sector"] + MODEL_FEATURE_COLUMNS].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        feature_frame, feature_columns = build_rank_augmented_feature_frame(feature_frame)
        feature_frame = feature_frame[feature_columns + ["sector"]].copy()
        feature_frame = pd.get_dummies(feature_frame, columns=["sector"], dummy_na=False)
        train_features = feature_frame.iloc[: len(train_frame.index)].copy()
        test_features = feature_frame.iloc[len(train_frame.index) :].copy()
        train_medians = train_features.median(numeric_only=True)
        train_features = train_features.fillna(train_medians)
        test_features = test_features.fillna(train_medians)
        means = train_features.mean(axis=0)
        stds = train_features.std(axis=0).replace(0.0, 1.0)
        train_matrix = ((train_features - means) / stds).to_numpy(dtype=float)
        test_matrix = ((test_features - means) / stds).to_numpy(dtype=float)
        return train_matrix, test_matrix

    def _evaluate_runtime_selected_policy(self, frame: pd.DataFrame, *, target_column: str, policy_name: str) -> dict[str, object]:
        if frame.empty:
            return self._empty_policy_summary(policy_name)
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for scan_date, day_frame in frame.groupby("scan_date", sort=True):
            selected = day_frame[day_frame["selected"] == 1].copy()
            excluded = day_frame[day_frame["selected"] == 0].copy()
            if selected.empty:
                continue
            target = pd.to_numeric(selected[target_column], errors="coerce").dropna()
            excluded_target = pd.to_numeric(excluded[target_column], errors="coerce").dropna()
            if target.empty:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(scan_date),
                    "pick_count": len(selected.index),
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(excluded_target.mean()) if not excluded_target.empty else float("nan"),
                }
            )
        return self._aggregate_policy_daily_rows(rows, policy_name)

    def _evaluate_equal_weight_policy(self, frame: pd.DataFrame, *, target_column: str, policy_name: str) -> dict[str, object]:
        if frame.empty:
            return self._empty_policy_summary(policy_name)
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for snapshot_date, day_frame in frame.groupby("snapshot_date", sort=True):
            target = pd.to_numeric(day_frame[target_column], errors="coerce").dropna()
            if target.empty:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(snapshot_date),
                    "pick_count": len(day_frame.index),
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(target.mean()),
                }
            )
        return self._aggregate_policy_daily_rows(rows, policy_name)

    def _evaluate_policy(
        self,
        *,
        frame: pd.DataFrame,
        score_column: str,
        target_column: str,
        top_n: int,
        policy_name: str,
        date_column: str = "snapshot_date",
    ) -> dict[str, object]:
        if frame.empty:
            return self._empty_policy_summary(policy_name)
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for current_date, day_frame in frame.groupby(date_column, sort=True):
            ordered = day_frame.sort_values([score_column, "ticker"], ascending=[False, True]).copy()
            picks = ordered.head(int(top_n)).copy()
            target = pd.to_numeric(picks[target_column], errors="coerce").dropna()
            universe_target = pd.to_numeric(day_frame[target_column], errors="coerce").dropna()
            if target.empty or universe_target.empty:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(current_date),
                    "pick_count": len(picks.index),
                    "mean_target": float(target.mean()),
                    "hit_rate": float((target > 0.0).mean()),
                    "universe_mean_target": float(universe_target.mean()),
                }
            )
        return self._aggregate_policy_daily_rows(rows, policy_name)

    def _aggregate_policy_daily_rows(self, rows: list[dict[str, object]], policy_name: str) -> dict[str, object]:
        if not rows:
            return self._empty_policy_summary(policy_name)
        frame = pd.DataFrame(rows)
        return {
            "policy": policy_name,
            "dates": len(frame.index),
            "avg_pick_count": float(frame["pick_count"].mean()),
            "mean_target": float(frame["mean_target"].mean()),
            "hit_rate": float(frame["hit_rate"].mean()),
            "beat_universe_rate": float((frame["mean_target"] > frame["universe_mean_target"]).mean()),
            "positive_date_rate": float((frame["mean_target"] > 0.0).mean()),
            "ge_2pct_rate": float((frame["mean_target"] >= 0.02).mean()),
            "ge_5pct_rate": float((frame["mean_target"] >= 0.05).mean()),
        }

    def _empty_policy_summary(self, policy_name: str) -> dict[str, object]:
        return {
            "policy": policy_name,
            "dates": 0,
            "avg_pick_count": float("nan"),
            "mean_target": float("nan"),
            "hit_rate": float("nan"),
            "beat_universe_rate": float("nan"),
            "positive_date_rate": float("nan"),
            "ge_2pct_rate": float("nan"),
            "ge_5pct_rate": float("nan"),
        }

    def _render_policy_window(self, frame: pd.DataFrame, *, heading: str) -> list[str]:
        lines = [heading, ""]
        if frame.empty:
            lines.append("No policy results available.")
            lines.append("")
            return lines
        ordered = frame.sort_values(["mean_target", "beat_universe_rate", "policy"], ascending=[False, False, True]).reset_index(drop=True)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.policy}")
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

    def _fmt(self, value: float) -> str:
        if value is None or not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
