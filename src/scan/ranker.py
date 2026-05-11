from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.scan.service import ScanPolicy


FEATURE_SNAPSHOT_NUMERIC_COLUMNS = [
    "relative_strength_index_vs_spy",
    "roc_63",
    "vol_alpha",
    "sma_200_dist",
    "sma_50_dist",
    "rsi_14",
    "atr_14",
    "days_to_next_earnings",
    "days_since_last_earnings",
    "avg_abs_gap_pct_20",
    "max_gap_down_pct_60",
    "distance_above_20d_high",
    "base_range_pct_20",
    "base_atr_contraction_20",
    "base_volume_dryup_ratio_20",
    "breakout_volume_ratio_50",
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
]

BASE_NUMERIC_COLUMNS = [
    "signal_score",
    "setup_quality_score",
    "expected_alpha_score",
    "breadth_score",
    "freshness_score",
    "overlap_penalty",
    "opportunity_score",
    "md_volume_30d",
    "adj_close",
    "shares",
]

CATEGORICAL_COLUMNS = [
    "strategy_slot",
    "strategy_sector",
    "sector",
    "regime_etf",
]


@dataclass(frozen=True)
class RankerValidationReport:
    available: bool
    target_column: str
    train_rows: int
    validation_rows: int
    train_dates: int
    validation_dates: int
    prediction_correlation: float
    learned_mean_target: float
    learned_hit_rate: float
    handcrafted_mean_target: float
    handcrafted_hit_rate: float
    runtime_mean_target: float
    runtime_hit_rate: float
    latest_scan_date: str | None
    latest_learned_tickers: tuple[str, ...]
    latest_handcrafted_tickers: tuple[str, ...]
    latest_runtime_tickers: tuple[str, ...]
    feature_count: int


class CandidateRanker:
    def __init__(
        self,
        *,
        target_column: str = "alpha_vs_sector_10d",
        ridge_alpha: float = 1.0,
        min_train_rows: int = 40,
        min_train_dates: int = 8,
    ) -> None:
        self.target_column = target_column
        self.ridge_alpha = ridge_alpha
        self.min_train_rows = min_train_rows
        self.min_train_dates = min_train_dates
        self.numeric_columns_: list[str] = []
        self.categorical_levels_: dict[str, list[str]] = {}
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_stds_: dict[str, float] = {}
        self.design_columns_: list[str] = []
        self.beta_: np.ndarray | None = None

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        scan_policy: "ScanPolicy",
        top_n: int = 6,
    ) -> RankerValidationReport:
        working = self._prepare_frame(frame)
        if self.target_column not in working.columns:
            return self._empty_report()
        labeled = working.dropna(subset=[self.target_column]).copy()
        if labeled.empty:
            return self._empty_report()
        train_frame, validation_frame = self._chronological_split(labeled, train_ratio=0.7)
        train_dates = int(train_frame["scan_date"].nunique()) if not train_frame.empty else 0
        validation_dates = int(validation_frame["scan_date"].nunique()) if not validation_frame.empty else 0
        if (
            train_frame.empty
            or validation_frame.empty
            or len(train_frame.index) < self.min_train_rows
            or train_dates < self.min_train_dates
        ):
            return self._empty_report(
                train_rows=len(train_frame.index),
                validation_rows=len(validation_frame.index),
                train_dates=train_dates,
                validation_dates=validation_dates,
            )
        self.fit(train_frame)
        scored_validation = self.score(validation_frame)
        correlation = float(pd.Series(scored_validation["ranker_score"]).corr(scored_validation[self.target_column].astype(float)))
        if not math.isfinite(correlation):
            correlation = 0.0
        per_date_rows: list[dict[str, object]] = []
        latest_scan_date = str(scored_validation["scan_date"].max()) if not scored_validation.empty else None
        latest_learned_tickers: tuple[str, ...] = ()
        latest_handcrafted_tickers: tuple[str, ...] = ()
        latest_runtime_tickers: tuple[str, ...] = ()
        for scan_date, day_frame in scored_validation.groupby("scan_date", sort=True):
            learned = self._select_candidates(day_frame, scan_policy=scan_policy, score_column="ranker_score", top_n=top_n)
            handcrafted = self._select_candidates(day_frame, scan_policy=scan_policy, score_column="opportunity_score", top_n=top_n)
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            if str(scan_date) == latest_scan_date:
                latest_learned_tickers = tuple(learned["ticker"].astype(str).tolist())
                latest_handcrafted_tickers = tuple(handcrafted["ticker"].astype(str).tolist())
                runtime_sort_columns = ["ticker"]
                runtime_ascending = [True]
                if "selected_rank" in runtime.columns:
                    runtime_sort_columns = ["selected_rank", "ticker"]
                    runtime_ascending = [True, True]
                latest_runtime_tickers = tuple(runtime.sort_values(runtime_sort_columns, ascending=runtime_ascending)["ticker"].astype(str).tolist())
            per_date_rows.extend(
                [
                    self._selection_metrics(scan_date, "learned", learned),
                    self._selection_metrics(scan_date, "handcrafted", handcrafted),
                    self._selection_metrics(scan_date, "runtime", runtime),
                ]
            )
        metrics = pd.DataFrame(per_date_rows)
        return RankerValidationReport(
            available=True,
            target_column=self.target_column,
            train_rows=len(train_frame.index),
            validation_rows=len(validation_frame.index),
            train_dates=train_dates,
            validation_dates=validation_dates,
            prediction_correlation=correlation,
            learned_mean_target=self._method_metric(metrics, "learned", "mean_target"),
            learned_hit_rate=self._method_metric(metrics, "learned", "hit_rate"),
            handcrafted_mean_target=self._method_metric(metrics, "handcrafted", "mean_target"),
            handcrafted_hit_rate=self._method_metric(metrics, "handcrafted", "hit_rate"),
            runtime_mean_target=self._method_metric(metrics, "runtime", "mean_target"),
            runtime_hit_rate=self._method_metric(metrics, "runtime", "hit_rate"),
            latest_scan_date=latest_scan_date,
            latest_learned_tickers=latest_learned_tickers,
            latest_handcrafted_tickers=latest_handcrafted_tickers,
            latest_runtime_tickers=latest_runtime_tickers,
            feature_count=len(self.design_columns_),
        )

    def fit(self, frame: pd.DataFrame) -> None:
        working = self._prepare_frame(frame)
        base = self._base_feature_frame(working)
        self.numeric_columns_ = [
            column
            for column in BASE_NUMERIC_COLUMNS + FEATURE_SNAPSHOT_NUMERIC_COLUMNS + ["candidate_count_day", "candidate_count_slot_day", "candidate_count_sector_day", "already_owned"]
            if column in base.columns and base[column].notna().any()
        ]
        self.categorical_levels_ = {
            column: sorted({str(value) for value in base[column].fillna("").astype(str) if str(value) != ""})
            for column in CATEGORICAL_COLUMNS
            if column in base.columns
        }
        self.numeric_medians_ = {
            column: float(base[column].astype(float).median()) if base[column].notna().any() else 0.0
            for column in self.numeric_columns_
        }
        transformed = self._transform_base_frame(base, fit=True)
        y = working[self.target_column].astype(float).to_numpy()
        design = np.column_stack([np.ones(len(transformed.index)), transformed.to_numpy(dtype=float)])
        penalty = np.eye(design.shape[1], dtype=float) * float(self.ridge_alpha)
        penalty[0, 0] = 0.0
        self.beta_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        self.design_columns_ = transformed.columns.tolist()

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.beta_ is None:
            raise RuntimeError("CandidateRanker must be fit before scoring.")
        working = self._prepare_frame(frame)
        base = self._base_feature_frame(working)
        transformed = self._transform_base_frame(base, fit=False)
        design = np.column_stack([np.ones(len(transformed.index)), transformed.to_numpy(dtype=float)])
        scored = working.copy()
        scored["ranker_score"] = design @ self.beta_
        return scored

    def score_details(self, frame: pd.DataFrame, *, top_features: int = 3) -> pd.DataFrame:
        if self.beta_ is None:
            raise RuntimeError("CandidateRanker must be fit before scoring.")
        working = self._prepare_frame(frame)
        base = self._base_feature_frame(working)
        transformed = self._transform_base_frame(base, fit=False)
        design = np.column_stack([np.ones(len(transformed.index)), transformed.to_numpy(dtype=float)])
        scored = working.copy()
        scored["ranker_score"] = design @ self.beta_
        contributions = transformed.to_numpy(dtype=float) * self.beta_[1:]
        positive_reasons: list[tuple[str, ...]] = []
        negative_reasons: list[tuple[str, ...]] = []
        for row in contributions:
            positive_reasons.append(self._top_contribution_reasons(row, positive=True, top_features=top_features))
            negative_reasons.append(self._top_contribution_reasons(row, positive=False, top_features=top_features))
        scored["ranker_top_positive_reasons"] = positive_reasons
        scored["ranker_top_negative_reasons"] = negative_reasons
        return scored

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "scan_date" not in working.columns:
            working["scan_date"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        else:
            working["scan_date"] = working["scan_date"].astype(str)
        if "strategy_slot" not in working.columns:
            working["strategy_slot"] = ""
        if "sector" not in working.columns:
            working["sector"] = ""
        if "details_json" in working.columns:
            details = working["details_json"].map(lambda raw: json.loads(raw) if raw else {})
        elif "details" in working.columns:
            details = working["details"].map(lambda payload: payload if isinstance(payload, dict) else {})
        else:
            details = pd.Series([{}] * len(working.index), index=working.index)
        working["details"] = details
        working["already_owned"] = details.map(lambda payload: 1.0 if payload.get("already_owned", False) else 0.0)
        working["feature_snapshot"] = details.map(lambda payload: payload.get("feature_snapshot", {}) if isinstance(payload.get("feature_snapshot", {}), dict) else {})
        for column in FEATURE_SNAPSHOT_NUMERIC_COLUMNS:
            working[column] = working["feature_snapshot"].map(lambda payload, key=column: payload.get(key))
        working["candidate_count_day"] = working.groupby("scan_date")["ticker"].transform("count").astype(float)
        working["candidate_count_slot_day"] = working.groupby(["scan_date", "strategy_slot"])["ticker"].transform("count").astype(float)
        working["candidate_count_sector_day"] = working.groupby(["scan_date", "sector"])["ticker"].transform("count").astype(float)
        return working

    def _base_feature_frame(self, working: pd.DataFrame) -> pd.DataFrame:
        columns = [
            *[column for column in BASE_NUMERIC_COLUMNS if column in working.columns],
            *[column for column in FEATURE_SNAPSHOT_NUMERIC_COLUMNS if column in working.columns],
            "candidate_count_day",
            "candidate_count_slot_day",
            "candidate_count_sector_day",
            "already_owned",
            *[column for column in CATEGORICAL_COLUMNS if column in working.columns],
        ]
        return working[columns].copy()

    def _transform_base_frame(self, base: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        transformed = pd.DataFrame(index=base.index)
        for column in self.numeric_columns_:
            series = pd.to_numeric(base[column], errors="coerce")
            fill_value = self.numeric_medians_.get(column, 0.0)
            series = series.fillna(fill_value)
            if fit:
                mean_value = float(series.mean()) if len(series.index) else 0.0
                std_value = float(series.std(ddof=0)) if len(series.index) else 0.0
                if not math.isfinite(std_value) or std_value <= 0.0:
                    std_value = 1.0
                self.numeric_means_[column] = mean_value
                self.numeric_stds_[column] = std_value
            mean_value = self.numeric_means_.get(column, 0.0)
            std_value = self.numeric_stds_.get(column, 1.0)
            transformed[column] = (series - mean_value) / std_value
        for column, levels in self.categorical_levels_.items():
            values = base[column].fillna("").astype(str) if column in base.columns else pd.Series([""] * len(base.index), index=base.index)
            for level in levels:
                transformed[f"{column}={level}"] = (values == level).astype(float)
        if fit:
            self.design_columns_ = transformed.columns.tolist()
        else:
            transformed = transformed.reindex(columns=self.design_columns_, fill_value=0.0)
        return transformed

    def _chronological_split(self, frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(pd.to_datetime(frame["scan_date"]).drop_duplicates().tolist())
        if len(unique_dates) < 2:
            return frame.iloc[0:0].copy(), frame.iloc[0:0].copy()
        split_index = int(len(unique_dates) * train_ratio)
        split_index = min(max(split_index, 1), len(unique_dates) - 1)
        train_dates = {date_value.strftime("%Y-%m-%d") for date_value in unique_dates[:split_index]}
        validation_dates = {date_value.strftime("%Y-%m-%d") for date_value in unique_dates[split_index:]}
        return (
            frame[frame["scan_date"].astype(str).isin(train_dates)].copy(),
            frame[frame["scan_date"].astype(str).isin(validation_dates)].copy(),
        )

    def _select_candidates(
        self,
        frame: pd.DataFrame,
        *,
        scan_policy: "ScanPolicy",
        score_column: str,
        top_n: int,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        ranked = frame.copy()
        if "md_volume_30d" not in ranked.columns:
            ranked["md_volume_30d"] = 0.0
        ranked = ranked.sort_values(
            ["already_owned", score_column, "signal_score", "md_volume_30d", "ticker"],
            ascending=[True, False, False, False, True],
        ).reset_index(drop=True)
        selected_indices: list[int] = []
        slot_counts: dict[str, int] = {}
        sector_counts: dict[str, int] = {}
        max_total = min(int(top_n), int(scan_policy.max_candidates_total))
        for candidate in ranked.itertuples():
            if len(selected_indices) >= max_total:
                break
            if bool(getattr(candidate, "already_owned", False)):
                continue
            slot = str(candidate.strategy_slot)
            sector = str(candidate.sector)
            if slot_counts.get(slot, 0) >= int(scan_policy.max_candidates_per_slot):
                continue
            if sector_counts.get(sector, 0) >= int(scan_policy.max_candidates_per_sector):
                continue
            selected_indices.append(int(candidate.Index))
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return ranked.loc[selected_indices].copy() if selected_indices else ranked.iloc[0:0].copy()

    def _selection_metrics(self, scan_date, method: str, frame: pd.DataFrame) -> dict[str, object]:
        if frame.empty:
            return {
                "scan_date": str(scan_date),
                "method": method,
                "mean_target": np.nan,
                "hit_rate": np.nan,
            }
        series = frame[self.target_column].dropna().astype(float)
        if series.empty:
            return {
                "scan_date": str(scan_date),
                "method": method,
                "mean_target": np.nan,
                "hit_rate": np.nan,
            }
        return {
            "scan_date": str(scan_date),
            "method": method,
            "mean_target": float(series.mean()),
            "hit_rate": float((series > 0.0).mean()),
        }

    def _method_metric(self, metrics: pd.DataFrame, method: str, column: str) -> float:
        subset = metrics.loc[metrics["method"] == method, column].dropna().astype(float)
        if subset.empty:
            return float("nan")
        return float(subset.mean())

    def _empty_report(
        self,
        *,
        train_rows: int = 0,
        validation_rows: int = 0,
        train_dates: int = 0,
        validation_dates: int = 0,
    ) -> RankerValidationReport:
        return RankerValidationReport(
            available=False,
            target_column=self.target_column,
            train_rows=train_rows,
            validation_rows=validation_rows,
            train_dates=train_dates,
            validation_dates=validation_dates,
            prediction_correlation=float("nan"),
            learned_mean_target=float("nan"),
            learned_hit_rate=float("nan"),
            handcrafted_mean_target=float("nan"),
            handcrafted_hit_rate=float("nan"),
            runtime_mean_target=float("nan"),
            runtime_hit_rate=float("nan"),
            latest_scan_date=None,
            latest_learned_tickers=(),
            latest_handcrafted_tickers=(),
            latest_runtime_tickers=(),
            feature_count=0,
        )

    def _top_contribution_reasons(
        self,
        row: np.ndarray,
        *,
        positive: bool,
        top_features: int,
    ) -> tuple[str, ...]:
        if row.size == 0 or not self.design_columns_:
            return ()
        order = np.argsort(-row if positive else row)
        reasons: list[str] = []
        for index in order:
            contribution = float(row[int(index)])
            if positive and contribution <= 0.0:
                continue
            if not positive and contribution >= 0.0:
                continue
            label = self._format_design_column(self.design_columns_[int(index)])
            reasons.append(f"{label} ({contribution:+.4f})")
            if len(reasons) >= int(top_features):
                break
        return tuple(reasons)

    def _format_design_column(self, column: str) -> str:
        if "=" in column:
            key, value = column.split("=", 1)
            return f"{key}: {value}"
        return column
