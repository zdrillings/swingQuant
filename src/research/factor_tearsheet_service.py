from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
import pandas as pd

from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


FACTOR_COLUMNS = [
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
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
]


@dataclass(frozen=True)
class FactorQuintileSummary:
    quintile: int
    row_count: int
    date_count: int
    mean_target: float
    hit_rate: float


@dataclass(frozen=True)
class FactorSummary:
    feature: str
    coverage_rows: int
    coverage_dates: int
    coverage_rate: float
    q5_minus_q1_spread: float
    daily_ic_mean: float
    daily_ic_t_stat: float
    turnover: float
    best_regime: str
    best_regime_mean_target: float
    worst_regime: str
    worst_regime_mean_target: float
    quintiles: tuple[FactorQuintileSummary, ...]


@dataclass(frozen=True)
class UnavailableFactorSummary:
    feature: str
    reason: str
    coverage_rows: int
    coverage_dates: int


@dataclass(frozen=True)
class FactorTearsheetReport:
    output_path: str
    sector: str
    horizon_days: int
    factor_count: int
    snapshot_rows: int
    snapshot_dates: int


class FactorTearsheetService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("factor_tearsheet")

    def run(
        self,
        *,
        sector: str,
        horizon_days: int = 10,
    ) -> FactorTearsheetReport:
        self.db_manager.initialize()
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")

        horizon_column = f"alpha_vs_sector_{int(horizon_days)}d"
        if horizon_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")

        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        working = working[working["sector"].astype(str) == str(sector)].copy()
        working = working.dropna(subset=[horizon_column]).copy()
        if working.empty:
            raise ValueError(f"No matured universe snapshots found for sector={sector!r} horizon_days={horizon_days}.")
        working["target"] = pd.to_numeric(working[horizon_column], errors="coerce")
        working = working.dropna(subset=["target"]).copy()
        if working.empty:
            raise ValueError(f"No numeric {horizon_column} rows found for sector={sector!r}.")
        working["regime_label"] = self._build_regime_labels(working)

        factor_summaries: list[FactorSummary] = []
        unavailable_summaries: list[UnavailableFactorSummary] = []
        for column in FACTOR_COLUMNS:
            if column not in working.columns:
                unavailable_summaries.append(
                    UnavailableFactorSummary(
                        feature=column,
                        reason="missing_column",
                        coverage_rows=0,
                        coverage_dates=0,
                    )
                )
                continue
            summary, unavailable = self._summarize_factor(working, feature=column)
            if summary is not None:
                factor_summaries.append(summary)
            elif unavailable is not None:
                unavailable_summaries.append(unavailable)
        if not factor_summaries:
            raise ValueError(f"No factor columns with usable coverage found for sector={sector!r}.")
        factor_summaries.sort(
            key=lambda summary: (
                -self._sort_value(summary.q5_minus_q1_spread),
                -self._sort_value(summary.daily_ic_mean),
                -float(summary.coverage_rows),
                summary.feature,
            )
        )

        report_path = self.db_manager.paths.reports_dir / f"factor_tearsheet_{self._slugify(sector)}_{int(horizon_days)}d.md"
        report_path.write_text(
            "\n".join(
                self._render_report(
                    working,
                    sector=str(sector),
                    horizon_days=int(horizon_days),
                    summaries=factor_summaries,
                    unavailable_summaries=unavailable_summaries,
                )
            ),
            encoding="utf-8",
        )
        return FactorTearsheetReport(
            output_path=str(report_path),
            sector=str(sector),
            horizon_days=int(horizon_days),
            factor_count=len(factor_summaries),
            snapshot_rows=len(working.index),
            snapshot_dates=int(working["snapshot_date"].nunique()),
        )

    def _summarize_factor(
        self,
        frame: pd.DataFrame,
        *,
        feature: str,
    ) -> tuple[FactorSummary | None, UnavailableFactorSummary | None]:
        subset = frame[["snapshot_date", "ticker", "target", "regime_label", feature]].copy()
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
        subset = subset.dropna(subset=[feature, "target"]).copy()
        if subset.empty:
            return None, UnavailableFactorSummary(
                feature=feature,
                reason="no_non_null_rows",
                coverage_rows=0,
                coverage_dates=0,
            )
        subset = self._assign_feature_quintiles(subset, feature=feature)
        subset = subset.dropna(subset=["factor_quintile"]).copy()
        if subset.empty:
            return None, UnavailableFactorSummary(
                feature=feature,
                reason="no_rankable_rows",
                coverage_rows=int(frame[feature].notna().sum()) if feature in frame.columns else 0,
                coverage_dates=int(frame.loc[frame[feature].notna(), "snapshot_date"].nunique()) if feature in frame.columns else 0,
            )
        subset["factor_quintile"] = subset["factor_quintile"].astype(int)
        quintiles = self._quintile_summaries(subset)
        q5_minus_q1 = self._quintile_spread(quintiles, high_quintile=5, low_quintile=1)
        ic_values = self._daily_ic_values(subset, feature=feature)
        daily_ic_mean, _daily_ic_std, daily_ic_t_stat = self._ic_summary(ic_values)
        turnover = self._top_quintile_turnover(subset)
        best_regime, best_mean, worst_regime, worst_mean = self._best_worst_regimes(subset)
        return (
            FactorSummary(
                feature=feature,
                coverage_rows=len(subset.index),
                coverage_dates=int(subset["snapshot_date"].nunique()),
                coverage_rate=float(len(subset.index) / len(frame.index)) if len(frame.index) else 0.0,
                q5_minus_q1_spread=q5_minus_q1,
                daily_ic_mean=daily_ic_mean,
                daily_ic_t_stat=daily_ic_t_stat,
                turnover=turnover,
                best_regime=best_regime,
                best_regime_mean_target=best_mean,
                worst_regime=worst_regime,
                worst_regime_mean_target=worst_mean,
                quintiles=quintiles,
            ),
            None,
        )

    def _assign_feature_quintiles(self, frame: pd.DataFrame, *, feature: str) -> pd.DataFrame:
        working = frame.copy()
        working["factor_quintile"] = pd.Series(dtype=int)
        for _, day_frame in working.groupby("snapshot_date", sort=True):
            ordered = day_frame.sort_values([feature, "ticker"], ascending=[True, True]).copy()
            row_count = len(ordered.index)
            if row_count == 0:
                continue
            quintiles = ((np.arange(row_count, dtype=int) * 5) // row_count) + 1
            ordered["factor_quintile"] = quintiles.astype(int)
            working.loc[ordered.index, "factor_quintile"] = ordered["factor_quintile"]
        return working

    def _quintile_summaries(self, frame: pd.DataFrame) -> tuple[FactorQuintileSummary, ...]:
        summaries: list[FactorQuintileSummary] = []
        for quintile in range(1, 6):
            subset = frame[frame["factor_quintile"].astype(int) == quintile].copy()
            target = pd.to_numeric(subset["target"], errors="coerce").dropna()
            summaries.append(
                FactorQuintileSummary(
                    quintile=quintile,
                    row_count=len(subset.index),
                    date_count=int(subset["snapshot_date"].nunique()) if not subset.empty else 0,
                    mean_target=float(target.mean()) if not target.empty else float("nan"),
                    hit_rate=float((target > 0.0).mean()) if not target.empty else float("nan"),
                )
            )
        return tuple(summaries)

    def _quintile_spread(
        self,
        summaries: tuple[FactorQuintileSummary, ...],
        *,
        high_quintile: int,
        low_quintile: int,
    ) -> float:
        high = next((summary.mean_target for summary in summaries if summary.quintile == high_quintile), float("nan"))
        low = next((summary.mean_target for summary in summaries if summary.quintile == low_quintile), float("nan"))
        if not math.isfinite(high) or not math.isfinite(low):
            return float("nan")
        return float(high - low)

    def _daily_ic_values(self, frame: pd.DataFrame, *, feature: str) -> list[float]:
        values: list[float] = []
        for _, day_frame in frame.groupby("snapshot_date", sort=True):
            if len(day_frame.index) < 3:
                continue
            feature_values = pd.to_numeric(day_frame[feature], errors="coerce").dropna()
            target_values = pd.to_numeric(day_frame["target"], errors="coerce").dropna()
            if feature_values.empty or target_values.empty:
                continue
            if feature_values.nunique(dropna=True) < 2 or target_values.nunique(dropna=True) < 2:
                continue
            correlation = pd.Series(day_frame[feature]).corr(pd.Series(day_frame["target"]), method="spearman")
            if math.isfinite(float(correlation)):
                values.append(float(correlation))
        return values

    def _ic_summary(self, values: list[float]) -> tuple[float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan")
        array = np.asarray(values, dtype=float)
        mean_value = float(np.nanmean(array))
        std_value = float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0
        if len(array) > 1 and std_value > 0:
            t_stat = float((mean_value / std_value) * math.sqrt(len(array)))
        else:
            t_stat = float("nan")
        return mean_value, std_value, t_stat

    def _top_quintile_turnover(self, frame: pd.DataFrame) -> float:
        top_sets: list[set[str]] = []
        for _, day_frame in frame.groupby("snapshot_date", sort=True):
            tickers = {
                str(ticker)
                for ticker in day_frame[day_frame["factor_quintile"].astype(int) == 5]["ticker"].astype(str).tolist()
            }
            if tickers:
                top_sets.append(tickers)
        if len(top_sets) < 2:
            return float("nan")
        turnover_values: list[float] = []
        for previous, current in zip(top_sets, top_sets[1:]):
            union = previous.union(current)
            if not union:
                continue
            overlap = previous.intersection(current)
            turnover_values.append(float(1.0 - (len(overlap) / len(union))))
        return float(np.nanmean(np.asarray(turnover_values, dtype=float))) if turnover_values else float("nan")

    def _best_worst_regimes(self, frame: pd.DataFrame) -> tuple[str, float, str, float]:
        regime_rows: list[tuple[str, float]] = []
        top_quintile = frame[frame["factor_quintile"].astype(int) == 5].copy()
        if top_quintile.empty:
            return "n/a", float("nan"), "n/a", float("nan")
        for regime_label, subset in top_quintile.groupby("regime_label", sort=True):
            target = pd.to_numeric(subset["target"], errors="coerce").dropna()
            if target.empty:
                continue
            regime_rows.append((str(regime_label), float(target.mean())))
        if not regime_rows:
            return "n/a", float("nan"), "n/a", float("nan")
        regime_rows.sort(key=lambda item: (item[1], item[0]))
        worst_regime, worst_mean = regime_rows[0]
        best_regime, best_mean = regime_rows[-1]
        return best_regime, best_mean, worst_regime, worst_mean

    def _build_regime_labels(self, frame: pd.DataFrame) -> pd.Series:
        breadth = pd.to_numeric(frame.get("sector_pct_above_50"), errors="coerce")
        breadth_bucket = pd.Series("breadth_unknown", index=frame.index, dtype=object)
        breadth_bucket.loc[breadth < 0.40] = "breadth_weak"
        breadth_bucket.loc[(breadth >= 0.40) & (breadth < 0.70)] = "breadth_mixed"
        breadth_bucket.loc[breadth >= 0.70] = "breadth_strong"
        market_bucket = frame.get("regime_green", pd.Series(False, index=frame.index)).fillna(False).map(
            lambda value: "market_green" if bool(value) else "market_red"
        )
        return market_bucket.astype(str) + " / " + breadth_bucket.astype(str)

    def _render_report(
        self,
        frame: pd.DataFrame,
        *,
        sector: str,
        horizon_days: int,
        summaries: list[FactorSummary],
        unavailable_summaries: list[UnavailableFactorSummary],
    ) -> list[str]:
        lines = [
            "# Factor Tearsheet",
            "",
            f"- sector: {sector}",
            f"- horizon_days: {horizon_days}",
            f"- snapshot_rows: {len(frame.index)}",
            f"- snapshot_dates: {int(frame['snapshot_date'].nunique())}",
            f"- tickers: {int(frame['ticker'].nunique())}",
            f"- matured_date_min: {frame['snapshot_date'].min().date()}",
            f"- matured_date_max: {frame['snapshot_date'].max().date()}",
            "",
        ]
        for summary in summaries:
            lines.append(f"## {summary.feature}")
            lines.append("")
            lines.append(f"- coverage_rows: {summary.coverage_rows}")
            lines.append(f"- coverage_dates: {summary.coverage_dates}")
            lines.append(f"- coverage_rate: {summary.coverage_rate:.4f}")
            lines.append(f"- q5_minus_q1_spread: {self._format_float(summary.q5_minus_q1_spread)}")
            lines.append(f"- daily_ic_mean: {self._format_float(summary.daily_ic_mean)}")
            lines.append(f"- ic_t_stat: {self._format_float(summary.daily_ic_t_stat)}")
            lines.append(f"- turnover_q5: {self._format_float(summary.turnover)}")
            lines.append(
                f"- best_regime: {summary.best_regime} ({self._format_float(summary.best_regime_mean_target)})"
            )
            lines.append(
                f"- worst_regime: {summary.worst_regime} ({self._format_float(summary.worst_regime_mean_target)})"
            )
            lines.append("")
            lines.append("### Mean Forward Sector Alpha By Quintile")
            lines.append("")
            for quintile in summary.quintiles:
                lines.append(
                    f"- Q{quintile.quintile}: mean_target={self._format_float(quintile.mean_target)} "
                    f"hit_rate={self._format_float(quintile.hit_rate, digits=4)} "
                    f"rows={quintile.row_count} dates={quintile.date_count}"
                )
            lines.append("")
        if unavailable_summaries:
            lines.append("## Unavailable Factors")
            lines.append("")
            for summary in unavailable_summaries:
                lines.append(
                    f"- {summary.feature}: reason={summary.reason} coverage_rows={summary.coverage_rows} coverage_dates={summary.coverage_dates}"
                )
            lines.append("")
        return lines

    def _sort_value(self, value: float) -> float:
        return float(value) if math.isfinite(value) else float("-inf")

    def _format_float(self, value: float, *, digits: int = 6) -> str:
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
