from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


FEATURE_PROFILE_COLUMNS = [
    "relative_strength_index_vs_spy",
    "roc_63",
    "vol_alpha",
    "sma_200_dist",
    "sma_50_dist",
    "rsi_14",
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
]


@dataclass(frozen=True)
class UniverseAnalysisReport:
    output_path: str
    snapshot_rows: int
    snapshot_dates: int
    horizon_days: int


class UniverseAnalysisService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("universe_analysis")

    def run(
        self,
        *,
        top: int = 10,
        horizon_days: int = 10,
        recent_dates: int = 20,
    ) -> UniverseAnalysisReport:
        self.db_manager.initialize()
        frame = self.db_manager.load_universe_daily_snapshots()
        if frame.empty:
            raise ValueError("No universe snapshots found. Run `sq universe-backfill` first.")

        horizon_column = f"alpha_vs_sector_{int(horizon_days)}d"
        raw_return_column = f"fwd_return_{int(horizon_days)}d"
        if horizon_column not in frame.columns or raw_return_column not in frame.columns:
            raise ValueError(f"Universe snapshots do not include horizon_days={horizon_days}.")

        working = frame.copy()
        working["passed_any_strategy"] = working["passed_any_strategy"].astype(bool)
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        matured = working.dropna(subset=[horizon_column, raw_return_column]).copy()
        if matured.empty:
            raise ValueError(f"No matured universe snapshot rows are available for horizon_days={horizon_days}.")

        matured["winner_decile"] = self._assign_deciles_by_date(matured, score_column=horizon_column, buckets=10)
        missed_winners = matured[
            (~matured["passed_any_strategy"])
            & (matured["winner_decile"] == 1)
        ].copy()
        passed_winners = matured[
            matured["passed_any_strategy"]
            & (pd.to_numeric(matured[horizon_column], errors="coerce") > 0.0)
        ].copy()
        recent_dates_values = sorted(matured["snapshot_date"].drop_duplicates().tolist())[-max(int(recent_dates), 1):]
        recent_missed_winners = missed_winners[missed_winners["snapshot_date"].isin(recent_dates_values)].copy()

        report_path = self.db_manager.paths.reports_dir / "universe_analysis.md"
        lines = [
            "# Universe Analysis",
            "",
            f"- horizon_days: {int(horizon_days)}",
            f"- snapshot_rows: {len(working.index)}",
            f"- snapshot_dates: {int(working['snapshot_date'].nunique())}",
            f"- tickers: {int(working['ticker'].nunique())}",
            f"- snapshot_date_min: {working['snapshot_date'].min().date()}",
            f"- snapshot_date_max: {working['snapshot_date'].max().date()}",
            f"- matured_date_max: {matured['snapshot_date'].max().date()}",
            f"- passed_any_rate: {working['passed_any_strategy'].mean():.4f}",
            "",
        ]
        lines.extend(self._render_gate_outcome_summary(matured, raw_return_column=raw_return_column, horizon_column=horizon_column))
        lines.extend(self._render_sector_pass_rates(working, top=top))
        lines.extend(self._render_sector_outcome_comparison(matured, raw_return_column=raw_return_column, horizon_column=horizon_column, top=top))
        lines.extend(self._render_missed_winner_sectors(missed_winners, horizon_column=horizon_column, top=top))
        lines.extend(self._render_recent_missed_winners(recent_missed_winners, horizon_column=horizon_column, top=top))
        lines.extend(self._render_feature_profile_comparison(passed_winners, missed_winners))
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return UniverseAnalysisReport(
            output_path=str(report_path),
            snapshot_rows=len(working.index),
            snapshot_dates=int(working["snapshot_date"].nunique()),
            horizon_days=int(horizon_days),
        )

    def _assign_deciles_by_date(self, frame: pd.DataFrame, *, score_column: str, buckets: int) -> pd.Series:
        result = pd.Series(index=frame.index, dtype=int)
        for _, day_frame in frame.groupby("snapshot_date", sort=True):
            ordered = day_frame.sort_values([score_column, "ticker"], ascending=[False, True]).copy()
            row_count = len(ordered.index)
            if row_count == 0:
                continue
            bucket_values = ((np.arange(row_count, dtype=int) * int(buckets)) // row_count) + 1
            result.loc[ordered.index] = bucket_values.astype(int)
        return result

    def _render_gate_outcome_summary(self, frame: pd.DataFrame, *, raw_return_column: str, horizon_column: str) -> list[str]:
        lines = ["## Gate Outcome Summary", ""]
        for passed, title in ((True, "Passed"), (False, "Not Passed")):
            subset = frame[frame["passed_any_strategy"] == passed].copy()
            raw_series = pd.to_numeric(subset[raw_return_column], errors="coerce").dropna()
            alpha_series = pd.to_numeric(subset[horizon_column], errors="coerce").dropna()
            lines.append(f"### {title}")
            lines.append(f"- rows: {len(subset.index)}")
            lines.append(f"- mean_{raw_return_column}: {raw_series.mean():.6f}")
            lines.append(f"- hit_rate_{raw_return_column}: {(raw_series > 0.0).mean():.4f}")
            lines.append(f"- mean_{horizon_column}: {alpha_series.mean():.6f}")
            lines.append(f"- hit_rate_{horizon_column}: {(alpha_series > 0.0).mean():.4f}")
            lines.append("")
        return lines

    def _render_sector_pass_rates(self, frame: pd.DataFrame, *, top: int) -> list[str]:
        lines = ["## Sector Pass Rates", ""]
        grouped = (
            frame.groupby("sector", dropna=False)
            .agg(
                rows=("ticker", "count"),
                pass_rate=("passed_any_strategy", "mean"),
                avg_pass_count=("strategy_pass_count", "mean"),
            )
            .reset_index()
            .sort_values(["pass_rate", "rows", "sector"], ascending=[False, False, True])
            .head(top)
        )
        for row in grouped.itertuples(index=False):
            lines.append(f"### {row.sector}")
            lines.append(f"- rows: {int(row.rows)}")
            lines.append(f"- pass_rate: {float(row.pass_rate):.4f}")
            lines.append(f"- avg_pass_count: {float(row.avg_pass_count):.4f}")
            lines.append("")
        return lines

    def _render_sector_outcome_comparison(
        self,
        frame: pd.DataFrame,
        *,
        raw_return_column: str,
        horizon_column: str,
        top: int,
    ) -> list[str]:
        lines = ["## Sector Outcome Comparison", ""]
        rows: list[dict[str, object]] = []
        for sector, sector_frame in frame.groupby("sector", dropna=False):
            passed = sector_frame[sector_frame["passed_any_strategy"]].copy()
            not_passed = sector_frame[~sector_frame["passed_any_strategy"]].copy()
            passed_alpha = pd.to_numeric(passed[horizon_column], errors="coerce").dropna()
            not_passed_alpha = pd.to_numeric(not_passed[horizon_column], errors="coerce").dropna()
            if passed_alpha.empty and not_passed_alpha.empty:
                continue
            rows.append(
                {
                    "sector": sector,
                    "passed_rows": len(passed.index),
                    "not_passed_rows": len(not_passed.index),
                    "passed_mean_alpha": float(passed_alpha.mean()) if not passed_alpha.empty else float("nan"),
                    "not_passed_mean_alpha": float(not_passed_alpha.mean()) if not not_passed_alpha.empty else float("nan"),
                    "alpha_spread": (
                        float(passed_alpha.mean() - not_passed_alpha.mean())
                        if not passed_alpha.empty and not not_passed_alpha.empty
                        else float("nan")
                    ),
                    "passed_mean_return": float(pd.to_numeric(passed[raw_return_column], errors="coerce").dropna().mean()) if len(passed.index) else float("nan"),
                    "not_passed_mean_return": float(pd.to_numeric(not_passed[raw_return_column], errors="coerce").dropna().mean()) if len(not_passed.index) else float("nan"),
                }
            )
        summary = pd.DataFrame(rows)
        if summary.empty:
            lines.append("No sector outcome comparison rows available.")
            lines.append("")
            return lines
        summary = summary.sort_values(["alpha_spread", "passed_rows", "sector"], ascending=[False, False, True]).head(top)
        for row in summary.itertuples(index=False):
            lines.append(f"### {row.sector}")
            lines.append(f"- passed_rows: {int(row.passed_rows)}")
            lines.append(f"- not_passed_rows: {int(row.not_passed_rows)}")
            lines.append(f"- passed_mean_{horizon_column}: {self._format_float(row.passed_mean_alpha)}")
            lines.append(f"- not_passed_mean_{horizon_column}: {self._format_float(row.not_passed_mean_alpha)}")
            lines.append(f"- alpha_spread: {self._format_float(row.alpha_spread)}")
            lines.append(f"- passed_mean_{raw_return_column}: {self._format_float(row.passed_mean_return)}")
            lines.append(f"- not_passed_mean_{raw_return_column}: {self._format_float(row.not_passed_mean_return)}")
            lines.append("")
        return lines

    def _render_missed_winner_sectors(self, frame: pd.DataFrame, *, horizon_column: str, top: int) -> list[str]:
        lines = ["## Missed Winner Sectors", ""]
        if frame.empty:
            lines.append("No non-passed top-decile winners found.")
            lines.append("")
            return lines
        grouped = (
            frame.groupby("sector", dropna=False)
            .agg(
                missed_winner_rows=("ticker", "count"),
                mean_alpha=(horizon_column, "mean"),
                hit_rate=(horizon_column, lambda series: float((pd.to_numeric(series, errors="coerce") > 0.0).mean())),
                distinct_tickers=("ticker", "nunique"),
            )
            .reset_index()
            .sort_values(["missed_winner_rows", "mean_alpha", "sector"], ascending=[False, False, True])
            .head(top)
        )
        for row in grouped.itertuples(index=False):
            lines.append(f"### {row.sector}")
            lines.append(f"- missed_winner_rows: {int(row.missed_winner_rows)}")
            lines.append(f"- distinct_tickers: {int(row.distinct_tickers)}")
            lines.append(f"- mean_{horizon_column}: {float(row.mean_alpha):.6f}")
            lines.append(f"- hit_rate_{horizon_column}: {float(row.hit_rate):.4f}")
            lines.append("")
        return lines

    def _render_recent_missed_winners(self, frame: pd.DataFrame, *, horizon_column: str, top: int) -> list[str]:
        lines = ["## Recent Missed Winners", ""]
        if frame.empty:
            lines.append("No recent non-passed top-decile winners found.")
            lines.append("")
            return lines
        ranked = frame.sort_values(
            ["snapshot_date", horizon_column, "ticker"],
            ascending=[False, False, True],
        ).head(top)
        raw_return_column = f"fwd_return_{horizon_column.split('_')[-1]}"
        for row in ranked.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- snapshot_date: {pd.Timestamp(row.snapshot_date).date()}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- {horizon_column}: {float(getattr(row, horizon_column)):.6f}")
            lines.append(f"- {raw_return_column}: {self._format_float(getattr(row, raw_return_column, math.nan))}")
            lines.append(f"- relative_strength_index_vs_spy: {self._format_float(getattr(row, 'relative_strength_index_vs_spy', math.nan))}")
            lines.append(f"- roc_63: {self._format_float(getattr(row, 'roc_63', math.nan))}")
            lines.append(f"- vol_alpha: {self._format_float(getattr(row, 'vol_alpha', math.nan))}")
            lines.append("")
        return lines

    def _render_feature_profile_comparison(self, passed_winners: pd.DataFrame, missed_winners: pd.DataFrame) -> list[str]:
        lines = ["## Feature Profile Comparison", ""]
        if passed_winners.empty or missed_winners.empty:
            lines.append("Not enough rows to compare passed winners against missed winners.")
            lines.append("")
            return lines
        lines.append(f"- passed_winner_rows: {len(passed_winners.index)}")
        lines.append(f"- missed_winner_rows: {len(missed_winners.index)}")
        lines.append("")
        for column in FEATURE_PROFILE_COLUMNS:
            if column not in passed_winners.columns or column not in missed_winners.columns:
                continue
            passed_series = pd.to_numeric(passed_winners[column], errors="coerce").dropna()
            missed_series = pd.to_numeric(missed_winners[column], errors="coerce").dropna()
            if passed_series.empty or missed_series.empty:
                continue
            lines.append(f"### {column}")
            lines.append(f"- passed_winner_mean: {passed_series.mean():.6f}")
            lines.append(f"- missed_winner_mean: {missed_series.mean():.6f}")
            lines.append(f"- mean_delta_missed_minus_passed: {(missed_series.mean() - passed_series.mean()):.6f}")
            lines.append("")
        return lines

    def _format_float(self, value: float) -> str:
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
