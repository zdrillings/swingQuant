from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd

from src.scan.ranker import CandidateRanker
from src.scan.service import ScanPolicy, ScanService
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector


OUTCOME_HORIZONS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class ScanAnalysisReport:
    output_path: str
    scan_date: str
    candidate_count: int
    selected_count: int
    refreshed: bool


class ScanAnalysisService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("scan_analysis")

    def run(
        self,
        *,
        scan_date: str | None = None,
        refresh: bool = False,
        horizons: tuple[int, ...] = (5, 10, 20),
    ) -> ScanAnalysisReport:
        self.db_manager.initialize()
        if refresh:
            ScanService(self.db_manager, email_sender=lambda subject, html_body, settings: None).run(dry_run=True)

        all_candidates = self.db_manager.load_scan_candidates()
        if all_candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan` or `sq scan-analysis --refresh` first.")
        all_candidates = self._backfill_outcomes(all_candidates)
        selected_scan_date = scan_date or str(all_candidates["scan_date"].max())
        candidates = all_candidates[all_candidates["scan_date"] == selected_scan_date].copy().reset_index(drop=True)
        if candidates.empty:
            raise ValueError(f"No scan snapshots found for scan_date={selected_scan_date}.")
        selected = candidates[candidates["selected"] == 1].copy()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        candidates["details"] = candidates["details_json"].map(lambda raw: json.loads(raw) if raw else {})
        all_candidates["details"] = all_candidates["details_json"].map(lambda raw: json.loads(raw) if raw else {})
        ranker_report = CandidateRanker(target_column="alpha_vs_sector_10d").evaluate(
            all_candidates,
            scan_policy=scan_policy,
            top_n=scan_policy.max_candidates_total,
        )
        current_ranker_review = self._build_current_ranker_review(
            all_candidates=all_candidates,
            candidates=candidates,
            scan_policy=scan_policy,
            target_column="alpha_vs_sector_10d",
        )
        slot_level_attribution = self._build_slot_level_selector_attribution(
            all_candidates=all_candidates,
            scan_policy=scan_policy,
            target_column="alpha_vs_sector_10d",
        )

        lines = [
            "# Scan Analysis",
            "",
            f"- scan_date: {selected_scan_date}",
            f"- refreshed: {'yes' if refresh else 'no'}",
            f"- candidate_count: {len(candidates.index)}",
            f"- selected_count: {len(selected.index)}",
            f"- min_opportunity_score: {scan_policy.min_opportunity_score:.2f}",
            "",
        ]
        lines.extend(self._render_selection_summary(candidates, selected))
        lines.extend(self._render_threshold_diagnostics(candidates, scan_policy))
        lines.extend(self._render_slot_summary(candidates))
        lines.extend(self._render_owned_strength_watchlist(candidates, scan_policy))
        lines.extend(self._render_learned_buy_review(current_ranker_review, scan_policy))
        lines.extend(self._render_slot_level_selector_attribution(slot_level_attribution))
        lines.extend(self._render_selected_candidates(selected))
        lines.extend(self._render_excluded_candidates(candidates, scan_policy))
        lines.extend(self._render_outcome_coverage(candidates, horizons))
        lines.extend(self._render_forward_attribution(candidates, selected, horizons))
        lines.extend(self._render_ranker_validation(ranker_report))

        report_path = self.db_manager.paths.reports_dir / "scan_analysis.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return ScanAnalysisReport(
            output_path=str(report_path),
            scan_date=selected_scan_date,
            candidate_count=len(candidates.index),
            selected_count=len(selected.index),
            refreshed=refresh,
        )

    def _backfill_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        tickers = sorted(set(frame["ticker"].astype(str)).union({"SPY"}))
        sector_benchmarks = {
            benchmark_etf_for_sector(str(row["sector"] or row["strategy_sector"]))
            for row in frame.to_dict(orient="records")
            if benchmark_etf_for_sector(str(row["sector"] or row["strategy_sector"])) is not None
        }
        tickers = sorted(set(tickers).union(sector_benchmarks))
        try:
            history = self.db_manager.load_price_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load price history for scan attribution: %s", exc)
            return frame
        if history.empty:
            return frame
        history_context = self._history_context(history)
        updater = getattr(self.db_manager, "update_scan_candidate_outcomes", None)
        enriched_frames: list[pd.DataFrame] = []
        for current_scan_date, group in frame.groupby("scan_date", sort=False):
            outcome_rows = [
                self._candidate_outcomes_for_row(row, history_context=history_context)
                for row in group.to_dict(orient="records")
            ]
            outcome_rows = [row for row in outcome_rows if row is not None]
            if callable(updater) and outcome_rows:
                updater(scan_date=str(current_scan_date), rows=outcome_rows)
            if not outcome_rows:
                enriched_frames.append(group.copy())
                continue
            outcome_frame = pd.DataFrame(outcome_rows)
            merged = group.merge(
                outcome_frame,
                on=["ticker", "strategy_slot"],
                how="left",
                suffixes=("", "_new"),
            )
            for column in [
                "fwd_return_1d",
                "fwd_return_3d",
                "fwd_return_5d",
                "fwd_return_10d",
                "fwd_return_20d",
                "alpha_vs_spy_1d",
                "alpha_vs_spy_3d",
                "alpha_vs_spy_5d",
                "alpha_vs_spy_10d",
                "alpha_vs_spy_20d",
                "alpha_vs_sector_1d",
                "alpha_vs_sector_3d",
                "alpha_vs_sector_5d",
                "alpha_vs_sector_10d",
                "alpha_vs_sector_20d",
                "mfe_20d",
                "mae_20d",
            ]:
                new_column = f"{column}_new"
                if new_column not in merged.columns:
                    continue
                merged[column] = merged[new_column]
                merged = merged.drop(columns=[new_column])
            enriched_frames.append(merged)
        return pd.concat(enriched_frames, ignore_index=True) if enriched_frames else frame

    def _history_context(self, history: pd.DataFrame) -> dict[str, dict[str, object]]:
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        context: dict[str, dict[str, object]] = {}
        for ticker, group in working.groupby("ticker", sort=False):
            ordered = group.sort_values("date").reset_index(drop=True)
            context[str(ticker)] = {
                "frame": ordered,
                "index_by_date": {
                    pd.Timestamp(date_value).normalize().strftime("%Y-%m-%d"): int(index)
                    for index, date_value in enumerate(ordered["date"])
                },
            }
        return context

    def _candidate_outcomes_for_row(
        self,
        row: dict,
        *,
        history_context: dict[str, dict[str, object]],
    ) -> dict[str, object] | None:
        ticker = str(row["ticker"])
        scan_date = str(row["scan_date"])
        ticker_context = history_context.get(ticker)
        if ticker_context is None:
            return None
        ticker_frame = ticker_context["frame"]
        index_by_date = ticker_context["index_by_date"]
        index = index_by_date.get(scan_date)
        if index is None:
            return None
        benchmark_ticker = benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
        payload: dict[str, object] = {
            "ticker": ticker,
            "strategy_slot": str(row["strategy_slot"]),
        }
        for horizon in OUTCOME_HORIZONS:
            payload[f"fwd_return_{horizon}d"] = self._forward_return(
                ticker_frame=ticker_frame,
                index=int(index),
                horizon=int(horizon),
            )
            payload[f"alpha_vs_spy_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                scan_date=scan_date,
                index=int(index),
                horizon=int(horizon),
                benchmark_ticker="SPY",
            )
            payload[f"alpha_vs_sector_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                scan_date=scan_date,
                index=int(index),
                horizon=int(horizon),
                benchmark_ticker=benchmark_ticker,
            )
        payload["mfe_20d"] = self._excursion(
            ticker_frame=ticker_frame,
            index=int(index),
            horizon=20,
            column="high",
            use_max=True,
        )
        payload["mae_20d"] = self._excursion(
            ticker_frame=ticker_frame,
            index=int(index),
            horizon=20,
            column="low",
            use_max=False,
        )
        return payload

    def _forward_return(self, *, ticker_frame: pd.DataFrame, index: int, horizon: int) -> float:
        future_index = index + int(horizon)
        if future_index >= len(ticker_frame.index):
            return float("nan")
        entry_price = float(ticker_frame.loc[index, "adj_close"])
        future_price = float(ticker_frame.loc[future_index, "adj_close"])
        return (future_price / entry_price) - 1.0

    def _alpha_vs_benchmark(
        self,
        *,
        history_context: dict[str, dict[str, object]],
        ticker_frame: pd.DataFrame,
        scan_date: str,
        index: int,
        horizon: int,
        benchmark_ticker: str | None,
    ) -> float:
        raw_return = self._forward_return(ticker_frame=ticker_frame, index=index, horizon=horizon)
        if benchmark_ticker in (None, "") or not math.isfinite(raw_return):
            return float("nan")
        benchmark_context = history_context.get(str(benchmark_ticker))
        if benchmark_context is None:
            return float("nan")
        benchmark_index = benchmark_context["index_by_date"].get(scan_date)
        if benchmark_index is None:
            return float("nan")
        benchmark_return = self._forward_return(
            ticker_frame=benchmark_context["frame"],
            index=int(benchmark_index),
            horizon=horizon,
        )
        if not math.isfinite(benchmark_return):
            return float("nan")
        return raw_return - benchmark_return

    def _excursion(
        self,
        *,
        ticker_frame: pd.DataFrame,
        index: int,
        horizon: int,
        column: str,
        use_max: bool,
    ) -> float:
        start_index = index + 1
        end_index = min(index + int(horizon), len(ticker_frame.index) - 1)
        if start_index > end_index:
            return float("nan")
        window = ticker_frame.loc[start_index:end_index, column].astype(float)
        if window.empty:
            return float("nan")
        entry_price = float(ticker_frame.loc[index, "adj_close"])
        extreme_price = float(window.max() if use_max else window.min())
        return (extreme_price / entry_price) - 1.0

    def _render_selection_summary(self, candidates: pd.DataFrame, selected: pd.DataFrame) -> list[str]:
        lines = ["## Selection Summary", ""]
        lines.extend(self._render_group_stats("Eligible universe", candidates))
        lines.extend(self._render_group_stats("Selected ideas", selected))
        if not selected.empty:
            top_ranked = candidates.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            ).head(len(selected.index)).copy()
            lines.extend(self._render_group_stats("Top raw opportunity ideas", top_ranked))
        lines.append("")
        return lines

    def _render_group_stats(self, title: str, frame: pd.DataFrame) -> list[str]:
        lines = [f"### {title}"]
        if frame.empty:
            lines.append("- count: 0")
            lines.append("")
            return lines
        lines.append(f"- count: {len(frame.index)}")
        for column in [
            "opportunity_score",
            "signal_score",
            "setup_quality_score",
            "expected_alpha_score",
            "freshness_score",
            "breadth_score",
            "overlap_penalty",
        ]:
            if column not in frame.columns:
                continue
            values = frame[column].dropna().astype(float)
            if values.empty:
                continue
            lines.append(f"- avg_{column}: {values.mean():.4f}")
        lines.append("")
        return lines

    def _render_slot_summary(self, candidates: pd.DataFrame) -> list[str]:
        lines = ["## Slot Summary", ""]
        grouped = (
            candidates.groupby(["strategy_slot", "strategy_sector"], dropna=False)
            .agg(
                eligible_count=("ticker", "count"),
                selected_count=("selected", "sum"),
                avg_opportunity=("opportunity_score", "mean"),
                avg_signal_score=("signal_score", "mean"),
            )
            .reset_index()
            .sort_values(["selected_count", "avg_opportunity", "strategy_slot"], ascending=[False, False, True])
        )
        if grouped.empty:
            lines.append("No slot data.")
            lines.append("")
            return lines
        for row in grouped.itertuples(index=False):
            lines.append(f"### {row.strategy_slot}")
            lines.append(f"- strategy_sector: {row.strategy_sector}")
            lines.append(f"- eligible_count: {int(row.eligible_count)}")
            lines.append(f"- selected_count: {int(row.selected_count)}")
            lines.append(f"- avg_opportunity: {float(row.avg_opportunity):.4f}")
            lines.append(f"- avg_signal_score: {float(row.avg_signal_score):.4f}")
            lines.append("")
        return lines

    def _render_threshold_diagnostics(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> list[str]:
        lines = ["## Threshold Diagnostics", ""]
        if candidates.empty:
            lines.append("No candidates available.")
            lines.append("")
            return lines
        threshold = float(scan_policy.min_opportunity_score)
        diagnostics = candidates.copy()
        diagnostics["pre_penalty_opportunity_score"] = (
            diagnostics["opportunity_score"].astype(float) + diagnostics["overlap_penalty"].astype(float)
        )
        details_series = diagnostics["details_json"].map(lambda raw: json.loads(raw) if raw else {})
        diagnostics["already_owned"] = details_series.map(lambda payload: bool(payload.get("already_owned", False)))
        pre_count = int((diagnostics["pre_penalty_opportunity_score"] >= threshold).sum())
        post_count = int((diagnostics["opportunity_score"].astype(float) >= threshold).sum())
        overlap_blocked = diagnostics[
            (diagnostics["pre_penalty_opportunity_score"] >= threshold)
            & (diagnostics["opportunity_score"].astype(float) < threshold)
            & (diagnostics["overlap_penalty"].astype(float) > 0.0)
        ].copy()
        lines.append(f"- threshold: {threshold:.2f}")
        lines.append(f"- count_above_threshold_before_overlap: {pre_count}")
        lines.append(f"- count_above_threshold_after_overlap: {post_count}")
        lines.append(f"- overlap_blocked_count: {len(overlap_blocked.index)}")
        lines.append(f"- already_owned_count: {int(diagnostics['already_owned'].sum())}")
        lines.append("")
        if overlap_blocked.empty:
            return lines
        lines.append("### Top Overlap-Blocked Candidates")
        ranked = overlap_blocked.sort_values(
            ["pre_penalty_opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        ).head(10)
        for row in ranked.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- slot: {row.strategy_slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- pre_penalty_opportunity_score: {float(row.pre_penalty_opportunity_score):.4f}")
            lines.append(f"- overlap_penalty: {float(row.overlap_penalty):.4f}")
            lines.append(f"- post_penalty_opportunity_score: {float(row.opportunity_score):.4f}")
            lines.append("")
        return lines

    def _render_selected_candidates(self, selected: pd.DataFrame) -> list[str]:
        lines = ["## Selected Candidates", ""]
        if selected.empty:
            lines.append("No candidates were selected.")
            lines.append("")
            return lines
        sort_columns = []
        ascending = []
        if "selected_rank" in selected.columns:
            sort_columns.append("selected_rank")
            ascending.append(True)
        sort_columns.extend(["opportunity_score", "signal_score", "ticker"])
        ascending.extend([False, False, True])
        ordered = selected.sort_values(sort_columns, ascending=ascending)
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            if pd.notna(getattr(row, "selected_rank", np.nan)):
                lines.append(f"- selected_rank: {int(row.selected_rank)}")
            lines.append(f"- slot: {row.strategy_slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- opportunity_score: {float(row.opportunity_score):.4f}")
            lines.append(f"- signal_score: {float(row.signal_score):.4f}")
            lines.append(f"- setup_quality_score: {float(row.setup_quality_score):.4f}")
            lines.append(f"- expected_alpha_score: {float(row.expected_alpha_score):.4f}")
            lines.append(f"- freshness_score: {float(row.freshness_score):.4f}")
            lines.append(f"- breadth_score: {float(row.breadth_score):.4f}")
            lines.append(f"- overlap_penalty: {float(row.overlap_penalty):.4f}")
            lines.append("")
        return lines

    def _render_owned_strength_watchlist(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> list[str]:
        lines = ["## Owned Strength Watchlist", ""]
        if candidates.empty:
            lines.append("No candidates available.")
            lines.append("")
            return lines
        owned = candidates[candidates["details"].map(lambda payload: bool(payload.get("already_owned", False)))].copy()
        if owned.empty:
            lines.append("No owned names matched the current setup filters.")
            lines.append("")
            return lines
        owned["pre_penalty_opportunity_score"] = owned["details"].map(
            lambda payload: float(payload.get("pre_penalty_opportunity_score", 0.0))
        )
        ranked = owned.sort_values(
            ["pre_penalty_opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        )
        strong_owned = ranked[ranked["pre_penalty_opportunity_score"] >= float(scan_policy.min_opportunity_score)].copy()
        if strong_owned.empty:
            lines.append("Owned names matched, but none cleared the buy threshold before ownership penalties.")
            lines.append("")
            return lines
        for row in strong_owned.itertuples(index=False):
            details = row.details if isinstance(row.details, dict) else {}
            lines.append(f"### {row.ticker}")
            lines.append("- status: already owned, setup still valid")
            lines.append(f"- slot: {row.strategy_slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- pre_penalty_opportunity_score: {float(details.get('pre_penalty_opportunity_score', 0.0)):.4f}")
            lines.append(f"- post_penalty_opportunity_score: {float(row.opportunity_score):.4f}")
            lines.append(f"- overlap_penalty: {float(row.overlap_penalty):.4f}")
            lines.extend(self._render_overlap_component_lines(details))
            lines.append("")
        return lines

    def _render_excluded_candidates(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> list[str]:
        lines = ["## Best Excluded Candidates", ""]
        excluded = candidates[candidates["selected"] == 0].copy()
        if excluded.empty:
            lines.append("No excluded candidates.")
            lines.append("")
            return lines
        excluded = excluded.sort_values(
            ["opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        ).head(10)
        for row in excluded.itertuples(index=False):
            details = row.details if isinstance(row.details, dict) else {}
            reason = self._candidate_exclusion_reason(row, scan_policy=scan_policy, details=details)
            lines.append(f"### {row.ticker}")
            lines.append(f"- slot: {row.strategy_slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- opportunity_score: {float(row.opportunity_score):.4f}")
            lines.append(f"- signal_score: {float(row.signal_score):.4f}")
            lines.append(f"- overlap_penalty: {float(row.overlap_penalty):.4f}")
            lines.append(f"- exclusion_reason: {reason}")
            lines.extend(self._render_overlap_component_lines(details))
            lines.append("")
        return lines

    def _render_overlap_component_lines(self, details: dict) -> list[str]:
        overlap_components = details.get("overlap_components", {})
        if not isinstance(overlap_components, dict):
            return []
        active = [
            f"{name}={float(value):.2f}"
            for name, value in overlap_components.items()
            if float(value) > 0.0
        ]
        if not active:
            return []
        return [f"- overlap_components: {', '.join(active)}"]

    def _render_outcome_coverage(self, candidates: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
        lines = ["## Outcome Coverage", ""]
        if candidates.empty:
            lines.append("No candidates available.")
            lines.append("")
            return lines
        for horizon in horizons:
            raw_count = int(candidates[f"fwd_return_{horizon}d"].notna().sum()) if f"fwd_return_{horizon}d" in candidates.columns else 0
            sector_count = int(candidates[f"alpha_vs_sector_{horizon}d"].notna().sum()) if f"alpha_vs_sector_{horizon}d" in candidates.columns else 0
            spy_count = int(candidates[f"alpha_vs_spy_{horizon}d"].notna().sum()) if f"alpha_vs_spy_{horizon}d" in candidates.columns else 0
            lines.append(f"- {horizon}d: forward={raw_count} sector_alpha={sector_count} spy_alpha={spy_count}")
        if "mfe_20d" in candidates.columns:
            lines.append(f"- mfe_20d_available: {int(candidates['mfe_20d'].notna().sum())}")
        if "mae_20d" in candidates.columns:
            lines.append(f"- mae_20d_available: {int(candidates['mae_20d'].notna().sum())}")
        lines.append("")
        return lines

    def _render_forward_attribution(
        self,
        candidates: pd.DataFrame,
        selected: pd.DataFrame,
        horizons: tuple[int, ...],
    ) -> list[str]:
        lines = ["## Forward Attribution", ""]
        if candidates.empty:
            lines.append("No candidates available.")
            lines.append("")
            return lines
        if not selected.empty:
            top_ranked = candidates.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            ).head(len(selected.index)).copy()
        else:
            top_ranked = candidates.iloc[0:0].copy()
        available = False
        for horizon in horizons:
            raw_column = f"fwd_return_{horizon}d"
            sector_column = f"alpha_vs_sector_{horizon}d"
            spy_column = f"alpha_vs_spy_{horizon}d"
            if raw_column in candidates.columns and candidates[raw_column].notna().any():
                available = True
                lines.append(f"### {horizon}-Day Forward Return")
                lines.extend(self._render_return_group("Eligible universe", candidates, raw_column))
                lines.extend(self._render_return_group("Selected ideas", selected, raw_column))
                lines.extend(self._render_return_group("Top raw opportunity ideas", top_ranked, raw_column))
                lines.append("")
            if sector_column in candidates.columns and candidates[sector_column].notna().any():
                lines.append(f"### {horizon}-Day Sector Alpha")
                lines.extend(self._render_return_group("Eligible universe", candidates, sector_column))
                lines.extend(self._render_return_group("Selected ideas", selected, sector_column))
                lines.extend(self._render_return_group("Top raw opportunity ideas", top_ranked, sector_column))
                lines.append("")
            if spy_column in candidates.columns and candidates[spy_column].notna().any():
                lines.append(f"### {horizon}-Day SPY Alpha")
                lines.extend(self._render_return_group("Eligible universe", candidates, spy_column))
                lines.extend(self._render_return_group("Selected ideas", selected, spy_column))
                lines.extend(self._render_return_group("Top raw opportunity ideas", top_ranked, spy_column))
                lines.append("")
        if not available:
            lines.append("No forward return windows are fully available yet for this scan date.")
            lines.append("")
        return lines

    def _render_return_group(self, title: str, frame: pd.DataFrame, column: str) -> list[str]:
        lines = [f"- {title}:"]
        if frame.empty or column not in frame.columns:
            lines.append("  unavailable")
            return lines
        series = frame[column].dropna().astype(float)
        if series.empty:
            lines.append("  unavailable")
            return lines
        lines.append(f"  mean={series.mean():.4f} median={series.median():.4f} n={len(series.index)}")
        return lines

    def _render_ranker_validation(self, report) -> list[str]:
        lines = ["## Candidate Ranker Validation", ""]
        if not report.available:
            lines.append("Not enough labeled historical scan candidates yet to validate the ranker.")
            lines.append(f"- target_column: {report.target_column}")
            lines.append(f"- train_rows: {report.train_rows}")
            lines.append(f"- validation_rows: {report.validation_rows}")
            lines.append(f"- train_dates: {report.train_dates}")
            lines.append(f"- validation_dates: {report.validation_dates}")
            lines.append("")
            return lines
        lines.append(f"- target_column: {report.target_column}")
        lines.append(f"- train_rows: {report.train_rows}")
        lines.append(f"- validation_rows: {report.validation_rows}")
        lines.append(f"- train_dates: {report.train_dates}")
        lines.append(f"- validation_dates: {report.validation_dates}")
        lines.append(f"- feature_count: {report.feature_count}")
        lines.append(f"- prediction_correlation: {report.prediction_correlation:.6f}")
        lines.append("")
        lines.append("### Validation Averages")
        lines.append(f"- learned_mean_target: {report.learned_mean_target:.6f}")
        lines.append(f"- learned_hit_rate: {report.learned_hit_rate:.6f}")
        lines.append(f"- handcrafted_mean_target: {report.handcrafted_mean_target:.6f}")
        lines.append(f"- handcrafted_hit_rate: {report.handcrafted_hit_rate:.6f}")
        lines.append(f"- runtime_mean_target: {report.runtime_mean_target:.6f}")
        lines.append(f"- runtime_hit_rate: {report.runtime_hit_rate:.6f}")
        lines.append("")
        if report.latest_scan_date is not None:
            lines.append(f"### Latest Validation Date: {report.latest_scan_date}")
            lines.append(f"- learned_tickers: {', '.join(report.latest_learned_tickers) if report.latest_learned_tickers else 'none'}")
            lines.append(f"- handcrafted_tickers: {', '.join(report.latest_handcrafted_tickers) if report.latest_handcrafted_tickers else 'none'}")
            lines.append(f"- runtime_tickers: {', '.join(report.latest_runtime_tickers) if report.latest_runtime_tickers else 'none'}")
            lines.append("")
        return lines

    def _build_current_ranker_review(
        self,
        *,
        all_candidates: pd.DataFrame,
        candidates: pd.DataFrame,
        scan_policy: ScanPolicy,
        target_column: str,
    ) -> dict[str, object]:
        selected_scan_date = str(candidates["scan_date"].iloc[0]) if not candidates.empty else None
        if selected_scan_date is None:
            return {"available": False, "reason": "No scan date available."}
        ranker = CandidateRanker(target_column=target_column)
        if target_column not in all_candidates.columns:
            return {"available": False, "reason": f"Missing target column: {target_column}"}
        historical = all_candidates[
            pd.to_datetime(all_candidates["scan_date"]).dt.normalize() < pd.Timestamp(selected_scan_date).normalize()
        ].copy()
        labeled = historical.dropna(subset=[target_column]).copy()
        train_rows = len(labeled.index)
        train_dates = int(labeled["scan_date"].nunique()) if not labeled.empty else 0
        if train_rows < ranker.min_train_rows or train_dates < ranker.min_train_dates:
            return {
                "available": False,
                "reason": "Not enough prior labeled scan history.",
                "train_rows": train_rows,
                "train_dates": train_dates,
                "target_column": target_column,
            }
        ranker.fit(labeled)
        scored = ranker.score_details(candidates, top_features=3)
        scored = scored.sort_values(
            ["ranker_score", "opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        scored["learned_rank"] = np.arange(1, len(scored.index) + 1, dtype=int)
        scored["runtime_selected"] = scored["selected"].astype(int) == 1
        learned_selected = ranker._select_candidates(
            scored,
            scan_policy=scan_policy,
            score_column="ranker_score",
            top_n=scan_policy.max_candidates_total,
        ).copy()
        learned_keys = {(str(row.ticker), str(row.strategy_slot)) for row in learned_selected.itertuples(index=False)}
        scored["learned_selected"] = [
            (str(row["ticker"]), str(row["strategy_slot"])) in learned_keys
            for row in scored.to_dict(orient="records")
        ]
        learned_selected = scored[scored["learned_selected"]].copy()
        learned_only = scored[(scored["learned_selected"]) & (~scored["runtime_selected"])].copy()
        runtime_only = scored[(scored["runtime_selected"]) & (~scored["learned_selected"])].copy()
        overlap_count = int(
            scored[(scored["runtime_selected"]) & (scored["learned_selected"])].shape[0]
        )
        selected_scores = scored.loc[scored["runtime_selected"], "ranker_score"].dropna().astype(float)
        rejected_scores = scored.loc[~scored["runtime_selected"], "ranker_score"].dropna().astype(float)
        return {
            "available": True,
            "target_column": target_column,
            "train_rows": train_rows,
            "train_dates": train_dates,
            "scored": scored,
            "learned_selected": learned_selected,
            "learned_only": learned_only,
            "runtime_only": runtime_only,
            "selected_mean_score": float(selected_scores.mean()) if not selected_scores.empty else float("nan"),
            "rejected_mean_score": float(rejected_scores.mean()) if not rejected_scores.empty else float("nan"),
            "overlap_count": overlap_count,
            "runtime_selected_count": int(scored["runtime_selected"].sum()),
            "learned_selected_count": int(scored["learned_selected"].sum()),
        }

    def _render_learned_buy_review(self, review: dict[str, object], scan_policy: ScanPolicy) -> list[str]:
        lines = ["## Learned Buy Review", ""]
        if not bool(review.get("available", False)):
            lines.append(str(review.get("reason", "Not enough historical scan data to score the current date.")))
            if "train_rows" in review:
                lines.append(f"- train_rows: {int(review.get('train_rows', 0))}")
            if "train_dates" in review:
                lines.append(f"- train_dates: {int(review.get('train_dates', 0))}")
            if "target_column" in review:
                lines.append(f"- target_column: {review.get('target_column')}")
            lines.append("")
            return lines
        scored = review["scored"]
        learned_selected = review["learned_selected"]
        learned_only = review["learned_only"]
        runtime_only = review["runtime_only"]
        lines.append(f"- target_column: {review['target_column']}")
        lines.append(f"- train_rows: {int(review['train_rows'])}")
        lines.append(f"- train_dates: {int(review['train_dates'])}")
        lines.append(f"- learned_selected_count: {int(review['learned_selected_count'])}")
        lines.append(f"- runtime_selected_count: {int(review['runtime_selected_count'])}")
        lines.append(
            f"- overlap_with_runtime_selected: {int(review['overlap_count'])}/{int(review['runtime_selected_count'])}"
        )
        selected_mean_score = float(review["selected_mean_score"])
        rejected_mean_score = float(review["rejected_mean_score"])
        if math.isfinite(selected_mean_score):
            lines.append(f"- runtime_selected_mean_predicted_alpha: {selected_mean_score:.6f}")
        if math.isfinite(rejected_mean_score):
            lines.append(f"- runtime_rejected_mean_predicted_alpha: {rejected_mean_score:.6f}")
        lines.append("")
        lines.extend(self._render_ranker_candidate_block("Top Learned Buys", learned_selected, scan_policy=scan_policy))
        lines.extend(
            self._render_ranker_candidate_block(
                "Best Learned Rejections",
                learned_only.head(5).copy(),
                scan_policy=scan_policy,
            )
        )
        lines.extend(
            self._render_ranker_candidate_block(
                "Runtime Overrides",
                runtime_only.sort_values(
                    ["selected_rank", "opportunity_score", "ticker"],
                    ascending=[True, False, True],
                ).head(5).copy(),
                scan_policy=scan_policy,
            )
        )
        return lines

    def _render_ranker_candidate_block(
        self,
        title: str,
        frame: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
    ) -> list[str]:
        lines = [f"### {title}"]
        if frame.empty:
            lines.append("- none")
            lines.append("")
            return lines
        ordered = frame.sort_values(
            ["learned_rank", "ranker_score", "opportunity_score", "ticker"],
            ascending=[True, False, False, True],
        )
        for row in ordered.itertuples(index=False):
            details = row.details if isinstance(row.details, dict) else {}
            lines.append(f"#### {row.ticker}")
            if pd.notna(getattr(row, "learned_rank", np.nan)):
                lines.append(f"- learned_rank: {int(row.learned_rank)}")
            if pd.notna(getattr(row, "selected_rank", np.nan)):
                lines.append(f"- runtime_selected_rank: {int(row.selected_rank)}")
            lines.append(f"- runtime_selected: {'yes' if bool(getattr(row, 'runtime_selected', False)) else 'no'}")
            lines.append(f"- slot: {row.strategy_slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- predicted_alpha_vs_sector_10d: {float(row.ranker_score):.6f}")
            lines.append(f"- opportunity_score: {float(row.opportunity_score):.4f}")
            lines.append(f"- signal_score: {float(row.signal_score):.4f}")
            exclusion_reason = self._candidate_exclusion_reason(row, scan_policy=scan_policy, details=details)
            if not bool(getattr(row, "runtime_selected", False)):
                lines.append(f"- runtime_exclusion_reason: {exclusion_reason}")
            positive = tuple(getattr(row, "ranker_top_positive_reasons", ()) or ())
            negative = tuple(getattr(row, "ranker_top_negative_reasons", ()) or ())
            if positive:
                lines.append(f"- ranker_positive_reasons: {', '.join(positive)}")
            if negative:
                lines.append(f"- ranker_negative_reasons: {', '.join(negative)}")
            lines.extend(self._render_overlap_component_lines(details))
            lines.append("")
        return lines

    def _candidate_exclusion_reason(self, row, *, scan_policy: ScanPolicy, details: dict) -> str:
        already_owned = bool(details.get("already_owned", False))
        if already_owned:
            if float(details.get("pre_penalty_opportunity_score", 0.0)) >= scan_policy.min_opportunity_score:
                return "already owned (setup still valid)"
            return "already owned"
        if float(row.opportunity_score) < scan_policy.min_opportunity_score:
            return "below opportunity threshold"
        return "portfolio cap / overlap selection loss"

    def _build_slot_level_selector_attribution(
        self,
        *,
        all_candidates: pd.DataFrame,
        scan_policy: ScanPolicy,
        target_column: str,
    ) -> dict[str, object]:
        ranker = CandidateRanker(target_column=target_column)
        if target_column not in all_candidates.columns:
            return {"available": False, "reason": f"Missing target column: {target_column}"}
        labeled = all_candidates.dropna(subset=[target_column]).copy()
        if labeled.empty:
            return {"available": False, "reason": "No labeled candidate history available.", "target_column": target_column}
        train_frame, validation_frame = ranker._chronological_split(labeled, train_ratio=0.7)
        train_rows = len(train_frame.index)
        train_dates = int(train_frame["scan_date"].nunique()) if not train_frame.empty else 0
        validation_dates = int(validation_frame["scan_date"].nunique()) if not validation_frame.empty else 0
        if (
            train_frame.empty
            or validation_frame.empty
            or train_rows < ranker.min_train_rows
            or train_dates < ranker.min_train_dates
        ):
            return {
                "available": False,
                "reason": "Not enough validation history for slot-level selector attribution.",
                "target_column": target_column,
                "train_rows": train_rows,
                "train_dates": train_dates,
                "validation_dates": validation_dates,
            }
        ranker.fit(train_frame)
        scored = ranker.score(validation_frame)
        rows: list[dict[str, object]] = []
        for (scan_date, slot), day_frame in scored.groupby(["scan_date", "strategy_slot"], sort=True):
            learned = self._slot_select_candidates(day_frame, score_column="ranker_score", scan_policy=scan_policy)
            handcrafted = self._slot_select_candidates(day_frame, score_column="opportunity_score", scan_policy=scan_policy)
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            rows.extend(
                [
                    self._slot_selection_metrics(scan_date, slot, "learned", learned, target_column),
                    self._slot_selection_metrics(scan_date, slot, "handcrafted", handcrafted, target_column),
                    self._slot_selection_metrics(scan_date, slot, "runtime", runtime, target_column),
                ]
            )
        metrics = pd.DataFrame(rows)
        summary_rows: list[dict[str, object]] = []
        for (slot, method), subset in metrics.groupby(["strategy_slot", "method"], sort=True):
            series = subset["mean_target"].dropna().astype(float)
            hit_series = subset["hit_rate"].dropna().astype(float)
            count_series = subset["pick_count"].dropna().astype(float)
            summary_rows.append(
                {
                    "strategy_slot": str(slot),
                    "method": str(method),
                    "mean_target": float(series.mean()) if not series.empty else float("nan"),
                    "hit_rate": float(hit_series.mean()) if not hit_series.empty else float("nan"),
                    "avg_pick_count": float(count_series.mean()) if not count_series.empty else float("nan"),
                    "validation_days": int(subset["scan_date"].nunique()),
                }
            )
        return {
            "available": True,
            "target_column": target_column,
            "train_rows": train_rows,
            "train_dates": train_dates,
            "validation_dates": validation_dates,
            "summary": pd.DataFrame(summary_rows),
        }

    def _render_slot_level_selector_attribution(self, report: dict[str, object]) -> list[str]:
        lines = ["## Slot-Level Selector Attribution", ""]
        if not bool(report.get("available", False)):
            lines.append(str(report.get("reason", "No slot-level selector attribution available.")))
            if "train_rows" in report:
                lines.append(f"- train_rows: {int(report.get('train_rows', 0))}")
            if "train_dates" in report:
                lines.append(f"- train_dates: {int(report.get('train_dates', 0))}")
            if "validation_dates" in report:
                lines.append(f"- validation_dates: {int(report.get('validation_dates', 0))}")
            if "target_column" in report:
                lines.append(f"- target_column: {report.get('target_column')}")
            lines.append("")
            return lines
        lines.append(f"- target_column: {report['target_column']}")
        lines.append(f"- train_rows: {int(report['train_rows'])}")
        lines.append(f"- train_dates: {int(report['train_dates'])}")
        lines.append(f"- validation_dates: {int(report['validation_dates'])}")
        lines.append("")
        summary = report["summary"]
        if summary.empty:
            lines.append("No slot-level rows available.")
            lines.append("")
            return lines
        method_order = {"learned": 0, "handcrafted": 1, "runtime": 2}
        grouped = summary.copy()
        grouped["method_order"] = grouped["method"].map(method_order).fillna(9).astype(int)
        grouped = grouped.sort_values(["strategy_slot", "method_order", "method"], ascending=[True, True, True])
        for slot, slot_frame in grouped.groupby("strategy_slot", sort=True):
            lines.append(f"### {slot}")
            for row in slot_frame.itertuples(index=False):
                lines.append(
                    f"- {row.method}: mean_target={float(row.mean_target):.6f} "
                    f"hit_rate={float(row.hit_rate):.4f} avg_pick_count={float(row.avg_pick_count):.2f} "
                    f"validation_days={int(row.validation_days)}"
                )
            lines.append("")
        return lines

    def _slot_select_candidates(self, frame: pd.DataFrame, *, score_column: str, scan_policy: ScanPolicy) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        ranked = frame.copy()
        if "md_volume_30d" not in ranked.columns:
            ranked["md_volume_30d"] = 0.0
        if "already_owned" not in ranked.columns:
            if "details_json" in ranked.columns:
                ranked["already_owned"] = ranked["details_json"].map(
                    lambda raw: bool(json.loads(raw).get("already_owned", False)) if raw else False
                )
            else:
                ranked["already_owned"] = False
        ranked = ranked[ranked["opportunity_score"].astype(float) >= float(scan_policy.min_opportunity_score)].copy()
        ranked = ranked.sort_values(
            ["already_owned", score_column, "opportunity_score", "signal_score", "md_volume_30d", "ticker"],
            ascending=[True, False, False, False, False, True],
        )
        ranked = ranked[~ranked["already_owned"].astype(bool)].copy()
        return ranked.head(int(scan_policy.max_candidates_per_slot)).copy()

    def _slot_selection_metrics(
        self,
        scan_date,
        slot: str,
        method: str,
        frame: pd.DataFrame,
        target_column: str,
    ) -> dict[str, object]:
        if frame.empty or target_column not in frame.columns:
            return {
                "scan_date": str(scan_date),
                "strategy_slot": str(slot),
                "method": method,
                "mean_target": np.nan,
                "hit_rate": np.nan,
                "pick_count": 0,
            }
        series = frame[target_column].dropna().astype(float)
        if series.empty:
            return {
                "scan_date": str(scan_date),
                "strategy_slot": str(slot),
                "method": method,
                "mean_target": np.nan,
                "hit_rate": np.nan,
                "pick_count": int(len(frame.index)),
            }
        return {
            "scan_date": str(scan_date),
            "strategy_slot": str(slot),
            "method": method,
            "mean_target": float(series.mean()),
            "hit_rate": float((series > 0.0).mean()),
            "pick_count": int(len(series.index)),
        }
