from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd

from src.scan.ranker import CandidateRanker
from src.scan.service import ScanPolicy, ScanService
from src.settings import load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot
from src.utils.strategy import SIGNAL_SCORE_MIN_KEY, evaluate_signal_gate, load_active_strategies, split_signal_indicators


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
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        all_candidates["details"] = all_candidates["details_json"].map(lambda raw: json.loads(raw) if raw else {})
        all_candidates = self._attach_feature_snapshot_columns(all_candidates)
        all_candidates = self._attach_selection_diagnostics(all_candidates, scan_policy=scan_policy)
        all_candidates = self._attach_regime_context(all_candidates)
        persisted_slot_diagnostics = self._load_persisted_slot_diagnostics(selected_scan_date)
        live_scan_context = None if persisted_slot_diagnostics else self._build_live_scan_snapshot_context()
        live_slot_gate_diagnostics = (
            self._persisted_gate_diagnostics(persisted_slot_diagnostics)
            if persisted_slot_diagnostics
            else self._build_live_slot_gate_diagnostics(live_scan_context)
        )
        live_slot_post_gate_dropoff = (
            self._persisted_post_gate_dropoff(persisted_slot_diagnostics)
            if persisted_slot_diagnostics
            else self._build_live_slot_post_gate_dropoff(
                live_scan_context,
                scan_policy=scan_policy,
            )
        )
        candidates = all_candidates[all_candidates["scan_date"] == selected_scan_date].copy().reset_index(drop=True)
        candidates = self._attach_selection_diagnostics(candidates, scan_policy=scan_policy)
        selected = candidates[candidates["selected"] == 1].copy()
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
        lines.extend(self._render_active_slot_coverage(candidates))
        lines.extend(self._render_live_slot_gate_waterfall(live_slot_gate_diagnostics))
        lines.extend(self._render_live_slot_post_gate_dropoff(live_slot_post_gate_dropoff, scan_policy=scan_policy))
        lines.extend(self._render_slot_summary(candidates))
        lines.extend(self._render_slot_allocation_drivers(candidates))
        lines.extend(self._render_owned_strength_watchlist(candidates, scan_policy))
        lines.extend(self._render_portfolio_strength_coverage(candidates))
        lines.extend(self._render_learned_buy_review(current_ranker_review, scan_policy))
        lines.extend(self._render_slot_level_selector_attribution(slot_level_attribution))
        lines.extend(self._render_selected_candidates(selected))
        lines.extend(self._render_excluded_candidates(candidates, scan_policy))
        lines.extend(self._render_outcome_coverage(candidates, horizons))
        lines.extend(self._render_post_change_selector_maturity(all_candidates, horizons=(1, 5, 10)))
        lines.extend(self._render_recent_early_read(all_candidates, target_column="alpha_vs_sector_1d", recent_dates=5))
        lines.extend(self._render_signal_first_selector_early_read(all_candidates, scan_policy=scan_policy, recent_dates=5, horizons=(1, 5, 10)))
        lines.extend(self._render_forward_attribution(candidates, selected, horizons))
        lines.extend(self._render_recent_selection_mistakes(all_candidates, target_column="alpha_vs_sector_10d"))
        lines.extend(self._render_mediocre_setup_diagnostics(all_candidates, target_column="alpha_vs_sector_10d"))
        lines.extend(self._render_regime_attribution(all_candidates, target_column="alpha_vs_sector_10d", recent_scan_dates=40))
        lines.extend(self._render_slot_internal_attribution(all_candidates, target_column="alpha_vs_sector_10d", recent_scan_dates=40, max_dimensions=6))
        lines.extend(self._render_selector_bakeoff(all_candidates, scan_policy=scan_policy, target_column="alpha_vs_sector_10d", recent_windows=(10, 20, 40)))
        lines.extend(self._render_selector_shadow_comparison(all_candidates, scan_policy=scan_policy, target_column="alpha_vs_sector_10d", recent_dates=10))
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
                try:
                    updater(scan_date=str(current_scan_date), rows=outcome_rows)
                except Exception as exc:
                    self.logger.warning(
                        "Unable to persist scan candidate outcomes for %s; continuing with in-memory analysis only: %s",
                        current_scan_date,
                        exc,
                    )
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

    def _render_active_slot_coverage(self, candidates: pd.DataFrame) -> list[str]:
        lines = ["## Active Slot Coverage", ""]
        active_slots = self._active_slot_map(candidates)
        if not active_slots:
            lines.append("No active slot metadata available.")
            lines.append("")
            return lines
        for slot, sector in active_slots:
            slot_frame = candidates[candidates["strategy_slot"].astype(str) == str(slot)].copy()
            eligible_count = len(slot_frame.index)
            selected_count = int(slot_frame["selected"].astype(int).sum()) if "selected" in slot_frame.columns else 0
            lines.append(f"### {slot}")
            lines.append(f"- strategy_sector: {sector}")
            lines.append(f"- eligible_count: {eligible_count}")
            lines.append(f"- selected_count: {selected_count}")
            if slot_frame.empty:
                lines.append("- status: active slot, but no candidates passed the hard gate and signal threshold")
                lines.append("")
                continue
            if selected_count == 0:
                lines.append("- status: candidates existed, but the slot was crowded out by higher-ranked names")
            elif selected_count < eligible_count:
                lines.append("- status: slot contributed picks, but some eligible names lost in final ranking")
            else:
                lines.append("- status: every eligible name from this slot was selected")
            top_signal = slot_frame.sort_values(
                ["signal_score", "opportunity_score", "ticker"],
                ascending=[False, False, True],
            ).head(3)
            lines.append(
                f"- top_signal_names: {', '.join(top_signal['ticker'].astype(str).tolist()) if not top_signal.empty else 'none'}"
            )
            lines.append("")
        return lines

    def _render_live_slot_gate_waterfall(self, diagnostics: dict[str, dict[str, object]]) -> list[str]:
        lines = ["## Live Slot Gate Waterfall", ""]
        if not diagnostics:
            lines.append("Live gate diagnostics unavailable.")
            lines.append("")
            return lines
        for slot, payload in sorted(diagnostics.items(), key=lambda item: item[0]):
            lines.append(f"### {slot}")
            lines.append(f"- strategy_sector: {payload.get('sector', 'unknown')}")
            counts = payload.get("counts", [])
            if counts:
                counts_rendered = ", ".join(f"{label}={int(value)}" for label, value in counts)
                lines.append(f"- gate_counts: {counts_rendered}")
            else:
                lines.append("- gate_counts: unavailable")
            lines.append(f"- first_zero_gate: {payload.get('first_zero_gate', 'unavailable')}")
            component_counts = payload.get("component_positive_counts", [])
            if component_counts:
                support_rendered = ", ".join(f"{label}={int(value)}" for label, value in component_counts)
                lines.append(f"- component_positive_counts: {support_rendered}")
            else:
                lines.append("- component_positive_counts: unavailable")
            lines.append("")
        return lines

    def _render_live_slot_post_gate_dropoff(
        self,
        dropoff: dict[str, dict[str, object]],
        *,
        scan_policy: ScanPolicy,
    ) -> list[str]:
        lines = ["## Live Post-Gate Dropoff", ""]
        if not dropoff:
            lines.append("Live post-gate diagnostics unavailable.")
            lines.append("")
            return lines
        lines.append(f"- min_opportunity_score: {float(scan_policy.min_opportunity_score):.2f}")
        lines.append("")
        for slot, payload in sorted(dropoff.items(), key=lambda item: item[0]):
            lines.append(f"### {slot}")
            lines.append(f"- strategy_sector: {payload.get('sector', 'unknown')}")
            lines.append(f"- gated_count: {int(payload.get('gated_count', 0))}")
            lines.append(f"- cleared_opportunity_count: {int(payload.get('cleared_count', 0))}")
            lines.append(f"- dropped_after_opportunity_count: {int(payload.get('dropped_count', 0))}")
            avg_opp = payload.get("avg_opportunity_score")
            if avg_opp is not None and math.isfinite(float(avg_opp)):
                lines.append(f"- avg_gated_opportunity_score: {float(avg_opp):.4f}")
            top_cleared = payload.get("top_cleared", [])
            top_dropped = payload.get("top_dropped", [])
            lines.append(
                f"- top_cleared: {', '.join(top_cleared) if top_cleared else 'none'}"
            )
            lines.append(
                f"- top_dropped: {', '.join(top_dropped) if top_dropped else 'none'}"
            )
            drop_examples = payload.get("drop_examples", [])
            if drop_examples:
                for row in drop_examples:
                    lines.append(
                        f"- dropped: {row['ticker']} opportunity={float(row['opportunity_score']):.4f} "
                        f"signal={float(row['signal_score']):.4f}"
                    )
            lines.append("")
        return lines

    def _load_persisted_slot_diagnostics(self, scan_date: str) -> dict[str, dict[str, object]]:
        loader = getattr(self.db_manager, "load_scan_slot_diagnostics", None)
        if not callable(loader):
            return {}
        frame = loader(scan_date)
        if frame is None or frame.empty:
            return {}
        diagnostics: dict[str, dict[str, object]] = {}
        for row in frame.to_dict(orient="records"):
            diagnostics[str(row["strategy_slot"])] = {
                "strategy_sector": str(row.get("strategy_sector", "")),
                "gate_counts": json.loads(row.get("gate_counts_json") or "[]"),
                "first_zero_gate": str(row.get("first_zero_gate", "unavailable")),
                "component_positive_counts": json.loads(row.get("component_positive_counts_json") or "[]"),
                "gated_count": int(row.get("gated_count", 0) or 0),
                "cleared_opportunity_count": int(row.get("cleared_opportunity_count", 0) or 0),
                "dropped_after_opportunity_count": int(row.get("dropped_after_opportunity_count", 0) or 0),
                "avg_gated_opportunity_score": row.get("avg_gated_opportunity_score"),
                "top_cleared": json.loads(row.get("top_cleared_json") or "[]"),
                "top_dropped": json.loads(row.get("top_dropped_json") or "[]"),
                "drop_examples": json.loads(row.get("drop_examples_json") or "[]"),
            }
        return diagnostics

    def _persisted_gate_diagnostics(self, diagnostics: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        return {
            slot: {
                "sector": payload.get("strategy_sector", "unknown"),
                "counts": payload.get("gate_counts", []),
                "first_zero_gate": payload.get("first_zero_gate", "unavailable"),
                "component_positive_counts": payload.get("component_positive_counts", []),
            }
            for slot, payload in diagnostics.items()
        }

    def _persisted_post_gate_dropoff(self, diagnostics: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        return {
            slot: {
                "sector": payload.get("strategy_sector", "unknown"),
                "gated_count": payload.get("gated_count", 0),
                "cleared_count": payload.get("cleared_opportunity_count", 0),
                "dropped_count": payload.get("dropped_after_opportunity_count", 0),
                "avg_opportunity_score": payload.get("avg_gated_opportunity_score"),
                "top_cleared": payload.get("top_cleared", []),
                "top_dropped": payload.get("top_dropped", []),
                "drop_examples": payload.get("drop_examples", []),
            }
            for slot, payload in diagnostics.items()
        }

    def _render_slot_allocation_drivers(self, candidates: pd.DataFrame) -> list[str]:
        lines = ["## Slot Allocation Drivers", ""]
        active_slots = self._active_slot_map(candidates)
        if not active_slots:
            lines.append("No slot allocation data available.")
            lines.append("")
            return lines
        for slot, sector in active_slots:
            slot_frame = candidates[candidates["strategy_slot"].astype(str) == str(slot)].copy()
            lines.append(f"### {slot}")
            lines.append(f"- strategy_sector: {sector}")
            if slot_frame.empty:
                lines.append("- status: no eligible candidates; this slot dropped out before final ranking")
                lines.append("")
                continue
            selected = slot_frame[slot_frame["selected"].astype(int) == 1].copy()
            raw_sorted = slot_frame.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            )
            adjusted_sorted = slot_frame.sort_values(
                ["selection_score", "opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, False, True],
            )
            compare_count = max(len(selected.index), min(3, len(slot_frame.index)))
            raw_leaders = raw_sorted.head(compare_count)
            adjusted_leaders = adjusted_sorted.head(compare_count)
            raw_tickers = raw_leaders["ticker"].astype(str).tolist()
            adjusted_tickers = adjusted_leaders["ticker"].astype(str).tolist()
            lines.append(f"- raw_slot_leaders: {', '.join(raw_tickers) if raw_tickers else 'none'}")
            lines.append(f"- adjusted_slot_leaders: {', '.join(adjusted_tickers) if adjusted_tickers else 'none'}")
            promoted = [ticker for ticker in adjusted_tickers if ticker not in raw_tickers]
            demoted = [ticker for ticker in raw_tickers if ticker not in adjusted_tickers]
            lines.append(f"- overlay_promoted: {', '.join(promoted) if promoted else 'none'}")
            lines.append(f"- overlay_demoted: {', '.join(demoted) if demoted else 'none'}")
            if "slot_overlay_adjustment" in slot_frame.columns:
                avg_overlay = pd.to_numeric(slot_frame["slot_overlay_adjustment"], errors="coerce").dropna()
                if not avg_overlay.empty:
                    lines.append(f"- avg_slot_overlay_adjustment: {float(avg_overlay.mean()):+.4f}")
            if not selected.empty:
                weakest_selected = selected.sort_values(
                    ["selection_score", "opportunity_score", "signal_score", "ticker"],
                    ascending=[True, True, True, True],
                ).head(1)
                strongest_excluded = slot_frame[slot_frame["selected"].astype(int) == 0].sort_values(
                    ["selection_score", "opportunity_score", "signal_score", "ticker"],
                    ascending=[False, False, False, True],
                ).head(1)
                if not weakest_selected.empty and not strongest_excluded.empty:
                    lines.append(
                        f"- cutoff_pair: kept {str(weakest_selected.iloc[0]['ticker'])} over {str(strongest_excluded.iloc[0]['ticker'])}"
                    )
            lines.append("")
        return lines

    def _active_slot_map(self, candidates: pd.DataFrame) -> list[tuple[str, str]]:
        slot_to_sector: dict[str, str] = {}
        try:
            for slot, strategy in load_active_strategies().items():
                slot_to_sector[str(slot)] = str(strategy.sector)
        except Exception:
            pass
        if not candidates.empty:
            observed = (
                candidates[["strategy_slot", "strategy_sector"]]
                .dropna(subset=["strategy_slot"])
                .drop_duplicates()
                .itertuples(index=False)
            )
            for row in observed:
                slot = str(row.strategy_slot)
                sector = str(row.strategy_sector)
                slot_to_sector.setdefault(slot, sector)
        return sorted(slot_to_sector.items(), key=lambda item: item[0])

    def _build_live_scan_snapshot_context(self) -> dict[str, object] | None:
        try:
            strategies = load_active_strategies()
        except Exception:
            return None
        list_universe_rows = getattr(self.db_manager, "list_universe_rows", None)
        load_price_history = getattr(self.db_manager, "load_price_history", None)
        if not callable(list_universe_rows) or not callable(load_price_history):
            return None
        universe_rows = list_universe_rows(active_only=True)
        if not universe_rows:
            return None
        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        try:
            price_history = load_price_history(tickers)
        except Exception:
            return None
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        if analysis_frame.empty:
            return {
                "strategies": strategies,
                "universe_rows": universe_rows,
                "full_snapshot": pd.DataFrame(),
                "regime_snapshot": pd.DataFrame(),
                "overlap_context": {"tickers": set(), "slots": set(), "sectors": set(), "regimes": set()},
            }
        full_snapshot = latest_snapshot(analysis_frame)
        full_snapshot = full_snapshot[full_snapshot["ticker"].isin(universe_tickers)].copy()
        regime_snapshot = full_snapshot[full_snapshot["regime_green"].fillna(False)].copy()
        open_trade_loader = getattr(self.db_manager, "list_open_trades", None)
        open_trades = open_trade_loader() if callable(open_trade_loader) else []
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        overlap_context = ScanService(self.db_manager, email_sender=lambda *args, **kwargs: None)._build_overlap_context(
            open_trades=open_trades,
            strategies=strategies,
            sector_map=sector_map,
        )
        return {
            "strategies": strategies,
            "universe_rows": universe_rows,
            "full_snapshot": full_snapshot,
            "regime_snapshot": regime_snapshot,
            "overlap_context": overlap_context,
        }

    def _build_live_slot_gate_diagnostics(self, context: dict[str, object] | None) -> dict[str, dict[str, object]]:
        if not context:
            return {}
        strategies = context.get("strategies", {})
        full_snapshot = context.get("full_snapshot")
        regime_snapshot = context.get("regime_snapshot")
        if not isinstance(full_snapshot, pd.DataFrame) or not isinstance(regime_snapshot, pd.DataFrame):
            return {
                str(slot): {
                    "sector": strategy.sector,
                    "counts": [],
                    "first_zero_gate": "unavailable",
                    "component_positive_counts": [],
                }
                for slot, strategy in strategies.items()
            }
        if full_snapshot.empty:
            return {
                str(slot): {
                    "sector": strategy.sector,
                    "counts": [("universe", 0), ("regime_green", 0), ("sector_scope", 0), (SIGNAL_SCORE_MIN_KEY, 0)],
                    "first_zero_gate": "sector_scope",
                    "component_positive_counts": [],
                }
                for slot, strategy in strategies.items()
            }
        return {
            str(slot): {
                "sector": strategy.sector,
                **self._build_gate_diagnostic(
                    full_snapshot=full_snapshot,
                    regime_snapshot=regime_snapshot,
                    indicators=strategy.indicators,
                    sector=strategy.sector,
                ),
            }
            for slot, strategy in strategies.items()
        }

    def _build_live_slot_post_gate_dropoff(
        self,
        context: dict[str, object] | None,
        *,
        scan_policy: ScanPolicy,
    ) -> dict[str, dict[str, object]]:
        if not context:
            return {}
        strategies = context.get("strategies", {})
        full_snapshot = context.get("full_snapshot")
        regime_snapshot = context.get("regime_snapshot")
        overlap_context = context.get("overlap_context", {"tickers": set(), "slots": set(), "sectors": set(), "regimes": set()})
        if not isinstance(regime_snapshot, pd.DataFrame):
            return {}
        helper = ScanService(self.db_manager, email_sender=lambda *args, **kwargs: None)
        diagnostics: dict[str, dict[str, object]] = {}
        for slot, strategy in strategies.items():
            if not isinstance(regime_snapshot, pd.DataFrame) or regime_snapshot.empty:
                diagnostics[str(slot)] = {
                    "sector": strategy.sector,
                    "gated_count": 0,
                    "cleared_count": 0,
                    "dropped_count": 0,
                    "avg_opportunity_score": None,
                    "top_cleared": [],
                    "top_dropped": [],
                    "drop_examples": [],
                }
                continue
            scoped_snapshot = helper._scope_snapshot(regime_snapshot, strategy)
            gated = filter_signal_candidates(scoped_snapshot, strategy.indicators)
            if gated.empty:
                diagnostics[str(slot)] = {
                    "sector": strategy.sector,
                    "gated_count": 0,
                    "cleared_count": 0,
                    "dropped_count": 0,
                    "avg_opportunity_score": None,
                    "top_cleared": [],
                    "top_dropped": [],
                    "drop_examples": [],
                }
                continue
            gated = gated.copy()
            gated["strategy_slot"] = str(slot)
            gated["strategy_sector"] = strategy.sector
            scored = pd.DataFrame(
                [
                    helper._score_candidate(
                        row=row,
                        strategy_slot=str(slot),
                        strategy=strategy,
                        scan_policy=scan_policy,
                        overlap_context=overlap_context,
                    )
                    for row in gated.to_dict(orient="records")
                ]
            )
            gated = gated.merge(
                scored,
                on=["ticker", "strategy_slot", "strategy_sector"],
                how="left",
            )
            gated["opportunity_score"] = pd.to_numeric(gated["opportunity_score"], errors="coerce")
            cleared = gated[gated["opportunity_score"] >= float(scan_policy.min_opportunity_score)].copy()
            dropped = gated[gated["opportunity_score"] < float(scan_policy.min_opportunity_score)].copy()
            cleared = cleared.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            )
            dropped = dropped.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            )
            diagnostics[str(slot)] = {
                "sector": strategy.sector,
                "gated_count": len(gated.index),
                "cleared_count": len(cleared.index),
                "dropped_count": len(dropped.index),
                "avg_opportunity_score": float(gated["opportunity_score"].dropna().mean()) if not gated["opportunity_score"].dropna().empty else None,
                "top_cleared": cleared["ticker"].astype(str).head(3).tolist(),
                "top_dropped": dropped["ticker"].astype(str).head(3).tolist(),
                "drop_examples": [
                    {
                        "ticker": str(row["ticker"]),
                        "opportunity_score": float(row["opportunity_score"]),
                        "signal_score": float(row["signal_score"]),
                    }
                    for _, row in dropped.head(3).iterrows()
                ],
            }
        return diagnostics

    def _build_gate_diagnostic(
        self,
        *,
        full_snapshot: pd.DataFrame,
        regime_snapshot: pd.DataFrame,
        indicators: dict[str, float],
        sector: str,
    ) -> dict[str, object]:
        counts: list[tuple[str, int]] = [
            ("universe", len(full_snapshot.index)),
            ("regime_green", len(regime_snapshot.index)),
        ]
        hard_filters, score_components, pass_score = split_signal_indicators(indicators)
        working = regime_snapshot.copy()
        if sector != "ALL":
            working = working[working["sector"] == sector].copy()
        counts.append(("sector_scope", len(working.index)))

        first_zero_gate: str | None = "sector_scope" if len(working.index) == 0 else None
        for threshold_name, threshold_value in hard_filters.items():
            feature_name = threshold_name[:-4] if threshold_name.endswith(("_min", "_max")) else threshold_name
            if feature_name not in working.columns:
                working = working.iloc[0:0].copy()
            elif threshold_name.endswith("_min"):
                mask = working[feature_name].astype(float) >= float(threshold_value)
                working = working.loc[mask].copy()
            elif threshold_name.endswith("_max"):
                mask = working[feature_name].astype(float) <= float(threshold_value)
                working = working.loc[mask].copy()
            else:
                mask = working[feature_name].astype(float) == float(threshold_value)
                working = working.loc[mask].copy()
            counts.append((threshold_name, len(working.index)))
            if first_zero_gate is None and len(working.index) == 0:
                first_zero_gate = threshold_name

        component_counter = {threshold_name: 0 for threshold_name in score_components}
        if not working.empty:
            passing_rows = 0
            for row in working.to_dict(orient="records"):
                passed, details, _ = evaluate_signal_gate(
                    {
                        **hard_filters,
                        **score_components,
                        SIGNAL_SCORE_MIN_KEY: pass_score,
                    },
                    row,
                )
                for threshold_name in score_components:
                    score = float(details[threshold_name]["score"])
                    if score > 0:
                        component_counter[threshold_name] += 1
                if passed:
                    passing_rows += 1
            counts.append((SIGNAL_SCORE_MIN_KEY, passing_rows))
            if first_zero_gate is None and passing_rows == 0:
                first_zero_gate = SIGNAL_SCORE_MIN_KEY
        else:
            counts.append((SIGNAL_SCORE_MIN_KEY, 0))
            if first_zero_gate is None:
                first_zero_gate = SIGNAL_SCORE_MIN_KEY
        return {
            "counts": counts,
            "first_zero_gate": first_zero_gate or "none",
            "component_positive_counts": list(component_counter.items()),
        }

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

    def _attach_selection_diagnostics(self, candidates: pd.DataFrame, *, scan_policy: ScanPolicy | None = None) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        enriched = candidates.copy()
        enriched["selection_source"] = enriched["details"].map(
            lambda payload: payload.get("ranking_components", {}).get("selection_source")
            if isinstance(payload, dict)
            else None
        )
        enriched["model_name"] = enriched["details"].map(
            lambda payload: payload.get("ranking_components", {}).get("model_name")
            if isinstance(payload, dict)
            else None
        )
        enriched["model_predicted_alpha"] = enriched["details"].map(
            lambda payload: float(payload.get("ranking_components", {}).get("model_predicted_alpha", float("nan")))
            if isinstance(payload, dict) and payload.get("ranking_components", {}).get("model_predicted_alpha") is not None
            else float("nan")
        )
        enriched["model_rank"] = enriched["details"].map(
            lambda payload: int(payload.get("ranking_components", {}).get("model_rank"))
            if isinstance(payload, dict) and payload.get("ranking_components", {}).get("model_rank") is not None
            else np.nan
        )
        enriched["model_generated_at"] = enriched["details"].map(
            lambda payload: payload.get("ranking_components", {}).get("model_generated_at")
            if isinstance(payload, dict)
            else None
        )
        enriched["model_reason_summary"] = enriched["details"].map(
            lambda payload: payload.get("ranking_components", {}).get("model_reason_summary")
            if isinstance(payload, dict)
            else None
        )
        enriched["model_comparison_summary"] = enriched["details"].map(
            lambda payload: payload.get("ranking_components", {}).get("model_comparison_summary")
            if isinstance(payload, dict)
            else None
        )
        enriched["selection_score"] = enriched["details"].map(
            lambda payload: float(payload.get("ranking_components", {}).get("selection_score", float("nan")))
            if isinstance(payload, dict)
            else float("nan")
        )
        enriched["selection_score"] = pd.to_numeric(enriched["selection_score"], errors="coerce")
        enriched["selection_score"] = enriched["selection_score"].where(
            enriched["selection_score"].notna(),
            pd.to_numeric(enriched["opportunity_score"], errors="coerce"),
        )
        enriched["recent_feedback_adjustment"] = enriched["details"].map(
            lambda payload: float(payload.get("ranking_components", {}).get("recent_feedback_adjustment", 0.0))
            if isinstance(payload, dict)
            else 0.0
        )
        enriched["slot_overlay_adjustment"] = enriched["details"].map(
            lambda payload: float(payload.get("ranking_components", {}).get("slot_overlay_adjustment", 0.0))
            if isinstance(payload, dict)
            else 0.0
        )
        enriched["slot_overlay_components"] = enriched["details"].map(
            lambda payload: payload.get("slot_overlay_components", {})
            if isinstance(payload, dict)
            else {}
        )
        if scan_policy is not None:
            slot_has_overlay = enriched["strategy_slot"].astype(str).map(
                lambda slot: bool(scan_policy.slot_selection_overlay_weights.get(str(slot), {}))
            )
            fallback_mask = (
                pd.to_numeric(enriched["slot_overlay_adjustment"], errors="coerce").fillna(0.0).abs() <= 1e-12
            ) & slot_has_overlay
            if bool(fallback_mask.any()):
                helper = ScanService(self.db_manager, email_sender=lambda subject, html_body, settings: None)
                fallback_adjustments: list[float] = []
                fallback_components: list[dict[str, float]] = []
                for row in enriched.to_dict(orient="records"):
                    adjustment, components = helper._slot_selection_overlay_adjustment(row, scan_policy=scan_policy)
                    fallback_adjustments.append(float(adjustment))
                    fallback_components.append(components)
                enriched.loc[fallback_mask, "slot_overlay_adjustment"] = pd.Series(fallback_adjustments, index=enriched.index)[fallback_mask]
                enriched.loc[fallback_mask, "slot_overlay_components"] = pd.Series(fallback_components, index=enriched.index)[fallback_mask]
        enriched["recent_drag_picks"] = enriched["details"].map(
            lambda payload: int(payload.get("recent_selection_memory", {}).get("drag_picks", 0))
            if isinstance(payload, dict)
            else 0
        )
        enriched["recent_drag_mean_target"] = enriched["details"].map(
            lambda payload: float(payload.get("recent_selection_memory", {}).get("drag_mean_target", float("nan")))
            if isinstance(payload, dict) and payload.get("recent_selection_memory", {}).get("drag_mean_target") is not None
            else float("nan")
        )
        enriched["recent_missed_winner_count"] = enriched["details"].map(
            lambda payload: int(payload.get("recent_selection_memory", {}).get("missed_winner_count", 0))
            if isinstance(payload, dict)
            else 0
        )
        enriched["recent_missed_winner_mean_gap"] = enriched["details"].map(
            lambda payload: float(payload.get("recent_selection_memory", {}).get("missed_winner_mean_gap", float("nan")))
            if isinstance(payload, dict) and payload.get("recent_selection_memory", {}).get("missed_winner_mean_gap") is not None
            else float("nan")
        )
        raw_ranked = enriched.sort_values(
            ["opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        raw_rank_lookup = {
            (str(row.ticker), str(row.strategy_slot)): index + 1
            for index, row in enumerate(raw_ranked.itertuples(index=False))
        }
        selection_ranked = enriched.sort_values(
            ["selection_score", "opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        selection_rank_lookup = {
            (str(row.ticker), str(row.strategy_slot)): index + 1
            for index, row in enumerate(selection_ranked.itertuples(index=False))
        }
        enriched["raw_opportunity_rank"] = [
            raw_rank_lookup.get((str(row["ticker"]), str(row["strategy_slot"])))
            for row in enriched.to_dict(orient="records")
        ]
        enriched["adjusted_selection_rank"] = [
            selection_rank_lookup.get((str(row["ticker"]), str(row["strategy_slot"])))
            for row in enriched.to_dict(orient="records")
        ]
        return enriched

    def _attach_feature_snapshot_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "details" not in frame.columns:
            return frame
        enriched = frame.copy()
        snapshot_fields = [
            "sector_pct_above_50",
            "sector_pct_above_200",
            "sector_median_roc_63",
            "distance_above_20d_high",
            "sma_200_dist",
            "sma_50_dist",
            "roc_63",
            "roc_126",
            "relative_strength_index_vs_spy",
            "relative_strength_index_vs_qqq",
            "relative_strength_index_vs_xlk",
            "relative_strength_index_vs_subindustry",
            "avg_abs_gap_pct_20",
            "max_gap_down_pct_60",
            "base_range_pct_20",
            "base_atr_contraction_20",
            "base_volume_dryup_ratio_20",
            "breakout_volume_ratio_50",
        ]
        for field in snapshot_fields:
            if field in enriched.columns:
                continue
            enriched[field] = enriched["details"].map(
                lambda payload: payload.get("feature_snapshot", {}).get(field)
                if isinstance(payload, dict)
                else None
            )
        return enriched

    def _attach_regime_context(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        benchmarks = {
            benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or ""))
            for row in frame.to_dict(orient="records")
        }
        tickers = sorted({"SPY", "QQQ"}.union({ticker for ticker in benchmarks if ticker}))
        try:
            history = self.db_manager.load_price_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load regime context history for scan analysis: %s", exc)
            return frame
        if history.empty:
            return frame
        regime_maps: dict[str, dict[str, bool]] = {}
        for ticker, group in history.groupby("ticker", sort=False):
            ordered = group.copy()
            ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
            ordered = ordered.sort_values("date").reset_index(drop=True)
            ordered["sma_200"] = ordered["adj_close"].astype(float).rolling(window=200, min_periods=200).mean()
            ordered["regime_green"] = ordered["adj_close"].astype(float) > ordered["sma_200"].astype(float)
            valid = ordered.dropna(subset=["sma_200"]).copy()
            regime_maps[str(ticker)] = {
                row["date"].strftime("%Y-%m-%d"): bool(row["regime_green"])
                for row in valid.to_dict(orient="records")
            }
        enriched = frame.copy()
        scan_dates = enriched["scan_date"].astype(str)
        enriched["sector_benchmark"] = enriched.apply(
            lambda row: benchmark_etf_for_sector(str(row.get("sector") or row.get("strategy_sector") or "")),
            axis=1,
        )
        enriched["spy_regime_green"] = scan_dates.map(lambda scan_date: regime_maps.get("SPY", {}).get(scan_date))
        enriched["qqq_regime_green"] = scan_dates.map(lambda scan_date: regime_maps.get("QQQ", {}).get(scan_date))
        enriched["sector_regime_green"] = [
            regime_maps.get(str(row["sector_benchmark"]), {}).get(str(row["scan_date"]))
            for row in enriched.to_dict(orient="records")
        ]
        enriched["sector_breadth_bucket"] = enriched.apply(self._sector_breadth_bucket, axis=1)
        return enriched

    def _sector_breadth_bucket(self, row) -> str:
        above_50 = pd.to_numeric(pd.Series([row.get("sector_pct_above_50")]), errors="coerce").iloc[0]
        above_200 = pd.to_numeric(pd.Series([row.get("sector_pct_above_200")]), errors="coerce").iloc[0]
        if pd.isna(above_50) or pd.isna(above_200):
            return "unknown"
        if above_50 >= 0.60 and above_200 >= 0.60:
            return "high"
        if above_50 <= 0.40 and above_200 <= 0.40:
            return "low"
        return "mixed"

    def _render_recent_early_read(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        recent_dates: int,
    ) -> list[str]:
        lines = ["## Recent Early Read", ""]
        if frame.empty or target_column not in frame.columns:
            lines.append("No early-read scan data is available.")
            lines.append("")
            return lines
        matured_dates = (
            frame.loc[frame[target_column].notna(), "scan_date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(int(recent_dates))
            .tolist()
        )
        if not matured_dates:
            lines.append(f"No scan dates have populated {target_column} yet.")
            lines.append("")
            return lines
        subset = frame[frame["scan_date"].isin(matured_dates)].copy()
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- dates: {', '.join(sorted(matured_dates))}")
        lines.append("")
        grouped = (
            subset.groupby("scan_date", dropna=False)
            .apply(
                lambda day: pd.Series(
                    {
                        "selected_mean": day.loc[day["selected"].astype(int) == 1, target_column].astype(float).mean(),
                        "excluded_mean": day.loc[day["selected"].astype(int) == 0, target_column].astype(float).mean(),
                        "selected_hit": (day.loc[day["selected"].astype(int) == 1, target_column].astype(float) > 0.0).mean(),
                        "selected_count": int((day["selected"].astype(int) == 1).sum()),
                    }
                )
            )
            .reset_index()
            .sort_values("scan_date")
        )
        for row in grouped.itertuples(index=False):
            lines.append(f"### {row.scan_date}")
            lines.append(f"- selected_mean_target: {float(row.selected_mean):.6f}")
            lines.append(f"- excluded_mean_target: {float(row.excluded_mean):.6f}")
            lines.append(f"- selected_hit_rate: {float(row.selected_hit):.4f}")
            lines.append(f"- selected_count: {int(row.selected_count)}")
            selected_tickers = subset[(subset["scan_date"] == row.scan_date) & (subset["selected"].astype(int) == 1)]["ticker"].astype(str).tolist()
            lines.append(f"- selected_tickers: {', '.join(selected_tickers)}")
            lines.append("")
        return lines

    def _render_post_change_selector_maturity(
        self,
        frame: pd.DataFrame,
        *,
        horizons: tuple[int, ...],
    ) -> list[str]:
        lines = ["## Post-Change Selector Maturity", ""]
        if frame.empty or "recent_feedback_adjustment" not in frame.columns:
            lines.append("No selector-maturity data is available.")
            lines.append("")
            return lines
        post_change_dates = (
            frame.loc[pd.to_numeric(frame["recent_feedback_adjustment"], errors="coerce").fillna(0.0).abs() > 1e-12, "scan_date"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not post_change_dates:
            lines.append("No scan dates show active recent-feedback selector adjustments yet.")
            lines.append("")
            return lines
        subset = frame[frame["scan_date"].isin(post_change_dates)].copy()
        selected_subset = subset[subset["selected"].astype(int) == 1].copy()
        lines.append("- basis: scan dates with at least one non-zero recent_feedback_adjustment")
        lines.append(f"- post_change_dates: {', '.join(post_change_dates)}")
        lines.append(f"- post_change_scan_dates: {len(post_change_dates)}")
        lines.append(f"- post_change_selected_rows: {len(selected_subset.index)}")
        lines.append("")
        for horizon in horizons:
            column = f"alpha_vs_sector_{horizon}d"
            if column not in selected_subset.columns:
                lines.append(f"- {horizon}d: column unavailable")
                continue
            available_mask = selected_subset[column].notna()
            available_dates = (
                selected_subset.loc[available_mask, "scan_date"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            lines.append(
                f"- {horizon}d: selected_rows_with_outcomes={int(available_mask.sum())}/{len(selected_subset.index)} "
                f"dates_with_outcomes={len(available_dates)}/{len(post_change_dates)}"
            )
            if available_dates:
                lines.append(f"  first_available_date: {available_dates[0]}")
                lines.append(f"  latest_available_date: {available_dates[-1]}")
            else:
                lines.append("  status: pending")
        lines.append("")
        return lines

    def _render_signal_first_selector_early_read(
        self,
        frame: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
        recent_dates: int,
        horizons: tuple[int, ...],
    ) -> list[str]:
        lines = ["## Signal-First Selector Early Read", ""]
        if frame.empty:
            lines.append("No signal-first selector data is available.")
            lines.append("")
            return lines
        candidate_dates = (
            frame.loc[frame["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .tolist()
        )
        if not candidate_dates:
            lines.append("No selected scan dates are available.")
            lines.append("")
            return lines
        selected_dates: list[str] = []
        for scan_date in candidate_dates:
            day_frame = frame[frame["scan_date"].astype(str) == str(scan_date)].copy()
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            opportunity = self._date_select_candidates(day_frame, score_column="opportunity_score", scan_policy=scan_policy)
            runtime_tickers = tuple(sorted(runtime["ticker"].astype(str).tolist()))
            opportunity_tickers = tuple(sorted(opportunity["ticker"].astype(str).tolist()))
            if runtime_tickers != opportunity_tickers:
                selected_dates.append(str(scan_date))
            if len(selected_dates) >= int(recent_dates):
                break
        if not selected_dates:
            selected_dates = candidate_dates[: int(recent_dates)]
        lines.append("- basis: recent live signal-first dates compared with opportunity-score counterfactual under the same caps")
        lines.append(f"- scan_dates: {', '.join(sorted(selected_dates))}")
        lines.append("")
        for scan_date in sorted(selected_dates):
            day_frame = frame[frame["scan_date"].astype(str) == str(scan_date)].copy()
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            opportunity = self._date_select_candidates(day_frame, score_column="opportunity_score", scan_policy=scan_policy)
            lines.append(f"### {scan_date}")
            lines.append(f"- runtime_tickers: {', '.join(runtime['ticker'].astype(str).tolist()) if not runtime.empty else 'none'}")
            lines.append(f"- opportunity_counterfactual: {', '.join(opportunity['ticker'].astype(str).tolist()) if not opportunity.empty else 'none'}")
            runtime_set = set(runtime["ticker"].astype(str).tolist())
            opportunity_set = set(opportunity["ticker"].astype(str).tolist())
            added = sorted(runtime_set - opportunity_set)
            removed = sorted(opportunity_set - runtime_set)
            lines.append(f"- added_vs_opportunity: {', '.join(added) if added else 'none'}")
            lines.append(f"- removed_vs_opportunity: {', '.join(removed) if removed else 'none'}")
            for horizon in horizons:
                column = f"alpha_vs_sector_{horizon}d"
                if column not in day_frame.columns:
                    lines.append(f"- {horizon}d: column unavailable")
                    continue
                runtime_values = runtime[column].dropna().astype(float)
                opportunity_values = opportunity[column].dropna().astype(float)
                if runtime_values.empty and opportunity_values.empty:
                    lines.append(f"- {horizon}d: pending")
                    continue
                runtime_text = f"{float(runtime_values.mean()):.6f}" if not runtime_values.empty else "pending"
                opportunity_text = f"{float(opportunity_values.mean()):.6f}" if not opportunity_values.empty else "pending"
                lines.append(
                    f"- {horizon}d: runtime_mean_target={runtime_text} "
                    f"opportunity_counterfactual_mean_target={opportunity_text}"
                )
            lines.append("")
        return lines

    def _render_selector_shadow_comparison(
        self,
        frame: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
        target_column: str,
        recent_dates: int,
    ) -> list[str]:
        lines = ["## Selector Shadow Comparison", ""]
        if frame.empty or target_column not in frame.columns:
            lines.append("No selector shadow data is available.")
            lines.append("")
            return lines
        matured_dates = (
            frame.loc[frame[target_column].notna(), "scan_date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(int(recent_dates))
            .tolist()
        )
        if not matured_dates:
            lines.append(f"No scan dates have populated {target_column} yet.")
            lines.append("")
            return lines
        subset = frame[frame["scan_date"].isin(matured_dates)].copy()
        rows: list[dict[str, object]] = []
        swaps: list[dict[str, object]] = []
        for (scan_date, slot), day_frame in subset.groupby(["scan_date", "strategy_slot"], sort=True):
            runtime = day_frame[day_frame["selected"].astype(int) == 1].copy()
            shadow_old = self._slot_select_candidates(day_frame, score_column="opportunity_score", scan_policy=scan_policy)
            shadow_new = self._slot_select_candidates(day_frame, score_column="selection_score", scan_policy=scan_policy)
            rows.extend(
                [
                    self._slot_selection_metrics(scan_date, slot, "runtime", runtime, target_column),
                    self._slot_selection_metrics(scan_date, slot, "shadow_old", shadow_old, target_column),
                    self._slot_selection_metrics(scan_date, slot, "shadow_new", shadow_new, target_column),
                ]
            )
            old_tickers = tuple(shadow_old["ticker"].astype(str).tolist())
            new_tickers = tuple(shadow_new["ticker"].astype(str).tolist())
            if old_tickers != new_tickers:
                swaps.append(
                    {
                        "scan_date": str(scan_date),
                        "strategy_slot": str(slot),
                        "old_tickers": old_tickers,
                        "new_tickers": new_tickers,
                        "old_mean": float(shadow_old[target_column].astype(float).mean()) if not shadow_old.empty else float("nan"),
                        "new_mean": float(shadow_new[target_column].astype(float).mean()) if not shadow_new.empty else float("nan"),
                    }
                )
        metrics = pd.DataFrame(rows)
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- recent_matured_dates: {len(matured_dates)}")
        lines.append(f"- date_range: {min(matured_dates)} to {max(matured_dates)}")
        lines.append("")
        if metrics.empty:
            lines.append("No selector shadow rows available.")
            lines.append("")
            return lines
        lines.append("### Summary")
        summary = (
            metrics.groupby("method", dropna=False)
            .agg(
                mean_target=("mean_target", "mean"),
                hit_rate=("hit_rate", "mean"),
                avg_pick_count=("pick_count", "mean"),
                days=("scan_date", "nunique"),
            )
            .reset_index()
        )
        method_order = {"runtime": 0, "shadow_old": 1, "shadow_new": 2}
        summary["method_order"] = summary["method"].map(method_order).fillna(9).astype(int)
        summary = summary.sort_values(["method_order", "method"])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.method}: mean_target={float(row.mean_target):.6f} "
                f"hit_rate={float(row.hit_rate):.4f} avg_pick_count={float(row.avg_pick_count):.2f} "
                f"days={int(row.days)}"
            )
        lines.append("")
        lines.append("### Slot Breakdown")
        slot_summary = (
            metrics.groupby(["strategy_slot", "method"], dropna=False)
            .agg(
                mean_target=("mean_target", "mean"),
                hit_rate=("hit_rate", "mean"),
                days=("scan_date", "nunique"),
            )
            .reset_index()
        )
        slot_summary["method_order"] = slot_summary["method"].map(method_order).fillna(9).astype(int)
        slot_summary = slot_summary.sort_values(["strategy_slot", "method_order", "method"])
        for slot, slot_frame in slot_summary.groupby("strategy_slot", sort=True):
            lines.append(f"#### {slot}")
            for row in slot_frame.itertuples(index=False):
                lines.append(
                    f"- {row.method}: mean_target={float(row.mean_target):.6f} "
                    f"hit_rate={float(row.hit_rate):.4f} days={int(row.days)}"
                )
            lines.append("")
        lines.append("### Biggest Old-vs-New Swaps")
        if not swaps:
            lines.append("- none")
            lines.append("")
            return lines
        ranked_swaps = sorted(
            swaps,
            key=lambda row: ((row["new_mean"] - row["old_mean"]) if pd.notna(row["new_mean"]) and pd.notna(row["old_mean"]) else float("-inf")),
            reverse=True,
        )[:10]
        for row in ranked_swaps:
            delta = float(row["new_mean"] - row["old_mean"]) if pd.notna(row["new_mean"]) and pd.notna(row["old_mean"]) else float("nan")
            lines.append(f"#### {row['scan_date']} {row['strategy_slot']}")
            lines.append(f"- shadow_old: {', '.join(row['old_tickers']) if row['old_tickers'] else 'none'}")
            lines.append(f"- shadow_new: {', '.join(row['new_tickers']) if row['new_tickers'] else 'none'}")
            lines.append(f"- old_mean_target: {float(row['old_mean']):.6f}")
            lines.append(f"- new_mean_target: {float(row['new_mean']):.6f}")
            lines.append(f"- delta: {delta:.6f}")
            lines.append("")
        return lines

    def _render_selector_bakeoff(
        self,
        frame: pd.DataFrame,
        *,
        scan_policy: ScanPolicy,
        target_column: str,
        recent_windows: tuple[int, ...],
    ) -> list[str]:
        lines = ["## Selector Bakeoff", ""]
        if frame.empty or target_column not in frame.columns:
            lines.append("No selector bakeoff data is available.")
            lines.append("")
            return lines
        all_matured_dates = (
            frame.loc[frame[target_column].notna(), "scan_date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .tolist()
        )
        if not all_matured_dates:
            lines.append(f"No scan dates have populated {target_column} yet.")
            lines.append("")
            return lines
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- current_live_policy: signal_score primary with opportunity_score tie-break and portfolio caps")
        lines.append(f"- available_matured_dates: {len(all_matured_dates)}")
        lines.append("")
        for window in recent_windows:
            matured_dates = all_matured_dates[: int(window)]
            if not matured_dates:
                continue
            subset = frame[frame["scan_date"].isin(matured_dates)].copy()
            rows: list[dict[str, object]] = []
            slot_rows: list[dict[str, object]] = []
            disagreements: list[dict[str, object]] = []
            for scan_date, day_frame in subset.groupby("scan_date", sort=True):
                policies = {
                    "runtime": day_frame[day_frame["selected"].astype(int) == 1].copy(),
                    "opportunity": self._date_select_candidates(day_frame, score_column="opportunity_score", scan_policy=scan_policy),
                    "signal": self._date_select_candidates(day_frame, score_column="signal_score", scan_policy=scan_policy),
                    "learned": self._date_select_candidates(day_frame, score_column="ranker_score", scan_policy=scan_policy),
                    "random": self._date_select_candidates(day_frame, score_column="random_selector_score", scan_policy=scan_policy),
                }
                for method, selected_frame in policies.items():
                    rows.append(self._selection_metrics(scan_date, method, selected_frame, target_column))
                    for slot, slot_frame in selected_frame.groupby("strategy_slot", sort=True):
                        slot_rows.append(self._selection_metrics(scan_date, method, slot_frame, target_column, strategy_slot=str(slot)))
                runtime_tickers = tuple(sorted(policies["runtime"]["ticker"].astype(str).tolist()))
                opportunity_tickers = tuple(sorted(policies["opportunity"]["ticker"].astype(str).tolist()))
                if runtime_tickers != opportunity_tickers:
                    disagreements.append(
                        {
                            "scan_date": str(scan_date),
                            "runtime_tickers": runtime_tickers,
                            "opportunity_tickers": opportunity_tickers,
                            "runtime_mean": self._frame_mean_target(policies["runtime"], target_column),
                            "opportunity_mean": self._frame_mean_target(policies["opportunity"], target_column),
                        }
                    )
            metrics = pd.DataFrame(rows)
            lines.append(f"### Window {int(window)}")
            lines.append(f"- recent_matured_dates: {len(matured_dates)}")
            lines.append(f"- date_range: {min(matured_dates)} to {max(matured_dates)}")
            lines.append("")
            if metrics.empty:
                lines.append("No bakeoff rows available.")
                lines.append("")
                continue
            lines.append("#### Summary")
            runtime_lookup = (
                metrics.loc[metrics["method"] == "runtime", ["scan_date", "mean_target"]]
                .rename(columns={"mean_target": "runtime_mean_target"})
                .copy()
            )
            summary = (
                metrics.groupby("method", dropna=False)
                .agg(
                    mean_target=("mean_target", "mean"),
                    hit_rate=("hit_rate", "mean"),
                    avg_pick_count=("pick_count", "mean"),
                    days=("scan_date", "nunique"),
                )
                .reset_index()
            )
            merged = metrics.merge(runtime_lookup, on="scan_date", how="left")
            runtime_beats = (
                merged.assign(beats_runtime=(merged["mean_target"] > merged["runtime_mean_target"]).astype(float))
                .groupby("method", dropna=False)["beats_runtime"]
                .mean()
                .reset_index()
            )
            summary = summary.merge(runtime_beats, on="method", how="left")
            method_order = {"runtime": 0, "opportunity": 1, "signal": 2, "learned": 3, "random": 4}
            summary["method_order"] = summary["method"].map(method_order).fillna(9).astype(int)
            summary = summary.sort_values(["method_order", "method"])
            for row in summary.itertuples(index=False):
                beats_runtime = "n/a" if row.method == "runtime" or pd.isna(row.beats_runtime) else f"{float(row.beats_runtime):.4f}"
                lines.append(
                    f"- {row.method}: mean_target={float(row.mean_target):.6f} "
                    f"hit_rate={float(row.hit_rate):.4f} avg_pick_count={float(row.avg_pick_count):.2f} "
                    f"days={int(row.days)} beats_runtime={beats_runtime}"
                )
            lines.append("")
            slot_metrics = pd.DataFrame(slot_rows)
            lines.append("#### Slot Breakdown")
            if slot_metrics.empty:
                lines.append("No slot-level bakeoff rows available.")
                lines.append("")
            else:
                slot_summary = (
                    slot_metrics.groupby(["strategy_slot", "method"], dropna=False)
                    .agg(
                        mean_target=("mean_target", "mean"),
                        hit_rate=("hit_rate", "mean"),
                        days=("scan_date", "nunique"),
                    )
                    .reset_index()
                )
                slot_summary["method_order"] = slot_summary["method"].map(method_order).fillna(9).astype(int)
                slot_summary = slot_summary.sort_values(["strategy_slot", "method_order", "method"])
                for slot, slot_frame in slot_summary.groupby("strategy_slot", sort=True):
                    lines.append(f"##### {slot}")
                    for row in slot_frame.itertuples(index=False):
                        lines.append(
                            f"- {row.method}: mean_target={float(row.mean_target):.6f} "
                            f"hit_rate={float(row.hit_rate):.4f} days={int(row.days)}"
                        )
                    lines.append("")
            if int(window) != int(recent_windows[0]):
                continue
            lines.append("#### Biggest Runtime-vs-Opportunity Disagreements")
            if not disagreements:
                lines.append("- none")
                lines.append("")
                continue
            ranked_disagreements = sorted(
                disagreements,
                key=lambda row: (
                    (row["opportunity_mean"] - row["runtime_mean"])
                    if pd.notna(row["opportunity_mean"]) and pd.notna(row["runtime_mean"])
                    else float("-inf")
                ),
                reverse=True,
            )[:10]
            for row in ranked_disagreements:
                delta = (
                    float(row["opportunity_mean"] - row["runtime_mean"])
                    if pd.notna(row["opportunity_mean"]) and pd.notna(row["runtime_mean"])
                    else float("nan")
                )
                lines.append(f"##### {row['scan_date']}")
                lines.append(f"- runtime: {', '.join(row['runtime_tickers']) if row['runtime_tickers'] else 'none'}")
                lines.append(f"- opportunity: {', '.join(row['opportunity_tickers']) if row['opportunity_tickers'] else 'none'}")
                lines.append(f"- runtime_mean_target: {float(row['runtime_mean']):.6f}")
                lines.append(f"- opportunity_mean_target: {float(row['opportunity_mean']):.6f}")
                lines.append(f"- delta: {delta:.6f}")
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
            selection_source = getattr(row, "selection_source", None)
            if isinstance(selection_source, str) and selection_source:
                lines.append(f"- selection_source: {selection_source}")
            model_name = getattr(row, "model_name", None)
            if isinstance(model_name, str) and model_name:
                lines.append(f"- model_name: {model_name}")
            if pd.notna(getattr(row, "model_predicted_alpha", np.nan)):
                lines.append(f"- model_predicted_alpha: {float(row.model_predicted_alpha):.4f}")
            if pd.notna(getattr(row, "model_rank", np.nan)):
                lines.append(f"- model_rank: {int(row.model_rank)}")
            model_reason_summary = getattr(row, "model_reason_summary", None)
            if isinstance(model_reason_summary, str) and model_reason_summary:
                lines.append(f"- model_reason_summary: {model_reason_summary}")
            model_comparison_summary = getattr(row, "model_comparison_summary", None)
            if isinstance(model_comparison_summary, str) and model_comparison_summary:
                lines.append(f"- model_comparison_summary: {model_comparison_summary}")
            lines.append(f"- opportunity_score: {float(row.opportunity_score):.4f}")
            if pd.notna(getattr(row, "selection_score", np.nan)):
                lines.append(f"- selection_score: {float(row.selection_score):.4f}")
            if pd.notna(getattr(row, "raw_opportunity_rank", np.nan)):
                lines.append(f"- raw_opportunity_rank: {int(row.raw_opportunity_rank)}")
            if pd.notna(getattr(row, "adjusted_selection_rank", np.nan)):
                lines.append(f"- adjusted_selection_rank: {int(row.adjusted_selection_rank)}")
            if pd.notna(getattr(row, "recent_feedback_adjustment", np.nan)):
                lines.append(f"- recent_feedback_adjustment: {float(row.recent_feedback_adjustment):+.4f}")
            if pd.notna(getattr(row, "slot_overlay_adjustment", np.nan)):
                lines.append(f"- slot_overlay_adjustment: {float(row.slot_overlay_adjustment):+.4f}")
            slot_overlay_components = getattr(row, "slot_overlay_components", {})
            if isinstance(slot_overlay_components, dict) and slot_overlay_components:
                component_text = ", ".join(
                    f"{name}={float(value):+.4f}"
                    for name, value in sorted(slot_overlay_components.items())
                )
                lines.append(f"- slot_overlay_components: {component_text}")
            if int(getattr(row, "recent_drag_picks", 0) or 0) > 0:
                lines.append(f"- recent_drag_picks: {int(row.recent_drag_picks)}")
            if pd.notna(getattr(row, "recent_drag_mean_target", np.nan)):
                lines.append(f"- recent_drag_mean_target: {float(row.recent_drag_mean_target):.4f}")
            if int(getattr(row, "recent_missed_winner_count", 0) or 0) > 0:
                lines.append(f"- recent_missed_winner_count: {int(row.recent_missed_winner_count)}")
            if pd.notna(getattr(row, "recent_missed_winner_mean_gap", np.nan)):
                lines.append(f"- recent_missed_winner_mean_gap: {float(row.recent_missed_winner_mean_gap):.4f}")
            lines.append(f"- signal_score: {self._display_signal_score(row)}")
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

    def _render_portfolio_strength_coverage(self, candidates: pd.DataFrame, *, top_n: int = 10) -> list[str]:
        lines = ["## Portfolio Strength Coverage", ""]
        if candidates.empty:
            lines.append("No candidates available.")
            lines.append("")
            return lines

        working = candidates.copy()
        if "details" not in working.columns:
            working["details"] = [{} for _ in range(len(working.index))]
        working["already_owned"] = working["details"].map(
            lambda payload: bool(payload.get("already_owned", False)) if isinstance(payload, dict) else False
        )
        working["pre_penalty_opportunity_score"] = working.apply(
            lambda row: self._pre_penalty_opportunity_score(row),
            axis=1,
        )
        if "selection_score" not in working.columns:
            working["selection_score"] = np.nan
        ranked = working.sort_values(
            ["pre_penalty_opportunity_score", "selection_score", "signal_score", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        if ranked.empty:
            lines.append("No ranked candidates available.")
            lines.append("")
            return lines

        top_six = ranked.head(6)
        top_slice = ranked.head(int(top_n))
        held_top_six = int(top_six["already_owned"].sum())
        held_top_n = int(top_slice["already_owned"].sum())
        owned_candidates = ranked[ranked["already_owned"]].copy()
        strongest_held = str(owned_candidates.iloc[0]["ticker"]) if not owned_candidates.empty else "none"
        strongest_unheld = str(ranked[~ranked["already_owned"]].iloc[0]["ticker"]) if not ranked[~ranked["already_owned"]].empty else "none"

        lines.append(f"- top_6_already_held: {held_top_six}/6")
        lines.append(f"- top_{int(top_n)}_already_held: {held_top_n}/{min(int(top_n), len(ranked.index))}")
        lines.append(f"- held_candidates_in_scan: {len(owned_candidates.index)}")
        lines.append(f"- strongest_held_candidate: {strongest_held}")
        lines.append(f"- strongest_unheld_candidate: {strongest_unheld}")

        open_tickers = self._open_trade_tickers()
        candidate_tickers = set(working["ticker"].astype(str))
        missing_open = sorted(open_tickers - candidate_tickers)
        if missing_open:
            lines.append(f"- open_holdings_not_in_candidate_set: {', '.join(missing_open)}")
        lines.append("")
        lines.append(f"### Top {min(int(top_n), len(ranked.index))} Candidates")
        for index, row in enumerate(top_slice.itertuples(index=False), start=1):
            status = "held" if bool(row.already_owned) else "not held"
            selected = "selected" if int(getattr(row, "selected", 0) or 0) == 1 else "not selected"
            selection_score = getattr(row, "selection_score", np.nan)
            selection_part = (
                f", selection_score={float(selection_score):.4f}"
                if pd.notna(selection_score)
                else ""
            )
            lines.append(
                f"{index}. {row.ticker} - {status}, {selected}, "
                f"score={float(row.pre_penalty_opportunity_score):.4f}, "
                f"post_penalty_score={float(row.opportunity_score):.4f}"
                f"{selection_part}, sector={row.sector}"
            )
        lines.append("")
        return lines

    def _pre_penalty_opportunity_score(self, row) -> float:
        details = row.get("details") if isinstance(row, pd.Series) else getattr(row, "details", {})
        if isinstance(details, dict):
            value = details.get("pre_penalty_opportunity_score")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        try:
            return float(row.get("opportunity_score") if isinstance(row, pd.Series) else getattr(row, "opportunity_score"))
        except (TypeError, ValueError):
            return float("-inf")

    def _open_trade_tickers(self) -> set[str]:
        open_trade_loader = getattr(self.db_manager, "list_open_trades", None)
        if not callable(open_trade_loader):
            return set()
        tickers: set[str] = set()
        for trade in open_trade_loader() or []:
            if isinstance(trade, dict):
                ticker = trade.get("ticker")
            else:
                try:
                    ticker = trade["ticker"]
                except (KeyError, TypeError, IndexError):
                    ticker = getattr(trade, "ticker", None)
            if ticker not in (None, ""):
                tickers.add(str(ticker))
        return tickers

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
            selection_source = getattr(row, "selection_source", None)
            if isinstance(selection_source, str) and selection_source:
                lines.append(f"- selection_source: {selection_source}")
            model_name = getattr(row, "model_name", None)
            if isinstance(model_name, str) and model_name:
                lines.append(f"- model_name: {model_name}")
            if pd.notna(getattr(row, "model_predicted_alpha", np.nan)):
                lines.append(f"- model_predicted_alpha: {float(row.model_predicted_alpha):.4f}")
            if pd.notna(getattr(row, "model_rank", np.nan)):
                lines.append(f"- model_rank: {int(row.model_rank)}")
            model_reason_summary = getattr(row, "model_reason_summary", None)
            if isinstance(model_reason_summary, str) and model_reason_summary:
                lines.append(f"- model_reason_summary: {model_reason_summary}")
            model_comparison_summary = getattr(row, "model_comparison_summary", None)
            if isinstance(model_comparison_summary, str) and model_comparison_summary:
                lines.append(f"- model_comparison_summary: {model_comparison_summary}")
            lines.append(f"- opportunity_score: {float(row.opportunity_score):.4f}")
            if pd.notna(getattr(row, "selection_score", np.nan)):
                lines.append(f"- selection_score: {float(row.selection_score):.4f}")
            if pd.notna(getattr(row, "raw_opportunity_rank", np.nan)):
                lines.append(f"- raw_opportunity_rank: {int(row.raw_opportunity_rank)}")
            if pd.notna(getattr(row, "adjusted_selection_rank", np.nan)):
                lines.append(f"- adjusted_selection_rank: {int(row.adjusted_selection_rank)}")
            if pd.notna(getattr(row, "recent_feedback_adjustment", np.nan)):
                lines.append(f"- recent_feedback_adjustment: {float(row.recent_feedback_adjustment):+.4f}")
            if pd.notna(getattr(row, "slot_overlay_adjustment", np.nan)):
                lines.append(f"- slot_overlay_adjustment: {float(row.slot_overlay_adjustment):+.4f}")
            slot_overlay_components = getattr(row, "slot_overlay_components", {})
            if isinstance(slot_overlay_components, dict) and slot_overlay_components:
                component_text = ", ".join(
                    f"{name}={float(value):+.4f}"
                    for name, value in sorted(slot_overlay_components.items())
                )
                lines.append(f"- slot_overlay_components: {component_text}")
            if int(getattr(row, "recent_drag_picks", 0) or 0) > 0:
                lines.append(f"- recent_drag_picks: {int(row.recent_drag_picks)}")
            if pd.notna(getattr(row, "recent_drag_mean_target", np.nan)):
                lines.append(f"- recent_drag_mean_target: {float(row.recent_drag_mean_target):.4f}")
            if int(getattr(row, "recent_missed_winner_count", 0) or 0) > 0:
                lines.append(f"- recent_missed_winner_count: {int(row.recent_missed_winner_count)}")
            if pd.notna(getattr(row, "recent_missed_winner_mean_gap", np.nan)):
                lines.append(f"- recent_missed_winner_mean_gap: {float(row.recent_missed_winner_mean_gap):.4f}")
            lines.append(f"- signal_score: {self._display_signal_score(row)}")
            lines.append(f"- overlap_penalty: {float(row.overlap_penalty):.4f}")
            lines.append(f"- exclusion_reason: {reason}")
            lines.extend(self._render_overlap_component_lines(details))
            lines.append("")
        return lines

    def _display_signal_score(self, row) -> str:
        if getattr(row, "selection_source", None) == "shortlist_model":
            return "n/a (model-sourced)"
        signal_score = getattr(row, "signal_score", np.nan)
        if pd.notna(signal_score):
            return f"{float(signal_score):.4f}"
        return "n/a"

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

    def _render_recent_selection_mistakes(
        self,
        all_candidates: pd.DataFrame,
        *,
        target_column: str,
        max_scan_dates: int = 20,
        max_swaps: int = 12,
        max_drags: int = 8,
    ) -> list[str]:
        lines = ["## Recent Selection Mistakes", ""]
        if target_column not in all_candidates.columns:
            lines.append(f"Missing target column: {target_column}")
            lines.append("")
            return lines
        matured = all_candidates.dropna(subset=[target_column]).copy()
        if matured.empty:
            lines.append("No matured scan candidates are available yet.")
            lines.append("")
            return lines
        recent_scan_dates = (
            matured.loc[matured["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(max_scan_dates)
            .tolist()
        )
        if not recent_scan_dates:
            lines.append("No matured selected scan dates are available yet.")
            lines.append("")
            return lines
        recent = matured[matured["scan_date"].astype(str).isin(recent_scan_dates)].copy()
        selected = recent[recent["selected"].astype(int) == 1].copy()
        excluded = recent[recent["selected"].astype(int) == 0].copy()
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- recent_scan_dates: {len(recent_scan_dates)}")
        lines.append(f"- date_range: {recent_scan_dates[-1]} -> {recent_scan_dates[0]}")
        lines.append("")
        lines.extend(self._render_recent_selection_group("Selected ideas", selected, target_column))
        lines.extend(self._render_recent_selection_group("Excluded eligible ideas", excluded, target_column))
        lines.extend(self._render_recent_selection_slot_breakdown(selected, target_column))
        lines.extend(self._render_recent_selection_slot_breakdown(excluded, target_column, title="Excluded Slot Breakdown"))
        lines.extend(self._render_recent_missed_swaps(recent, target_column=target_column, max_rows=max_swaps))
        lines.extend(self._render_recent_repeat_drags(selected, target_column=target_column, max_rows=max_drags))
        return lines

    def _render_recent_selection_group(self, title: str, frame: pd.DataFrame, target_column: str) -> list[str]:
        lines = [f"### {title}"]
        if frame.empty or target_column not in frame.columns:
            lines.append("- none")
            lines.append("")
            return lines
        series = frame[target_column].dropna().astype(float)
        if series.empty:
            lines.append("- none")
            lines.append("")
            return lines
        lines.append(f"- rows: {len(series.index)}")
        lines.append(f"- mean_target: {series.mean():.6f}")
        lines.append(f"- median_target: {series.median():.6f}")
        lines.append(f"- hit_rate: {(series > 0.0).mean():.4f}")
        lines.append("")
        return lines

    def _render_recent_selection_slot_breakdown(
        self,
        frame: pd.DataFrame,
        target_column: str,
        *,
        title: str = "Selected Slot Breakdown",
    ) -> list[str]:
        lines = [f"### {title}"]
        if frame.empty or target_column not in frame.columns:
            lines.append("- none")
            lines.append("")
            return lines
        grouped = (
            frame.groupby("strategy_slot", dropna=False)
            .agg(
                rows=(target_column, "count"),
                mean_target=(target_column, "mean"),
                hit_rate=(target_column, lambda values: float((pd.Series(values).astype(float) > 0.0).mean())),
            )
            .reset_index()
            .sort_values(["mean_target", "rows", "strategy_slot"], ascending=[False, False, True])
        )
        if grouped.empty:
            lines.append("- none")
            lines.append("")
            return lines
        for row in grouped.itertuples(index=False):
            lines.append(
                f"- {row.strategy_slot}: mean_target={float(row.mean_target):.6f} "
                f"hit_rate={float(row.hit_rate):.4f} rows={int(row.rows)}"
            )
        lines.append("")
        return lines

    def _render_recent_missed_swaps(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        max_rows: int,
    ) -> list[str]:
        lines = ["### Biggest Missed Swaps"]
        if frame.empty:
            lines.append("- none")
            lines.append("")
            return lines
        rows: list[dict[str, object]] = []
        for (scan_date, strategy_slot), subset in frame.groupby(["scan_date", "strategy_slot"], sort=True):
            selected = subset[subset["selected"].astype(int) == 1].copy()
            excluded = subset[subset["selected"].astype(int) == 0].copy()
            if selected.empty or excluded.empty:
                continue
            selected = selected.dropna(subset=[target_column]).sort_values(
                [target_column, "opportunity_score", "signal_score", "ticker"],
                ascending=[True, False, False, True],
            )
            excluded = excluded.dropna(subset=[target_column]).sort_values(
                [target_column, "opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, False, True],
            )
            if selected.empty or excluded.empty:
                continue
            worst_selected = selected.iloc[0]
            best_excluded = excluded.iloc[0]
            gap = float(best_excluded[target_column]) - float(worst_selected[target_column])
            if gap <= 0.0:
                continue
            rows.append(
                {
                    "scan_date": str(scan_date),
                    "strategy_slot": str(strategy_slot),
                    "selected_ticker": str(worst_selected["ticker"]),
                    "selected_rank": int(worst_selected["selected_rank"]) if pd.notna(worst_selected.get("selected_rank")) else None,
                    "selected_target": float(worst_selected[target_column]),
                    "selected_opportunity": float(worst_selected["opportunity_score"]),
                    "excluded_ticker": str(best_excluded["ticker"]),
                    "excluded_target": float(best_excluded[target_column]),
                    "excluded_opportunity": float(best_excluded["opportunity_score"]),
                    "gap": gap,
                }
            )
        if not rows:
            lines.append("- none")
            lines.append("")
            return lines
        ranked = sorted(rows, key=lambda row: (row["gap"], row["excluded_target"]), reverse=True)[:max_rows]
        for row in ranked:
            rank_label = f" rank={row['selected_rank']}" if row["selected_rank"] is not None else ""
            lines.append(f"#### {row['scan_date']} {row['strategy_slot']}")
            lines.append(
                f"- selected: {row['selected_ticker']}{rank_label} "
                f"target={row['selected_target']:.4f} opportunity={row['selected_opportunity']:.4f}"
            )
            lines.append(
                f"- better_excluded: {row['excluded_ticker']} "
                f"target={row['excluded_target']:.4f} opportunity={row['excluded_opportunity']:.4f}"
            )
            lines.append(f"- performance_gap: {row['gap']:.4f}")
            lines.append("")
        return lines

    def _render_recent_repeat_drags(
        self,
        selected: pd.DataFrame,
        *,
        target_column: str,
        max_rows: int,
    ) -> list[str]:
        lines = ["### Repeated Recent Drags"]
        if selected.empty or target_column not in selected.columns:
            lines.append("- none")
            lines.append("")
            return lines
        grouped = (
            selected.groupby(["ticker", "strategy_slot", "sector"], dropna=False)
            .agg(
                picks=(target_column, "count"),
                mean_target=(target_column, "mean"),
                median_target=(target_column, "median"),
                hit_rate=(target_column, lambda values: float((pd.Series(values).astype(float) > 0.0).mean())),
            )
            .reset_index()
        )
        drags = grouped[(grouped["picks"] >= 2) & (grouped["mean_target"] < 0.0)].copy()
        drags = drags.sort_values(["mean_target", "picks", "ticker"], ascending=[True, False, True]).head(max_rows)
        if drags.empty:
            lines.append("- none")
            lines.append("")
            return lines
        for row in drags.itertuples(index=False):
            lines.append(
                f"- {row.ticker} ({row.strategy_slot}): mean_target={float(row.mean_target):.4f} "
                f"median_target={float(row.median_target):.4f} hit_rate={float(row.hit_rate):.4f} picks={int(row.picks)}"
            )
        lines.append("")
        return lines

    def _render_mediocre_setup_diagnostics(
        self,
        all_candidates: pd.DataFrame,
        *,
        target_column: str,
        max_scan_dates: int = 20,
        max_rows: int = 8,
    ) -> list[str]:
        lines = ["## Mediocre Setup Diagnostics", ""]
        if target_column not in all_candidates.columns:
            lines.append(f"Missing target column: {target_column}")
            lines.append("")
            return lines
        matured = all_candidates.dropna(subset=[target_column]).copy()
        if matured.empty:
            lines.append("No matured scan candidates are available yet.")
            lines.append("")
            return lines
        recent_scan_dates = (
            matured.loc[matured["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(max_scan_dates)
            .tolist()
        )
        if not recent_scan_dates:
            lines.append("No matured selected scan dates are available yet.")
            lines.append("")
            return lines
        selected = matured[
            matured["scan_date"].astype(str).isin(recent_scan_dates)
            & (matured["selected"].astype(int) == 1)
        ].copy()
        if selected.empty:
            lines.append("No matured selected rows are available yet.")
            lines.append("")
            return lines
        selected["signal_score"] = pd.to_numeric(selected["signal_score"], errors="coerce")
        selected["setup_quality_score"] = pd.to_numeric(selected["setup_quality_score"], errors="coerce")
        selected["opportunity_score"] = pd.to_numeric(selected["opportunity_score"], errors="coerce")
        signal_q25 = float(selected["signal_score"].quantile(0.25))
        setup_q25 = float(selected["setup_quality_score"].quantile(0.25))
        opp_q25 = float(selected["opportunity_score"].quantile(0.25))
        low_signal = selected[selected["signal_score"] <= signal_q25].copy()
        low_setup = selected[selected["setup_quality_score"] <= setup_q25].copy()
        low_opportunity = selected[selected["opportunity_score"] <= opp_q25].copy()
        low_both = selected[
            (selected["signal_score"] <= signal_q25)
            & (selected["setup_quality_score"] <= setup_q25)
        ].copy()

        lines.append(f"- target_column: {target_column}")
        lines.append(f"- recent_scan_dates: {len(recent_scan_dates)}")
        lines.append(f"- date_range: {recent_scan_dates[-1]} -> {recent_scan_dates[0]}")
        lines.append(f"- signal_score_q25: {signal_q25:.4f}")
        lines.append(f"- setup_quality_q25: {setup_q25:.4f}")
        lines.append(f"- opportunity_score_q25: {opp_q25:.4f}")
        lines.append("")
        lines.extend(self._render_quality_group("All selected ideas", selected, target_column))
        lines.extend(self._render_quality_group("Low-signal selected ideas", low_signal, target_column))
        lines.extend(self._render_quality_group("Low-setup selected ideas", low_setup, target_column))
        lines.extend(self._render_quality_group("Low-opportunity selected ideas", low_opportunity, target_column))
        lines.extend(self._render_quality_group("Low-signal and low-setup selected ideas", low_both, target_column))
        lines.extend(
            self._render_repeated_quality_drags(
                low_both if not low_both.empty else low_signal,
                target_column=target_column,
                max_rows=max_rows,
            )
        )
        return lines

    def _render_quality_group(self, title: str, frame: pd.DataFrame, target_column: str) -> list[str]:
        lines = [f"### {title}"]
        if frame.empty or target_column not in frame.columns:
            lines.append("- none")
            lines.append("")
            return lines
        series = frame[target_column].dropna().astype(float)
        if series.empty:
            lines.append("- none")
            lines.append("")
            return lines
        lines.append(f"- rows: {len(series.index)}")
        lines.append(f"- mean_target: {series.mean():.6f}")
        lines.append(f"- median_target: {series.median():.6f}")
        lines.append(f"- hit_rate: {(series > 0.0).mean():.4f}")
        lines.append("")
        return lines

    def _render_repeated_quality_drags(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str,
        max_rows: int,
    ) -> list[str]:
        lines = ["### Repeated Mediocre Drags"]
        if frame.empty or target_column not in frame.columns:
            lines.append("- none")
            lines.append("")
            return lines
        grouped = (
            frame.groupby(["ticker", "strategy_slot"], dropna=False)
            .agg(
                picks=(target_column, "count"),
                mean_target=(target_column, "mean"),
                median_signal_score=("signal_score", "median"),
                median_setup_quality=("setup_quality_score", "median"),
                hit_rate=(target_column, lambda values: float((pd.Series(values).astype(float) > 0.0).mean())),
            )
            .reset_index()
        )
        drags = grouped[(grouped["picks"] >= 2) & (grouped["mean_target"] < 0.0)].copy()
        drags = drags.sort_values(["mean_target", "picks", "ticker"], ascending=[True, False, True]).head(max_rows)
        if drags.empty:
            lines.append("- none")
            lines.append("")
            return lines
        for row in drags.itertuples(index=False):
            lines.append(
                f"- {row.ticker} ({row.strategy_slot}): mean_target={float(row.mean_target):.4f} "
                f"hit_rate={float(row.hit_rate):.4f} picks={int(row.picks)} "
                f"median_signal_score={float(row.median_signal_score):.2f} "
                f"median_setup_quality={float(row.median_setup_quality):.4f}"
            )
        lines.append("")
        return lines

    def _render_regime_attribution(
        self,
        all_candidates: pd.DataFrame,
        *,
        target_column: str,
        recent_scan_dates: int,
    ) -> list[str]:
        lines = ["## Regime Attribution", ""]
        if target_column not in all_candidates.columns:
            lines.append(f"Missing target column: {target_column}")
            lines.append("")
            return lines
        matured = all_candidates.dropna(subset=[target_column]).copy()
        if matured.empty:
            lines.append("No matured scan candidates are available yet.")
            lines.append("")
            return lines
        recent_dates = (
            matured.loc[matured["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(recent_scan_dates)
            .tolist()
        )
        if not recent_dates:
            lines.append("No matured selected scan dates are available yet.")
            lines.append("")
            return lines
        frame = matured[matured["scan_date"].astype(str).isin(recent_dates)].copy()
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- recent_scan_dates: {len(recent_dates)}")
        lines.append(f"- date_range: {recent_dates[-1]} -> {recent_dates[0]}")
        lines.append("- purpose: compare selected vs excluded outcomes by market and sector regime")
        lines.append("")
        bucket_specs = (
            ("SPY 200d Regime", "spy_regime_green", self._boolean_regime_label),
            ("QQQ 200d Regime", "qqq_regime_green", self._boolean_regime_label),
            ("Sector ETF 200d Regime", "sector_regime_green", self._boolean_regime_label),
            ("Sector Breadth", "sector_breadth_bucket", lambda value: str(value)),
        )
        for title, column, formatter in bucket_specs:
            lines.extend(self._render_regime_bucket(title, frame, target_column=target_column, bucket_column=column, label_formatter=formatter))
        return lines

    def _render_slot_internal_attribution(
        self,
        all_candidates: pd.DataFrame,
        *,
        target_column: str,
        recent_scan_dates: int,
        max_dimensions: int,
    ) -> list[str]:
        lines = ["## Slot Internal Attribution", ""]
        if target_column not in all_candidates.columns:
            lines.append(f"Missing target column: {target_column}")
            lines.append("")
            return lines
        matured = all_candidates.dropna(subset=[target_column]).copy()
        if matured.empty:
            lines.append("No matured scan candidates are available yet.")
            lines.append("")
            return lines
        recent_dates = (
            matured.loc[matured["selected"].astype(int) == 1, "scan_date"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(recent_scan_dates)
            .tolist()
        )
        if not recent_dates:
            lines.append("No matured selected scan dates are available yet.")
            lines.append("")
            return lines
        frame = matured[matured["scan_date"].astype(str).isin(recent_dates)].copy()
        lines.append(f"- target_column: {target_column}")
        lines.append(f"- recent_scan_dates: {len(recent_dates)}")
        lines.append(f"- date_range: {recent_dates[-1]} -> {recent_dates[0]}")
        lines.append("- method: within each slot, compare high-half vs low-half of each feature on matured eligible rows")
        lines.append("")
        candidate_dimensions = (
            ("signal_score", "signal score"),
            ("setup_quality_score", "setup quality"),
            ("freshness_score", "freshness"),
            ("expected_alpha_score", "expected alpha"),
            ("sector_pct_above_200", "sector breadth 200d"),
            ("sector_pct_above_50", "sector breadth 50d"),
            ("distance_above_20d_high", "breakout extension"),
            ("sma_200_dist", "distance above 200d"),
            ("roc_63", "63d momentum"),
            ("roc_126", "126d momentum"),
            ("relative_strength_index_vs_spy", "RS vs SPY"),
            ("relative_strength_index_vs_qqq", "RS vs QQQ"),
            ("avg_abs_gap_pct_20", "avg gap"),
            ("max_gap_down_pct_60", "worst gap down"),
        )
        for slot, slot_frame in frame.groupby("strategy_slot", sort=True):
            lines.extend(
                self._render_slot_internal_attribution_for_slot(
                    str(slot),
                    slot_frame,
                    target_column=target_column,
                    candidate_dimensions=candidate_dimensions,
                    max_dimensions=max_dimensions,
                )
            )
        return lines

    def _render_slot_internal_attribution_for_slot(
        self,
        slot: str,
        frame: pd.DataFrame,
        *,
        target_column: str,
        candidate_dimensions: tuple[tuple[str, str], ...],
        max_dimensions: int,
    ) -> list[str]:
        lines = [f"### {slot}"]
        lines.append(f"- eligible_rows: {int(len(frame.index))}")
        lines.append(f"- selected_rows: {int((frame['selected'].astype(int) == 1).sum())}")
        metrics: list[dict[str, object]] = []
        for column, label in candidate_dimensions:
            if column not in frame.columns:
                continue
            subset = frame.copy()
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
            subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
            subset = subset.dropna(subset=[column, target_column]).copy()
            if len(subset.index) < 20 or subset[column].nunique(dropna=True) < 4:
                continue
            threshold = float(subset[column].median())
            low = subset[subset[column] < threshold].copy()
            high = subset[subset[column] >= threshold].copy()
            if low.empty or high.empty:
                continue
            low_target = low[target_column].astype(float)
            high_target = high[target_column].astype(float)
            low_selected = low.loc[low["selected"].astype(int) == 1, target_column].dropna().astype(float)
            high_selected = high.loc[high["selected"].astype(int) == 1, target_column].dropna().astype(float)
            metrics.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "eligible_spread": float(high_target.mean() - low_target.mean()),
                    "selected_spread": (
                        float(high_selected.mean() - low_selected.mean())
                        if not high_selected.empty and not low_selected.empty
                        else float("nan")
                    ),
                    "rows": int(len(subset.index)),
                }
            )
        if not metrics:
            lines.append("- no slot-internal feature buckets had enough support")
            lines.append("")
            return lines
        metrics_frame = pd.DataFrame(metrics)
        positive = metrics_frame.sort_values(["eligible_spread", "rows", "label"], ascending=[False, False, True]).head(max_dimensions)
        negative = metrics_frame.sort_values(["eligible_spread", "rows", "label"], ascending=[True, False, True]).head(max_dimensions)
        lines.append("#### Strongest Positive Discriminators")
        for row in positive.itertuples(index=False):
            selected_spread = "nan" if pd.isna(row.selected_spread) else f"{float(row.selected_spread):.6f}"
            lines.append(
                f"- {row.label}: threshold={float(row.threshold):.4f} "
                f"eligible_high_minus_low={float(row.eligible_spread):.6f} "
                f"selected_high_minus_low={selected_spread} rows={int(row.rows)}"
            )
        lines.append("")
        lines.append("#### Strongest Negative Discriminators")
        for row in negative.itertuples(index=False):
            selected_spread = "nan" if pd.isna(row.selected_spread) else f"{float(row.selected_spread):.6f}"
            lines.append(
                f"- {row.label}: threshold={float(row.threshold):.4f} "
                f"eligible_high_minus_low={float(row.eligible_spread):.6f} "
                f"selected_high_minus_low={selected_spread} rows={int(row.rows)}"
            )
        lines.append("")
        return lines

    def _render_regime_bucket(
        self,
        title: str,
        frame: pd.DataFrame,
        *,
        target_column: str,
        bucket_column: str,
        label_formatter,
    ) -> list[str]:
        lines = [f"### {title}"]
        if bucket_column not in frame.columns:
            lines.append("- unavailable")
            lines.append("")
            return lines
        working = frame.copy()
        if bucket_column != "sector_breadth_bucket":
            working[bucket_column] = working[bucket_column].map(label_formatter)
        else:
            working[bucket_column] = working[bucket_column].astype(str)
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=[target_column])
        if working.empty:
            lines.append("- unavailable")
            lines.append("")
            return lines
        bucket_values = [value for value in working[bucket_column].dropna().astype(str).drop_duplicates().tolist() if value]
        preferred_order = ["green", "red", "high", "mixed", "low", "unknown"]
        order_lookup = {value: index for index, value in enumerate(preferred_order)}
        bucket_values = sorted(bucket_values, key=lambda value: (order_lookup.get(value, 99), value))
        for bucket_value in bucket_values:
            subset = working[working[bucket_column].astype(str) == str(bucket_value)].copy()
            selected = subset[subset["selected"].astype(int) == 1].copy()
            excluded = subset[subset["selected"].astype(int) == 0].copy()
            lines.append(f"#### {bucket_value}")
            lines.extend(self._render_recent_selection_group("Selected ideas", selected, target_column))
            lines.extend(self._render_recent_selection_group("Excluded eligible ideas", excluded, target_column))
            slot_summary = self._regime_slot_summary(subset, target_column=target_column)
            if slot_summary.empty:
                lines.append("##### Slot Breakdown")
                lines.append("- none")
                lines.append("")
                continue
            lines.append("##### Slot Breakdown")
            for row in slot_summary.itertuples(index=False):
                selected_mean = "nan" if pd.isna(row.selected_mean_target) else f"{float(row.selected_mean_target):.6f}"
                excluded_mean = "nan" if pd.isna(row.excluded_mean_target) else f"{float(row.excluded_mean_target):.6f}"
                lines.append(
                    f"- {row.strategy_slot}: selected_mean_target={selected_mean} "
                    f"excluded_mean_target={excluded_mean} "
                    f"selected_rows={int(row.selected_rows)} excluded_rows={int(row.excluded_rows)}"
                )
            lines.append("")
        return lines

    def _regime_slot_summary(self, frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for slot, slot_frame in frame.groupby("strategy_slot", sort=True):
            selected = slot_frame[slot_frame["selected"].astype(int) == 1][target_column].dropna().astype(float)
            excluded = slot_frame[slot_frame["selected"].astype(int) == 0][target_column].dropna().astype(float)
            rows.append(
                {
                    "strategy_slot": str(slot),
                    "selected_mean_target": float(selected.mean()) if not selected.empty else float("nan"),
                    "excluded_mean_target": float(excluded.mean()) if not excluded.empty else float("nan"),
                    "selected_rows": int(len(selected.index)),
                    "excluded_rows": int(len(excluded.index)),
                }
            )
        if not rows:
            return pd.DataFrame()
        summary = pd.DataFrame(rows)
        return summary.sort_values(["selected_mean_target", "selected_rows", "strategy_slot"], ascending=[False, False, True])

    def _boolean_regime_label(self, value) -> str:
        if pd.isna(value):
            return "unknown"
        return "green" if bool(value) else "red"

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
            lines.append(f"- validation_method: {report.validation_method}")
            lines.append(f"- embargo_days: {report.embargo_days}")
            lines.append(f"- validation_blocks: {report.validation_blocks}")
            lines.append(f"- train_rows: {report.train_rows}")
            lines.append(f"- validation_rows: {report.validation_rows}")
            lines.append(f"- train_dates: {report.train_dates}")
            lines.append(f"- validation_dates: {report.validation_dates}")
            lines.append("")
            return lines
        lines.append(f"- target_column: {report.target_column}")
        lines.append(f"- validation_method: {report.validation_method}")
        lines.append(f"- embargo_days: {report.embargo_days}")
        lines.append(f"- validation_blocks: {report.validation_blocks}")
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
        lines.append("### Ranker Quintile Tear Sheet")
        if report.quintile_summaries:
            for summary in report.quintile_summaries:
                mean_score = f"{summary.mean_ranker_score:.6f}" if math.isfinite(summary.mean_ranker_score) else "nan"
                mean_target = f"{summary.mean_target:.6f}" if math.isfinite(summary.mean_target) else "nan"
                hit_rate = f"{summary.hit_rate:.4f}" if math.isfinite(summary.hit_rate) else "nan"
                lines.append(
                    f"- Q{summary.quintile}: mean_target={mean_target} "
                    f"hit_rate={hit_rate} mean_ranker_score={mean_score} "
                    f"rows={summary.row_count} dates={summary.date_count}"
                )
        else:
            lines.append("- No quintile calibration rows available.")
        if math.isfinite(report.q1_q5_spread):
            lines.append(f"- q1_minus_q5_spread: {report.q1_q5_spread:.6f}")
        else:
            lines.append("- q1_minus_q5_spread: nan")
        turnover_mean = f"{report.learned_turnover_mean:.4f}" if math.isfinite(report.learned_turnover_mean) else "nan"
        lines.append(f"- learned_top_n_turnover_mean: {turnover_mean}")
        lines.append(f"- learned_top_n_turnover_pairs: {report.learned_turnover_pairs}")
        lines.append("")
        lines.append("### Daily IC")
        daily_ic_mean = f"{report.daily_ic_mean:.6f}" if math.isfinite(report.daily_ic_mean) else "nan"
        daily_ic_std = f"{report.daily_ic_std:.6f}" if math.isfinite(report.daily_ic_std) else "nan"
        daily_ic_t_stat = f"{report.daily_ic_t_stat:.6f}" if math.isfinite(report.daily_ic_t_stat) else "nan"
        lines.append(f"- ic_mean: {daily_ic_mean}")
        lines.append(f"- ic_std: {daily_ic_std}")
        lines.append(f"- ic_t_stat: {daily_ic_t_stat}")
        lines.append(f"- ic_dates: {report.daily_ic_dates}")
        lines.append("")
        lines.extend(self._render_ranker_breakdowns("Slot Breakdown", report.slot_breakdowns))
        lines.extend(self._render_ranker_breakdowns("Sector Breakdown", report.sector_breakdowns))
        if report.latest_scan_date is not None:
            lines.append(f"### Latest Validation Date: {report.latest_scan_date}")
            lines.append(f"- learned_tickers: {', '.join(report.latest_learned_tickers) if report.latest_learned_tickers else 'none'}")
            lines.append(f"- handcrafted_tickers: {', '.join(report.latest_handcrafted_tickers) if report.latest_handcrafted_tickers else 'none'}")
            lines.append(f"- runtime_tickers: {', '.join(report.latest_runtime_tickers) if report.latest_runtime_tickers else 'none'}")
            lines.append("")
        return lines

    def _render_ranker_breakdowns(self, title: str, breakdowns) -> list[str]:
        lines = [f"### {title}"]
        if not breakdowns:
            lines.append("- none")
            lines.append("")
            return lines
        for summary in breakdowns:
            top_mean = f"{summary.top_quintile_mean_target:.6f}" if math.isfinite(summary.top_quintile_mean_target) else "nan"
            bottom_mean = f"{summary.bottom_quintile_mean_target:.6f}" if math.isfinite(summary.bottom_quintile_mean_target) else "nan"
            spread = f"{summary.spread:.6f}" if math.isfinite(summary.spread) else "nan"
            lines.append(
                f"- {summary.group_value}: q1_mean={top_mean} q5_mean={bottom_mean} "
                f"spread={spread} rows={summary.row_count} dates={summary.date_count}"
            )
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
        embargo_days = ranker._infer_target_horizon_days(default=10)
        folds = ranker._purged_walk_forward_folds(
            labeled,
            train_ratio=0.7,
            embargo_days=embargo_days,
            max_validation_blocks=5,
        )
        if not folds:
            return {
                "available": False,
                "reason": "Not enough validation history for slot-level selector attribution.",
                "target_column": target_column,
                "train_rows": 0,
                "train_dates": 0,
                "validation_dates": 0,
                "validation_blocks": 0,
                "embargo_days": embargo_days,
            }
        scored_frames: list[pd.DataFrame] = []
        fold_train_rows: list[int] = []
        fold_train_dates: list[int] = []
        for fold in folds:
            train_frame = fold["train"].copy()
            validation_frame = fold["validation"].copy()
            train_dates = int(train_frame["scan_date"].nunique()) if not train_frame.empty else 0
            if (
                train_frame.empty
                or validation_frame.empty
                or len(train_frame.index) < ranker.min_train_rows
                or train_dates < ranker.min_train_dates
            ):
                continue
            ranker.fit(train_frame)
            scored_fold = ranker.score(validation_frame)
            scored_fold["validation_fold"] = int(fold["fold_index"])
            scored_frames.append(scored_fold)
            fold_train_rows.append(len(train_frame.index))
            fold_train_dates.append(train_dates)
        if not scored_frames:
            validation_dates = len({date_value for fold in folds for date_value in fold["validation_dates"]})
            return {
                "available": False,
                "reason": "Not enough purged walk-forward folds cleared the minimum train support.",
                "target_column": target_column,
                "train_rows": int(round(float(np.mean(fold_train_rows)))) if fold_train_rows else 0,
                "train_dates": int(round(float(np.mean(fold_train_dates)))) if fold_train_dates else 0,
                "validation_dates": validation_dates,
                "validation_blocks": len(folds),
                "embargo_days": embargo_days,
            }
        scored = pd.concat(scored_frames, ignore_index=True)
        train_rows = int(round(float(np.mean(fold_train_rows)))) if fold_train_rows else 0
        train_dates = int(round(float(np.mean(fold_train_dates)))) if fold_train_dates else 0
        validation_dates = int(scored["scan_date"].nunique()) if not scored.empty else 0
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
            "validation_blocks": len(scored["validation_fold"].drop_duplicates()) if "validation_fold" in scored.columns else len(folds),
            "embargo_days": embargo_days,
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
            if "validation_blocks" in report:
                lines.append(f"- validation_blocks: {int(report.get('validation_blocks', 0))}")
            if "embargo_days" in report:
                lines.append(f"- embargo_days: {int(report.get('embargo_days', 0))}")
            if "target_column" in report:
                lines.append(f"- target_column: {report.get('target_column')}")
            lines.append("")
            return lines
        lines.append(f"- target_column: {report['target_column']}")
        lines.append("- validation_method: purged_walk_forward")
        lines.append(f"- embargo_days: {int(report['embargo_days'])}")
        lines.append(f"- validation_blocks: {int(report['validation_blocks'])}")
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

    def _date_select_candidates(self, frame: pd.DataFrame, *, score_column: str, scan_policy: ScanPolicy) -> pd.DataFrame:
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
        if ranked.empty:
            return ranked
        if score_column == "random_selector_score":
            ranked["selection_score"] = ranked.apply(
                lambda row: self._random_selector_score(scan_date=str(row["scan_date"]), ticker=str(row["ticker"]), slot=str(row["strategy_slot"])),
                axis=1,
            )
        else:
            if score_column in ranked.columns:
                score_series = pd.to_numeric(ranked[score_column], errors="coerce")
            else:
                score_series = pd.Series(float("nan"), index=ranked.index, dtype=float)
            ranked["selection_score"] = score_series.fillna(float("-inf"))
        selector = ScanService(self.db_manager, email_sender=lambda subject, html_body, settings: None)
        return selector._apply_portfolio_caps(ranked, scan_policy).copy()

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

    def _selection_metrics(
        self,
        scan_date,
        method: str,
        frame: pd.DataFrame,
        target_column: str,
        *,
        strategy_slot: str | None = None,
    ) -> dict[str, object]:
        if frame.empty or target_column not in frame.columns:
            return {
                "scan_date": str(scan_date),
                "method": method,
                "strategy_slot": strategy_slot,
                "mean_target": np.nan,
                "hit_rate": np.nan,
                "pick_count": 0,
            }
        series = frame[target_column].dropna().astype(float)
        if series.empty:
            return {
                "scan_date": str(scan_date),
                "method": method,
                "strategy_slot": strategy_slot,
                "mean_target": np.nan,
                "hit_rate": np.nan,
                "pick_count": int(len(frame.index)),
            }
        return {
            "scan_date": str(scan_date),
            "method": method,
            "strategy_slot": strategy_slot,
            "mean_target": float(series.mean()),
            "hit_rate": float((series > 0.0).mean()),
            "pick_count": int(len(series.index)),
        }

    def _frame_mean_target(self, frame: pd.DataFrame, target_column: str) -> float:
        if frame.empty or target_column not in frame.columns:
            return float("nan")
        series = frame[target_column].dropna().astype(float)
        return float(series.mean()) if not series.empty else float("nan")

    def _random_selector_score(self, *, scan_date: str, ticker: str, slot: str) -> float:
        digest = hashlib.sha256(f"{scan_date}|{slot}|{ticker}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(16**16)
