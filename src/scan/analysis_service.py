from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json

import numpy as np
import pandas as pd

from src.scan.service import ScanPolicy, ScanService
from src.settings import load_feature_config
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


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

        candidates = self._load_candidates(scan_date=scan_date)
        if candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan` or `sq scan-analysis --refresh` first.")
        selected_scan_date = str(candidates["scan_date"].iloc[0])
        selected = candidates[candidates["selected"] == 1].copy()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        candidates["details"] = candidates["details_json"].map(lambda raw: json.loads(raw) if raw else {})

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
        lines.extend(self._render_selected_candidates(selected))
        lines.extend(self._render_excluded_candidates(candidates, scan_policy))
        lines.extend(self._render_forward_attribution(candidates, selected, selected_scan_date, horizons))

        report_path = self.db_manager.paths.reports_dir / "scan_analysis.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return ScanAnalysisReport(
            output_path=str(report_path),
            scan_date=selected_scan_date,
            candidate_count=len(candidates.index),
            selected_count=len(selected.index),
            refreshed=refresh,
        )

    def _load_candidates(self, *, scan_date: str | None) -> pd.DataFrame:
        if scan_date is not None:
            return self.db_manager.load_scan_candidates(scan_date=scan_date)
        frame = self.db_manager.load_scan_candidates()
        if frame.empty:
            return frame
        latest_date = str(frame["scan_date"].max())
        return frame[frame["scan_date"] == latest_date].copy().reset_index(drop=True)

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
            values = frame[column].astype(float)
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
        ordered = selected.sort_values(
            ["opportunity_score", "signal_score", "ticker"],
            ascending=[False, False, True],
        )
        for row in ordered.itertuples(index=False):
            lines.append(f"### {row.ticker}")
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
            already_owned = bool(details.get("already_owned", False))
            if already_owned:
                reason = "already owned (setup still valid)" if float(details.get("pre_penalty_opportunity_score", 0.0)) >= scan_policy.min_opportunity_score else "already owned"
            elif float(row.opportunity_score) < scan_policy.min_opportunity_score:
                reason = "below opportunity threshold"
            else:
                reason = "portfolio cap / overlap selection loss"
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

    def _render_forward_attribution(
        self,
        candidates: pd.DataFrame,
        selected: pd.DataFrame,
        scan_date: str,
        horizons: tuple[int, ...],
    ) -> list[str]:
        lines = ["## Forward Attribution", ""]
        returns = self._build_forward_returns(candidates, scan_date=scan_date, horizons=horizons)
        if returns.empty:
            lines.append("No forward return windows are fully available yet for this scan date.")
            lines.append("")
            return lines
        candidates = candidates.merge(returns, on="ticker", how="left")
        selected = selected.merge(returns, on="ticker", how="left") if not selected.empty else selected.copy()
        if not selected.empty:
            top_ranked = candidates.sort_values(
                ["opportunity_score", "signal_score", "ticker"],
                ascending=[False, False, True],
            ).head(len(selected.index)).copy()
        else:
            top_ranked = candidates.iloc[0:0].copy()

        for horizon in horizons:
            column = f"fwd_return_{horizon}d"
            if column not in candidates.columns:
                continue
            lines.append(f"### {horizon}-Day Forward Return")
            lines.extend(self._render_return_group("Eligible universe", candidates, column))
            lines.extend(self._render_return_group("Selected ideas", selected, column))
            lines.extend(self._render_return_group("Top raw opportunity ideas", top_ranked, column))
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

    def _build_forward_returns(
        self,
        candidates: pd.DataFrame,
        *,
        scan_date: str,
        horizons: tuple[int, ...],
    ) -> pd.DataFrame:
        tickers = sorted(set(candidates["ticker"].astype(str)))
        try:
            history = self.db_manager.load_price_history(tickers)
        except Exception as exc:
            self.logger.warning("Unable to load price history for scan attribution: %s", exc)
            return pd.DataFrame(columns=["ticker"])
        if history.empty:
            return pd.DataFrame(columns=["ticker"])
        history["date"] = pd.to_datetime(history["date"]).dt.normalize()
        target_date = pd.Timestamp(scan_date).normalize()
        rows: list[dict[str, object]] = []
        for ticker, group in history.groupby("ticker", sort=False):
            ordered = group.sort_values("date").reset_index(drop=True)
            matches = ordered.index[ordered["date"] == target_date].tolist()
            if not matches:
                continue
            index = matches[-1]
            entry_price = float(ordered.loc[index, "adj_close"])
            payload: dict[str, object] = {"ticker": str(ticker)}
            available = False
            for horizon in horizons:
                column = f"fwd_return_{horizon}d"
                future_index = index + int(horizon)
                if future_index >= len(ordered.index):
                    payload[column] = np.nan
                    continue
                future_price = float(ordered.loc[future_index, "adj_close"])
                payload[column] = (future_price / entry_price) - 1.0
                available = True
            if available:
                rows.append(payload)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker"])
