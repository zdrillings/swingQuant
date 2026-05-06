from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.settings import get_settings, load_feature_config
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot, overlay_price_history
from src.utils.sizing import compute_position_size
from src.utils.strategy import ProductionStrategy, load_active_strategies


@dataclass(frozen=True)
class ScanReport:
    candidate_count: int
    emailed: bool


@dataclass(frozen=True)
class ScanPolicy:
    max_candidates_total: int
    max_candidates_per_slot: int
    max_candidates_per_sector: int
    pre_cap_candidates_per_slot: int

    @classmethod
    def from_config(cls, config: dict) -> "ScanPolicy":
        policy = config.get("scan_policy", {})
        return cls(
            max_candidates_total=int(policy.get("max_candidates_total", 6)),
            max_candidates_per_slot=int(policy.get("max_candidates_per_slot", 3)),
            max_candidates_per_sector=int(policy.get("max_candidates_per_sector", 2)),
            pre_cap_candidates_per_slot=int(policy.get("pre_cap_candidates_per_slot", 5)),
        )


class ScanService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
        email_sender=send_html_email,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.email_sender = email_sender
        self.logger = get_logger("scan")

    def run(self) -> ScanReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        settings = get_settings()
        config = load_feature_config()
        scan_policy = ScanPolicy.from_config(config)
        universe_rows = self.db_manager.list_universe_rows(active_only=True)
        if not universe_rows:
            raise ValueError("Universe is empty. Run `sq sync` first.")

        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        base_history = self.db_manager.load_price_history(tickers)
        recent_history = self._download_recent_daily_history(tickers)
        analysis_frame, _ = build_analysis_frame(
            overlay_price_history(base_history, recent_history),
            universe_rows,
        )
        snapshot = latest_snapshot(analysis_frame)
        snapshot = snapshot[
            snapshot["ticker"].isin(universe_tickers)
            & snapshot["regime_green"].fillna(False)
        ].copy()
        candidate_frames: list[pd.DataFrame] = []
        for slot, strategy in strategies.items():
            scoped_snapshot = self._scope_snapshot(snapshot, strategy)
            if scoped_snapshot.empty:
                continue
            candidates = filter_signal_candidates(scoped_snapshot, strategy.indicators)
            if candidates.empty:
                continue
            if "signal_score" not in candidates.columns:
                candidates["signal_score"] = 0.0
            candidates = candidates.sort_values(
                ["signal_score", "md_volume_30d", "ticker"],
                ascending=[False, False, True],
            ).head(scan_policy.pre_cap_candidates_per_slot).copy()
            candidates["strategy_slot"] = slot
            candidates["strategy_sector"] = strategy.sector
            candidates_frames = candidates.apply(
                lambda row: compute_position_size(
                    price=float(row["adj_close"]),
                    exit_rules=strategy.exit_rules,
                    settings=settings,
                    entry_atr=float(row["atr_14"]) if pd.notna(row.get("atr_14")) else None,
                ),
                axis=1,
            )
            candidates["shares"] = candidates_frames
            candidate_frames.append(candidates)

        if not candidate_frames:
            return ScanReport(candidate_count=0, emailed=False)
        candidates = pd.concat(candidate_frames, ignore_index=True)
        candidates = self._apply_portfolio_caps(candidates, scan_policy)
        candidates = candidates.sort_values(
            ["strategy_slot", "signal_score", "md_volume_30d", "ticker"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        html = self._build_email_html(candidates, scan_policy)
        self.email_sender(
            subject="Evening Brief",
            html_body=html,
            settings=settings,
        )
        return ScanReport(candidate_count=len(candidates.index), emailed=True)

    def _scope_snapshot(self, snapshot: pd.DataFrame, strategy: ProductionStrategy) -> pd.DataFrame:
        if strategy.sector == "ALL":
            return snapshot.copy()
        return snapshot.loc[snapshot["sector"] == strategy.sector].copy()

    def _download_recent_daily_history(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        start_date = date.today() - timedelta(days=10)
        for ticker_batch in chunked(tickers, 50):
            raw_batch = self.market_data_client.download_daily_history(ticker_batch, start_date)
            for ticker in ticker_batch:
                history = extract_ticker_history(raw_batch, ticker)
                if not history.empty:
                    frames.append(history)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _apply_portfolio_caps(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> pd.DataFrame:
        ranked = candidates.sort_values(
            ["signal_score", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        selected_indices: list[int] = []
        slot_counts: dict[str, int] = {}
        sector_counts: dict[str, int] = {}
        for candidate in ranked.itertuples():
            if len(selected_indices) >= scan_policy.max_candidates_total:
                break
            slot = str(candidate.strategy_slot)
            sector = str(candidate.sector)
            if slot_counts.get(slot, 0) >= scan_policy.max_candidates_per_slot:
                continue
            if sector_counts.get(sector, 0) >= scan_policy.max_candidates_per_sector:
                continue
            selected_indices.append(int(candidate.Index))
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if not selected_indices:
            return ranked.iloc[0:0].copy()
        return ranked.loc[selected_indices].copy()

    def _build_email_html(self, candidates: pd.DataFrame, scan_policy: ScanPolicy) -> str:
        sections: list[str] = []
        ordered_slots = (
            candidates.groupby(["strategy_slot", "strategy_sector"], sort=False)["signal_score"]
            .max()
            .reset_index()
            .sort_values(["signal_score", "strategy_slot"], ascending=[False, True])
        )
        for slot_row in ordered_slots.itertuples(index=False):
            slot_candidates = candidates.loc[candidates["strategy_slot"] == slot_row.strategy_slot].copy()
            slot_candidates = slot_candidates.sort_values(
                ["signal_score", "md_volume_30d", "ticker"],
                ascending=[False, False, True],
            )
            rows = []
            for candidate in slot_candidates.itertuples(index=False):
                rows.append(
                    "<tr>"
                    f"<td>{candidate.ticker}</td>"
                    f"<td>{candidate.sector}</td>"
                    f"<td>{candidate.regime_etf} Green</td>"
                    f"<td>{candidate.signal_score:.1f}</td>"
                    f"<td>{candidate.adj_close:.2f}</td>"
                    f"<td>{candidate.shares}</td>"
                    "</tr>"
                )
            sections.append(
                "<section>"
                f"<h2>Slot: {slot_row.strategy_slot}</h2>"
                f"<p>Strategy sector scope: {slot_row.strategy_sector} | Candidates: {len(slot_candidates.index)}</p>"
                "<table border='1' cellpadding='6' cellspacing='0'>"
                "<tr><th>Ticker</th><th>Sector</th><th>Regime</th><th>Signal Score</th><th>Adj Close</th><th>Shares</th></tr>"
                f"{''.join(rows)}"
                "</table>"
                "</section>"
            )
        return (
            "<html><body>"
            "<h1>Evening Brief</h1>"
            f"<p>Portfolio caps: total={scan_policy.max_candidates_total}, per_slot={scan_policy.max_candidates_per_slot}, per_sector={scan_policy.max_candidates_per_sector}</p>"
            f"{''.join(sections)}"
            "</body></html>"
        )
