from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.settings import get_settings
from src.sync.market_data import MarketDataClient, chunked, extract_ticker_history
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.emailer import send_html_email
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot, overlay_price_history
from src.utils.sizing import compute_position_size
from src.utils.strategy import load_active_strategy


@dataclass(frozen=True)
class ScanReport:
    candidate_count: int
    emailed: bool


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
        strategy = load_active_strategy()
        settings = get_settings()
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
        candidates = filter_signal_candidates(snapshot, strategy.indicators)
        candidates = candidates.sort_values(["md_volume_30d", "ticker"], ascending=[False, True]).head(5).copy()
        if candidates.empty:
            return ScanReport(candidate_count=0, emailed=False)

        candidates["shares"] = candidates["adj_close"].apply(
            lambda price: compute_position_size(
                price=float(price),
                trailing_stop_pct=strategy.exit_rules.trailing_stop_pct,
                settings=settings,
            )
        )
        html = self._build_email_html(candidates)
        self.email_sender(
            subject="Evening Brief",
            html_body=html,
            settings=settings,
        )
        return ScanReport(candidate_count=len(candidates.index), emailed=True)

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

    def _build_email_html(self, candidates: pd.DataFrame) -> str:
        rows = []
        for candidate in candidates.itertuples(index=False):
            rows.append(
                "<tr>"
                f"<td>{candidate.ticker}</td>"
                f"<td>{candidate.sector}</td>"
                f"<td>{candidate.regime_etf} Green</td>"
                f"<td>{candidate.adj_close:.2f}</td>"
                f"<td>{candidate.shares}</td>"
                "</tr>"
            )
        return (
            "<html><body>"
            "<h1>Evening Brief</h1>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Ticker</th><th>Sector</th><th>Regime</th><th>Adj Close</th><th>Shares</th></tr>"
            f"{''.join(rows)}"
            "</table>"
            "</body></html>"
        )
