from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.sync.market_data import MarketDataClient, chunked
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector


@dataclass(frozen=True)
class ExtendedHoursSnapshotReport:
    snapshot_date: str
    requested_tickers: int
    persisted_rows: int
    rows_with_extended_price: int
    output_path: Path


class ExtendedHoursSnapshotService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        market_data_client: MarketDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.market_data_client = market_data_client or MarketDataClient()
        self.logger = get_logger("extended_hours_snapshot")

    def run(
        self,
        *,
        snapshot_date: str | None = None,
        source: str = "research",
        top: int = 250,
        tickers: list[str] | None = None,
    ) -> ExtendedHoursSnapshotReport:
        self.db_manager.initialize()
        selected_date = snapshot_date or date.today().isoformat()
        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        sector_map = {str(row["ticker"]).upper(): row["sector"] for row in universe_rows}
        requested_tickers = self._resolve_tickers(source=source, top=top, tickers=tickers or [])
        benchmark_tickers = sorted(
            {
                benchmark_etf_for_sector(sector)
                for ticker, sector in sector_map.items()
                if ticker in requested_tickers and sector
            }
            | set(REFERENCE_TICKERS)
        )
        all_tickers = sorted(set(requested_tickers).union(benchmark_tickers))
        self.logger.info(
            "Capturing extended-hours snapshot: date=%s source=%s tickers=%s benchmarks=%s",
            selected_date,
            source,
            len(requested_tickers),
            len(benchmark_tickers),
        )
        intraday = self._download_intraday(all_tickers)
        captured_at = datetime.now(timezone.utc).isoformat()
        ticker_metrics = {
            ticker: self._metrics_for_ticker(intraday, ticker)
            for ticker in all_tickers
        }
        rows = []
        for ticker in requested_tickers:
            metrics = ticker_metrics.get(ticker, {})
            sector = sector_map.get(ticker)
            sector_etf = benchmark_etf_for_sector(sector) if sector else None
            sector_return = (
                ticker_metrics.get(sector_etf, {}).get("extended_return")
                if sector_etf
                else None
            )
            extended_return = metrics.get("extended_return")
            relative_return = (
                float(extended_return) - float(sector_return)
                if self._is_finite(extended_return) and self._is_finite(sector_return)
                else None
            )
            rows.append(
                {
                    "ticker": ticker,
                    "captured_at": captured_at,
                    "source": source,
                    "sector": sector,
                    "sector_etf": sector_etf,
                    "regular_close": metrics.get("regular_close"),
                    "extended_price": metrics.get("extended_price"),
                    "extended_return": extended_return,
                    "extended_volume": metrics.get("extended_volume"),
                    "last_trade_at": metrics.get("last_trade_at"),
                    "sector_etf_extended_return": sector_return,
                    "relative_extended_return": relative_return,
                    "details": {
                        "has_extended_rows": bool(metrics.get("has_extended_rows", False)),
                        "benchmark_tickers": benchmark_tickers,
                    },
                }
            )
        persisted_rows = self.db_manager.replace_extended_hours_snapshots(
            snapshot_date=selected_date,
            rows=rows,
        )
        rows_with_extended_price = sum(1 for row in rows if self._is_finite(row.get("extended_price")))
        output_path = self._write_report(
            snapshot_date=selected_date,
            rows=rows,
            requested_tickers=len(requested_tickers),
            persisted_rows=persisted_rows,
            rows_with_extended_price=rows_with_extended_price,
        )
        return ExtendedHoursSnapshotReport(
            snapshot_date=selected_date,
            requested_tickers=len(requested_tickers),
            persisted_rows=persisted_rows,
            rows_with_extended_price=rows_with_extended_price,
            output_path=output_path,
        )

    def _resolve_tickers(self, *, source: str, top: int, tickers: list[str]) -> list[str]:
        resolved = {str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()}
        if source in {"research", "all"}:
            resolved.update(str(row["ticker"]).upper() for row in self.db_manager.list_research_universe(limit=top))
        if source in {"open", "all"} and hasattr(self.db_manager, "list_open_trades"):
            resolved.update(str(row["ticker"]).upper() for row in self.db_manager.list_open_trades())
        if source not in {"research", "open", "all", "explicit"}:
            raise ValueError("source must be one of: research, open, all, explicit")
        return sorted(resolved)

    def _download_intraday(self, tickers: list[str]) -> pd.DataFrame:
        frames = []
        for ticker_batch in chunked(tickers, 75):
            try:
                frames.append(self.market_data_client.download_intraday_history(ticker_batch, include_prepost=True))
            except Exception as exc:
                self.logger.warning("Unable to download extended-hours intraday batch: %s", exc)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]

    def _metrics_for_ticker(self, raw_frame: pd.DataFrame, ticker: str) -> dict[str, object]:
        frame = self._intraday_frame_for_ticker(raw_frame, ticker)
        if frame.empty:
            return {}
        eastern = ZoneInfo("America/New_York")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if getattr(frame["timestamp"].dt, "tz", None) is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(eastern)
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert(eastern)
        frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            return {}
        regular = frame[frame["timestamp"].dt.time <= time(16, 0)].copy()
        extended = frame[frame["timestamp"].dt.time > time(16, 0)].copy()
        if regular.empty:
            return {}
        regular_close = float(regular.tail(1).iloc[0]["close"])
        if extended.empty:
            return {
                "regular_close": regular_close,
                "has_extended_rows": False,
            }
        latest = extended.tail(1).iloc[0]
        extended_price = float(latest["close"])
        extended_volume = pd.to_numeric(extended["volume"], errors="coerce").fillna(0).sum()
        return {
            "regular_close": regular_close,
            "extended_price": extended_price,
            "extended_return": (extended_price / regular_close) - 1.0 if regular_close else None,
            "extended_volume": int(extended_volume),
            "last_trade_at": latest["timestamp"].isoformat(),
            "has_extended_rows": True,
        }

    def _intraday_frame_for_ticker(self, raw_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if raw_frame.empty:
            return pd.DataFrame()
        frame = raw_frame.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            available = set(frame.columns.get_level_values(0))
            if ticker not in available:
                return pd.DataFrame()
            frame = frame[ticker].copy()
        if frame.empty:
            return pd.DataFrame()
        frame = frame.reset_index()
        rename_map = {
            "index": "timestamp",
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Close": "close",
            "Volume": "volume",
        }
        frame = frame.rename(columns=rename_map)
        if "timestamp" not in frame.columns or "close" not in frame.columns:
            return pd.DataFrame()
        if "volume" not in frame.columns:
            frame["volume"] = 0
        return frame[["timestamp", "close", "volume"]].copy()

    def _write_report(
        self,
        *,
        snapshot_date: str,
        rows: list[dict],
        requested_tickers: int,
        persisted_rows: int,
        rows_with_extended_price: int,
    ) -> Path:
        output_path = self.db_manager.paths.reports_dir / "extended_hours_snapshots.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(
            [row for row in rows if self._is_finite(row.get("relative_extended_return"))],
            key=lambda row: float(row["relative_extended_return"]),
            reverse=True,
        )[:20]
        lines = [
            "# Extended Hours Snapshot",
            "",
            f"- snapshot_date: {snapshot_date}",
            f"- requested_tickers: {requested_tickers}",
            f"- persisted_rows: {persisted_rows}",
            f"- rows_with_extended_price: {rows_with_extended_price}",
            "",
            "## Top Relative Extended-Hours Moves",
            "",
        ]
        if not ranked:
            lines.append("No extended-hours relative moves were available.")
        else:
            for row in ranked:
                lines.append(
                    f"- {row['ticker']}: extended={self._format_pct(row.get('extended_return'))}, "
                    f"sector={self._format_pct(row.get('sector_etf_extended_return'))}, "
                    f"relative={self._format_pct(row.get('relative_extended_return'))}, "
                    f"volume={row.get('extended_volume') or 0}"
                )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _is_finite(value) -> bool:
        try:
            return pd.notna(value) and float(value) == float(value)
        except (TypeError, ValueError):
            return False

    def _format_pct(self, value) -> str:
        if not self._is_finite(value):
            return "n/a"
        return f"{float(value) * 100.0:+.2f}%"
