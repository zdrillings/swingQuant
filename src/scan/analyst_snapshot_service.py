from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from src.scan.analyst_data import AnalystContext, AnalystDataClient
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


@dataclass(frozen=True)
class AnalystSnapshotReport:
    snapshot_date: str
    provider: str
    source: str
    requested_tickers: int
    persisted_rows: int
    rows_with_targets: int
    rows_with_recommendations: int
    output_path: Path


class AnalystSnapshotService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        analyst_data_client: AnalystDataClient | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.analyst_data_client = analyst_data_client or AnalystDataClient()
        self.logger = get_logger("analyst_snapshot")

    def run(
        self,
        *,
        snapshot_date: str | None = None,
        source: str = "research",
        top: int = 250,
        tickers: list[str] | None = None,
        provider: str = "yfinance",
    ) -> AnalystSnapshotReport:
        self.db_manager.initialize()
        snapshot_date = snapshot_date or date.today().isoformat()
        requested_tickers = self._resolve_tickers(source=source, top=top, tickers=tickers or [])
        self.logger.info(
            "Capturing analyst snapshots: date=%s provider=%s source=%s tickers=%s",
            snapshot_date,
            provider,
            source,
            len(requested_tickers),
        )
        contexts = self.analyst_data_client.load_contexts(requested_tickers)
        captured_at = datetime.now(timezone.utc).isoformat()
        rows = [
            self._row_from_context(
                ticker=ticker,
                context=contexts.get(ticker),
                captured_at=captured_at,
                source=source,
            )
            for ticker in requested_tickers
        ]
        persisted_rows = self.db_manager.replace_analyst_snapshots(
            snapshot_date=snapshot_date,
            provider=provider,
            rows=rows,
        )
        rows_with_targets = sum(
            1
            for row in rows
            if row.get("target_mean") is not None
            or row.get("target_median") is not None
            or row.get("target_low") is not None
            or row.get("target_high") is not None
        )
        rows_with_recommendations = sum(1 for row in rows if row.get("recommendation"))
        output_path = self._write_report(
            snapshot_date=snapshot_date,
            provider=provider,
            source=source,
            requested_tickers=len(requested_tickers),
            persisted_rows=persisted_rows,
            rows_with_targets=rows_with_targets,
            rows_with_recommendations=rows_with_recommendations,
            rows=rows,
        )
        return AnalystSnapshotReport(
            snapshot_date=snapshot_date,
            provider=provider,
            source=source,
            requested_tickers=len(requested_tickers),
            persisted_rows=persisted_rows,
            rows_with_targets=rows_with_targets,
            rows_with_recommendations=rows_with_recommendations,
            output_path=output_path,
        )

    def _resolve_tickers(self, *, source: str, top: int, tickers: list[str]) -> list[str]:
        if source == "research":
            rows = self.db_manager.list_research_universe(limit=max(0, int(top)))
            base = [str(row["ticker"]) for row in rows]
        elif source == "active":
            base = self.db_manager.list_universe_tickers(active_only=True)
            if top > 0:
                base = base[:top]
        else:
            raise ValueError(f"Unsupported analyst snapshot source: {source}")
        ordered: list[str] = []
        seen: set[str] = set()
        for raw_ticker in [*base, *tickers]:
            ticker = str(raw_ticker).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(ticker)
        return ordered

    def _row_from_context(
        self,
        *,
        ticker: str,
        context: AnalystContext | None,
        captured_at: str,
        source: str,
    ) -> dict:
        if context is None:
            return {
                "ticker": ticker,
                "captured_at": captured_at,
                "target_mean": None,
                "target_median": None,
                "target_low": None,
                "target_high": None,
                "analyst_count": None,
                "recommendation": None,
                "details": {"source": source, "has_context": False},
            }
        return {
            "ticker": ticker,
            "captured_at": captured_at,
            "target_mean": context.target_mean,
            "target_median": context.target_median,
            "target_low": context.target_low,
            "target_high": context.target_high,
            "analyst_count": context.analyst_count,
            "recommendation": context.recommendation,
            "details": {"source": source, "has_context": True},
        }

    def _write_report(
        self,
        *,
        snapshot_date: str,
        provider: str,
        source: str,
        requested_tickers: int,
        persisted_rows: int,
        rows_with_targets: int,
        rows_with_recommendations: int,
        rows: list[dict],
    ) -> Path:
        output_path = self.db_manager.paths.reports_dir / "analyst_snapshots.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Analyst Snapshot Capture",
            "",
            f"- snapshot_date: {snapshot_date}",
            f"- provider: {provider}",
            f"- source: {source}",
            f"- requested_tickers: {requested_tickers}",
            f"- persisted_rows: {persisted_rows}",
            f"- rows_with_targets: {rows_with_targets}",
            f"- rows_with_recommendations: {rows_with_recommendations}",
            "",
            "## Sample Rows",
            "",
            "| ticker | mean | median | low | high | analysts | recommendation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows[:25]:
            lines.append(
                "| {ticker} | {mean} | {median} | {low} | {high} | {analysts} | {recommendation} |".format(
                    ticker=row["ticker"],
                    mean=self._format_number(row.get("target_mean")),
                    median=self._format_number(row.get("target_median")),
                    low=self._format_number(row.get("target_low")),
                    high=self._format_number(row.get("target_high")),
                    analysts=row.get("analyst_count") if row.get("analyst_count") is not None else "",
                    recommendation=str(row.get("recommendation") or "").replace("|", "/"),
                )
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _format_number(value: object) -> str:
        if value is None:
            return ""
        return f"{float(value):.2f}"
