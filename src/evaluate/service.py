from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates, latest_snapshot


@dataclass(frozen=True)
class EvaluateReport:
    output_path: str
    rows_written: int


class EvaluateService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("evaluate")

    def run(self, *, top: int = 10, sector: str | None = None) -> EvaluateReport:
        self.db_manager.initialize()
        result_rows = self.db_manager.list_backtest_results()
        if not result_rows:
            raise ValueError("No backtest results found. Run `sq sweep` first.")

        records = []
        for row in result_rows:
            params = json.loads(row["params_json"])
            row_sector = params.get("sector", "ALL")
            if sector is not None and row_sector != sector:
                continue
            records.append(
                {
                    "id": row["id"],
                    "strategy_id": row["strategy_id"],
                    "params_json": row["params_json"],
                    "sector": row_sector,
                    "profit_factor": float(row["profit_factor"]),
                    "expectancy": float(row["expectancy"]),
                    "mdd": float(row["mdd"]),
                    "win_rate": float(row["win_rate"]),
                }
            )

        if not records:
            raise ValueError("No backtest results match the requested filter.")

        frame = pd.DataFrame(records)
        frame["norm_expectancy"] = self._min_max(frame["expectancy"])
        frame["norm_profit_factor"] = self._min_max(frame["profit_factor"])
        frame["norm_max_drawdown"] = self._min_max(frame["mdd"])
        frame["norm_score"] = (
            frame["norm_expectancy"] * 0.4
            + frame["norm_profit_factor"] * 0.3
            - frame["norm_max_drawdown"] * 0.3
        )
        self.db_manager.update_backtest_norm_scores(
            [(int(row.id), float(row.norm_score)) for row in frame.itertuples(index=False)]
        )

        candidate_links = self._build_candidate_links(frame)
        ranked = frame.sort_values(["norm_score", "expectancy"], ascending=[False, False]).head(top).copy()
        report_path = self.db_manager.paths.reports_dir / "candidates.md"
        lines = ["# Ranked Candidates", ""]
        for row in ranked.itertuples(index=False):
            params = json.loads(row.params_json)
            links = candidate_links.get(int(row.id), [])
            lines.append(f"## Result {row.id}")
            lines.append(f"- strategy_id: {row.strategy_id}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- norm_score: {row.norm_score:.6f}")
            lines.append(f"- expectancy: {row.expectancy:.6f}")
            lines.append(f"- profit_factor: {row.profit_factor:.6f}")
            lines.append(f"- mdd: {row.mdd:.6f}")
            lines.append(f"- win_rate: {row.win_rate:.6f}")
            lines.append(f"- params: `{json.dumps(params, sort_keys=True)}`")
            if links:
                lines.append(f"- TradingView: {', '.join(links)}")
            else:
                lines.append("- TradingView: none")
            lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return EvaluateReport(output_path=str(report_path), rows_written=len(ranked.index))

    def _build_candidate_links(self, ranked_frame: pd.DataFrame) -> dict[int, list[str]]:
        universe_rows = self.db_manager.list_research_universe(limit=250)
        if not universe_rows:
            return {}
        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            return {}

        analysis_frame, _ = build_analysis_frame(history, universe_rows)
        snapshot = latest_snapshot(analysis_frame)
        snapshot = snapshot[snapshot["ticker"].isin(universe_tickers) & snapshot["regime_green"].fillna(False)].copy()

        links: dict[int, list[str]] = {}
        for row in ranked_frame.itertuples(index=False):
            params = json.loads(row.params_json)
            candidates = filter_signal_candidates(snapshot, params["indicators"])
            if row.sector != "ALL":
                candidates = candidates[candidates["sector"] == row.sector]
            candidates = candidates.sort_values(["md_volume_30d", "ticker"], ascending=[False, True]).head(5)
            links[int(row.id)] = [
                f"https://www.tradingview.com/chart/?symbol={candidate.ticker}"
                for candidate in candidates.itertuples(index=False)
            ]
        return links

    def _min_max(self, series: pd.Series) -> pd.Series:
        min_value = series.min()
        max_value = series.max()
        if min_value == max_value:
            return pd.Series([0.0] * len(series.index), index=series.index)
        return (series - min_value) / (max_value - min_value)
