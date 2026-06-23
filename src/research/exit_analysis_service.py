from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.utils.db_manager import DatabaseManager
from src.utils.regime import benchmark_etf_for_sector
from src.utils.strategy import load_active_strategies, resolve_trade_strategy


@dataclass(frozen=True)
class ExitAnalysisReport:
    output_path: str
    closed_trade_count: int
    linked_trade_count: int
    analyzed_trade_count: int


class ExitAnalysisService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(self, *, horizons: tuple[int, ...] = (5, 10, 15, 20)) -> ExitAnalysisReport:
        self.db_manager.initialize()
        closed_trades = list(self.db_manager.list_closed_trades())
        output_path = self.db_manager.paths.reports_dir / "exit_analysis.md"
        if not closed_trades:
            output_path.write_text("# Exit Analysis\n\nNo closed trades found.\n", encoding="utf-8")
            return ExitAnalysisReport(output_path=str(output_path), closed_trade_count=0, linked_trade_count=0, analyzed_trade_count=0)

        linked_closed_trades = self._filter_to_recommended_trades(closed_trades)
        if not linked_closed_trades:
            output_path.write_text(
                "# Exit Analysis\n\nClosed trades were found, but none could be linked back to a selected SwingQuant scan recommendation on the entry date.\n",
                encoding="utf-8",
            )
            return ExitAnalysisReport(
                output_path=str(output_path),
                closed_trade_count=len(closed_trades),
                linked_trade_count=0,
                analyzed_trade_count=0,
            )

        universe_rows = self.db_manager.list_universe_rows(active_only=False)
        sector_map = {row["ticker"]: row["sector"] for row in universe_rows}
        strategies = load_active_strategies()
        benchmark_tickers = {
            benchmark
            for benchmark in (
                benchmark_etf_for_sector(str(sector_map.get(str(trade["ticker"]), "")))
                for trade in linked_closed_trades
            )
            if benchmark
        }
        tickers = sorted({str(trade["ticker"]) for trade in linked_closed_trades}.union({"SPY"}).union(benchmark_tickers))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            output_path.write_text("# Exit Analysis\n\nHistorical price data unavailable for closed trades.\n", encoding="utf-8")
            return ExitAnalysisReport(
                output_path=str(output_path),
                closed_trade_count=len(closed_trades),
                linked_trade_count=len(linked_closed_trades),
                analyzed_trade_count=0,
            )

        history_by_ticker = {
            ticker: frame.sort_values("date").reset_index(drop=True)
            for ticker, frame in price_history.groupby("ticker", sort=False)
        }
        backtest_lookup = getattr(self.db_manager, "get_backtest_result_by_strategy_id", None)

        trade_rows: list[dict[str, object]] = []
        for trade in linked_closed_trades:
            row = self._analyze_trade(
                trade=trade,
                sector_map=sector_map,
                strategies=strategies,
                history_by_ticker=history_by_ticker,
                backtest_lookup=backtest_lookup,
                horizons=horizons,
            )
            if row is not None:
                trade_rows.append(row)

        trade_frame = pd.DataFrame(trade_rows)
        if trade_frame.empty:
            output_path.write_text("# Exit Analysis\n\nLinked recommendation trades were found, but none had enough history to analyze.\n", encoding="utf-8")
            return ExitAnalysisReport(
                output_path=str(output_path),
                closed_trade_count=len(closed_trades),
                linked_trade_count=len(linked_closed_trades),
                analyzed_trade_count=0,
            )

        lines = [
            "# Exit Analysis",
            "",
            f"- closed_trade_count: {len(closed_trades)}",
            f"- linked_trade_count: {len(linked_closed_trades)}",
            f"- excluded_non_recommendation_trades: {len(closed_trades) - len(linked_closed_trades)}",
            f"- analyzed_trade_count: {len(trade_frame.index)}",
            f"- horizons: {', '.join(str(h) for h in horizons)}",
            "- note: this report includes only trades that can be linked to a selected SwingQuant scan recommendation on the entry date.",
            "- note: exact exit reason is not stored in the ledger yet; this report compares realized exits to simple counterfactual holds.",
            "",
        ]
        lines.extend(self._render_actual_summary(trade_frame))
        lines.extend(self._render_horizon_comparison(trade_frame, horizons))
        lines.extend(self._render_slot_breakdown(trade_frame, horizons))
        lines.extend(self._render_biggest_givebacks(trade_frame))
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return ExitAnalysisReport(
            output_path=str(output_path),
            closed_trade_count=len(closed_trades),
            linked_trade_count=len(linked_closed_trades),
            analyzed_trade_count=len(trade_frame.index),
        )

    def _filter_to_recommended_trades(self, closed_trades: list) -> list:
        loader = getattr(self.db_manager, "load_scan_candidates", None)
        if not callable(loader):
            return closed_trades
        scan_candidates = loader()
        if scan_candidates is None or scan_candidates.empty:
            return []
        selected = scan_candidates[scan_candidates["selected"].astype(int) == 1].copy()
        if selected.empty:
            return []
        selected["scan_date"] = selected["scan_date"].astype(str)
        selected["ticker"] = selected["ticker"].astype(str)
        selected["strategy_slot"] = selected["strategy_slot"].astype(str)
        linked_keys = {
            (str(row.scan_date), str(row.ticker), str(row.strategy_slot))
            for row in selected.itertuples(index=False)
        }
        linked_ticker_dates = {
            (str(row.scan_date), str(row.ticker))
            for row in selected.itertuples(index=False)
        }
        filtered = []
        for trade in closed_trades:
            entry_date = pd.Timestamp(trade["entry_date"]).normalize()
            candidate_dates = {
                str(entry_date.date()),
                str(pd.bdate_range(end=entry_date, periods=2)[0].date()),
            }
            ticker = str(trade["ticker"])
            strategy_slot = str(trade["strategy_slot"] or "")
            if any((candidate_date, ticker, strategy_slot) in linked_keys for candidate_date in candidate_dates) or any(
                (candidate_date, ticker) in linked_ticker_dates for candidate_date in candidate_dates
            ):
                filtered.append(trade)
        return filtered

    def _analyze_trade(
        self,
        *,
        trade,
        sector_map: dict[str, str],
        strategies: dict[str, object],
        history_by_ticker: dict[str, pd.DataFrame],
        backtest_lookup,
        horizons: tuple[int, ...],
    ) -> dict[str, object] | None:
        ticker = str(trade["ticker"])
        ticker_history = history_by_ticker.get(ticker)
        if ticker_history is None or ticker_history.empty:
            return None
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        exit_date = pd.Timestamp(trade["exit_date"]).normalize() if trade["exit_date"] else None
        if exit_date is None:
            return None
        window = ticker_history[
            (pd.to_datetime(ticker_history["date"]).dt.normalize() >= entry_date)
            & (pd.to_datetime(ticker_history["date"]).dt.normalize() <= exit_date)
        ].copy()
        if window.empty:
            return None
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        resolution = resolve_trade_strategy(
            trade=trade,
            strategies=strategies,
            sector_map=sector_map,
            backtest_lookup=backtest_lookup if callable(backtest_lookup) else None,
        )
        strategy = resolution.strategy
        slot = strategy.slot if strategy is not None else (trade["strategy_slot"] or "unknown")
        sector = sector_map.get(ticker) or (strategy.sector if strategy is not None else "Unknown")
        benchmark = benchmark_etf_for_sector(str(sector))

        actual_return = (exit_price / entry_price) - 1.0
        mfe_actual = float((window["high"].astype(float) / entry_price - 1.0).max())
        mae_actual = float((window["low"].astype(float) / entry_price - 1.0).min())
        actual_giveback = mfe_actual - actual_return
        holding_days = len(pd.bdate_range(entry_date, exit_date)) - 1
        row: dict[str, object] = {
            "ticker": ticker,
            "slot": str(slot),
            "sector": str(sector),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "holding_days": int(max(holding_days, 0)),
            "actual_return": float(actual_return),
            "mfe_actual": float(mfe_actual),
            "mae_actual": float(mae_actual),
            "actual_giveback": float(actual_giveback),
        }

        benchmark_history = history_by_ticker.get("SPY")
        row["actual_alpha_vs_spy"] = self._benchmark_alpha(
            benchmark_history,
            entry_date=entry_date,
            exit_date=exit_date,
            trade_return=actual_return,
        )
        sector_history = history_by_ticker.get(benchmark) if benchmark else None
        row["actual_alpha_vs_sector"] = self._benchmark_alpha(
            sector_history,
            entry_date=entry_date,
            exit_date=exit_date,
            trade_return=actual_return,
        )

        for horizon in horizons:
            target_date = pd.bdate_range(entry_date, periods=horizon + 1)[-1]
            horizon_row = self._row_on_or_after(ticker_history, target_date)
            if horizon_row is None:
                row[f"return_{horizon}d"] = np.nan
                row[f"alpha_vs_spy_{horizon}d"] = np.nan
                row[f"alpha_vs_sector_{horizon}d"] = np.nan
                row[f"delta_vs_actual_{horizon}d"] = np.nan
                continue
            horizon_return = float(horizon_row["adj_close"]) / entry_price - 1.0
            row[f"return_{horizon}d"] = horizon_return
            row[f"delta_vs_actual_{horizon}d"] = horizon_return - actual_return
            row[f"alpha_vs_spy_{horizon}d"] = self._benchmark_alpha(
                benchmark_history,
                entry_date=entry_date,
                exit_date=pd.Timestamp(horizon_row["date"]).normalize(),
                trade_return=horizon_return,
            )
            row[f"alpha_vs_sector_{horizon}d"] = self._benchmark_alpha(
                sector_history,
                entry_date=entry_date,
                exit_date=pd.Timestamp(horizon_row["date"]).normalize(),
                trade_return=horizon_return,
            )
        return row

    def _row_on_or_after(self, history: pd.DataFrame, target_date: pd.Timestamp):
        normalized_dates = pd.to_datetime(history["date"]).dt.normalize()
        eligible = history.loc[normalized_dates >= target_date]
        if eligible.empty:
            return None
        return eligible.sort_values("date").iloc[0]

    def _benchmark_alpha(
        self,
        benchmark_history: pd.DataFrame | None,
        *,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        trade_return: float,
    ) -> float | None:
        if benchmark_history is None or benchmark_history.empty:
            return None
        entry_row = self._row_on_or_after(benchmark_history, entry_date)
        exit_row = self._row_on_or_after(benchmark_history, exit_date)
        if entry_row is None or exit_row is None:
            return None
        entry_price = float(entry_row["adj_close"])
        exit_price = float(exit_row["adj_close"])
        if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(exit_price):
            return None
        benchmark_return = exit_price / entry_price - 1.0
        return float(trade_return - benchmark_return)

    def _render_actual_summary(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Actual Exit Summary", ""]
        lines.append(f"- mean_actual_return: {float(frame['actual_return'].mean()):.6f}")
        lines.append(f"- median_actual_return: {float(frame['actual_return'].median()):.6f}")
        lines.append(f"- hit_rate: {float((frame['actual_return'] > 0).mean()):.4f}")
        lines.append(f"- mean_holding_days: {float(frame['holding_days'].mean()):.2f}")
        lines.append(f"- mean_mfe_actual: {float(frame['mfe_actual'].mean()):.6f}")
        lines.append(f"- mean_mae_actual: {float(frame['mae_actual'].mean()):.6f}")
        lines.append(f"- mean_actual_giveback: {float(frame['actual_giveback'].mean()):.6f}")
        alpha_vs_spy = pd.to_numeric(frame["actual_alpha_vs_spy"], errors="coerce").dropna()
        if not alpha_vs_spy.empty:
            lines.append(f"- mean_actual_alpha_vs_spy: {float(alpha_vs_spy.mean()):.6f}")
        alpha_vs_sector = pd.to_numeric(frame["actual_alpha_vs_sector"], errors="coerce").dropna()
        if not alpha_vs_sector.empty:
            lines.append(f"- mean_actual_alpha_vs_sector: {float(alpha_vs_sector.mean()):.6f}")
        lines.append("")
        return lines

    def _render_horizon_comparison(self, frame: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
        lines = ["## Horizon Comparison", ""]
        for horizon in horizons:
            column = f"return_{horizon}d"
            delta_column = f"delta_vs_actual_{horizon}d"
            comparison = frame.dropna(subset=[column]).copy()
            lines.append(f"### {horizon}d")
            if comparison.empty:
                lines.append("- comparable_trades: 0")
                lines.append("")
                continue
            lines.append(f"- comparable_trades: {len(comparison.index)}")
            lines.append(f"- mean_counterfactual_return: {float(comparison[column].mean()):.6f}")
            lines.append(f"- mean_delta_vs_actual: {float(comparison[delta_column].mean()):.6f}")
            lines.append(f"- better_than_actual_rate: {float((comparison[delta_column] > 0).mean()):.4f}")
            alpha_sector_column = f"alpha_vs_sector_{horizon}d"
            alpha_sector = pd.to_numeric(comparison[alpha_sector_column], errors="coerce").dropna()
            if not alpha_sector.empty:
                lines.append(f"- mean_counterfactual_alpha_vs_sector: {float(alpha_sector.mean()):.6f}")
            lines.append("")
        return lines

    def _render_slot_breakdown(self, frame: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
        lines = ["## Slot Breakdown", ""]
        if frame.empty:
            lines.append("No analyzed trades.")
            lines.append("")
            return lines
        grouped = frame.groupby("slot", dropna=False)
        for slot, group in grouped:
            lines.append(f"### {slot}")
            lines.append(f"- trade_count: {len(group.index)}")
            lines.append(f"- mean_actual_return: {float(group['actual_return'].mean()):.6f}")
            lines.append(f"- mean_actual_giveback: {float(group['actual_giveback'].mean()):.6f}")
            for horizon in horizons:
                delta_column = f"delta_vs_actual_{horizon}d"
                comparable = pd.to_numeric(group[delta_column], errors="coerce").dropna()
                if comparable.empty:
                    continue
                lines.append(f"- mean_delta_vs_actual_{horizon}d: {float(comparable.mean()):.6f}")
            lines.append("")
        return lines

    def _render_biggest_givebacks(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Biggest Givebacks", ""]
        ranked = frame.sort_values(["actual_giveback", "ticker"], ascending=[False, True]).head(10)
        if ranked.empty:
            lines.append("No trades available.")
            lines.append("")
            return lines
        for row in ranked.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- slot: {row.slot}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- entry_date: {row.entry_date}")
            lines.append(f"- exit_date: {row.exit_date}")
            lines.append(f"- actual_return: {float(row.actual_return):.6f}")
            lines.append(f"- mfe_actual: {float(row.mfe_actual):.6f}")
            lines.append(f"- actual_giveback: {float(row.actual_giveback):.6f}")
            lines.append("")
        return lines
