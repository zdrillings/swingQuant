from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import pandas as pd

from src.settings import load_feature_config
from src.sweep.service import BenchmarkContext, SweepService
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame


@dataclass(frozen=True)
class SleeveResearchReport:
    output_path: str
    configs_evaluated: int


class SleeveResearchService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("sleeves")

    def run(
        self,
        *,
        top: int = 10,
        sectors: list[str] | None = None,
        walk_forward: bool = False,
        walk_forward_windows: int = 5,
        walk_forward_shortlist: int = 10,
    ) -> SleeveResearchReport:
        self.db_manager.initialize()
        research_rows = self.db_manager.list_research_universe(limit=250)
        if not research_rows:
            raise ValueError("Universe is empty or liquidity metrics are unavailable. Run `sq sync` first.")

        config = load_feature_config()
        sleeve_config = self._load_sleeve_config(config)
        target_sectors = sectors or list(sleeve_config["sectors"])
        universe_tickers = [row["ticker"] for row in research_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")

        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(price_history, research_rows, earnings_calendar=earnings_calendar)
        analysis_frame = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        analysis_frame = analysis_frame[analysis_frame["sector"].isin(target_sectors)].copy()
        if analysis_frame.empty:
            raise ValueError("No sleeve analysis frame could be built for the requested sectors.")

        sweep_service = SweepService(self.db_manager)
        benchmark_price_maps = sweep_service._build_benchmark_price_maps(price_history)
        backtest_costs = sweep_service._load_backtest_costs(config)
        latest_date = pd.to_datetime(analysis_frame["date"]).max()

        results: list[dict[str, object]] = []
        configs_evaluated = 0
        for sector in target_sectors:
            sector_frame = analysis_frame[analysis_frame["sector"] == sector].copy()
            if sector_frame.empty:
                continue
            for candidate in self._iter_sector_configurations(sleeve_config):
                metrics = self._evaluate_sector_configuration(
                    sector_frame=sector_frame,
                    sector=sector,
                    top_n=int(candidate["top_n"]),
                    horizon_days=int(candidate["horizon_days"]),
                    breadth_filters={
                        "sector_pct_above_50_min": float(candidate["sector_pct_above_50_min"]),
                        "sector_pct_above_200_min": float(candidate["sector_pct_above_200_min"]),
                        "sector_median_roc_63_min": float(candidate["sector_median_roc_63_min"]),
                    },
                    base_filters=sleeve_config["base_filters"],
                    rank_weights=sleeve_config["rank_weights"],
                    backtest_costs=backtest_costs,
                    benchmark_context=BenchmarkContext(
                        spy_ticker="SPY",
                        sector_ticker=benchmark_etf_for_sector(sector),
                        price_maps=benchmark_price_maps,
                    ),
                    latest_date=latest_date,
                    sweep_service=sweep_service,
                )
                results.append(
                    {
                        "sector": sector,
                        "top_n": int(candidate["top_n"]),
                        "horizon_days": int(candidate["horizon_days"]),
                        "sector_pct_above_50_min": float(candidate["sector_pct_above_50_min"]),
                        "sector_pct_above_200_min": float(candidate["sector_pct_above_200_min"]),
                        "sector_median_roc_63_min": float(candidate["sector_median_roc_63_min"]),
                        "base_filters": sleeve_config["base_filters"],
                        "rank_weights": sleeve_config["rank_weights"],
                        **metrics,
                    }
                )
                configs_evaluated += 1

        if not results:
            raise ValueError("No sleeve research results were produced.")
        frame = pd.DataFrame(results)
        frame = self._apply_practical_scoring(frame, sleeve_config["support"])
        ranked = frame.sort_values(
            ["practical_score", "alpha_vs_spy", "expectancy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        ranked["global_rank"] = range(1, len(ranked.index) + 1)
        ranked["sector_rank"] = ranked.groupby("sector")["practical_score"].rank(method="dense", ascending=False).astype(int)
        supported_ranked = self._supported_for_top_sections(ranked, support=sleeve_config["support"])

        walk_forward_available = False
        if walk_forward:
            walk_forward_metrics = self._build_walk_forward_stability(
                ranked=ranked,
                analysis_frame=analysis_frame,
                windows=walk_forward_windows,
                shortlist_size=walk_forward_shortlist,
                sleeve_config=sleeve_config,
                latest_date=latest_date,
                sweep_service=sweep_service,
                benchmark_price_maps=benchmark_price_maps,
                backtest_costs=backtest_costs,
            )
            if not walk_forward_metrics.empty:
                walk_forward_available = True
                ranked = ranked.merge(walk_forward_metrics, on=self._sleeve_key_columns(), how="left")

        report_path = self.db_manager.paths.reports_dir / "sleeve_research.md"
        lines = [
            "# Sleeve Research",
            "",
            f"- sectors: {', '.join(target_sectors)}",
            f"- configs_evaluated: {configs_evaluated}",
            "- model: sector breadth filter + within-sector rank score + fixed holding horizon",
            "- note: portfolio simulation is equal-weight within sleeve, max open positions = top_n, no cross-sector capital coupling",
            (
                f"- walk_forward: enabled on shortlist={walk_forward_shortlist} with rolling_windows={walk_forward_windows}"
                if walk_forward
                else "- walk_forward: disabled"
            ),
            "",
        ]
        lines.extend(
            self._render_section(
                "Top Ranked Sleeve Configurations",
                supported_ranked.head(top).copy(),
                empty_message="No sleeve configurations currently satisfy the top-section support floor.",
            )
        )
        lines.extend(
            self._render_section(
                "Best Configuration Per Sector",
                self._best_per_sector(supported_ranked),
                empty_message="No sector-level sleeve configurations currently satisfy the top-section support floor.",
            )
        )
        lines.extend(
            self._render_section(
                "Best Configuration Per Horizon",
                self._best_per_horizon(supported_ranked),
                empty_message="No horizon-level sleeve configurations currently satisfy the top-section support floor.",
            )
        )
        lines.extend(
            self._render_section(
                "Best Live Configurations With Enough Sample",
                self._best_live_supported(
                    ranked,
                    min_trade_count=int(sleeve_config["support"]["min_trade_count_for_live_section"]),
                    top=top,
                ),
                empty_message="No live sleeve configurations currently satisfy the minimum trade support floor.",
            )
        )
        lines.extend(
            self._render_section(
                "Best Supported Configuration Per Sector",
                self._best_supported_per_sector(ranked, support=sleeve_config["support"]),
                empty_message="No sector-level sleeve configurations currently satisfy the supported floor.",
            )
        )
        lines.extend(
            self._render_section(
                "Best Supported Live Configuration Per Sector",
                self._best_supported_live_per_sector(ranked, support=sleeve_config["support"]),
                empty_message="No sector-level live sleeve configurations currently satisfy the supported floor.",
            )
        )
        if walk_forward_available:
            lines.extend(
                self._render_section(
                    "Best Walk-Forward Stable Sleeve Configurations",
                    self._best_walk_forward_stable(ranked, top=top),
                    empty_message="No walk-forward sleeve metrics are available.",
                )
            )
            lines.extend(
                self._render_section(
                    "Best Stable Configuration Per Sector",
                    self._best_stable_per_sector(ranked, support=sleeve_config["support"]),
                    empty_message="No sector-level stable sleeve configurations currently satisfy the supported floor.",
                )
            )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return SleeveResearchReport(output_path=str(report_path), configs_evaluated=configs_evaluated)

    def _load_sleeve_config(self, config: dict) -> dict:
        default = {
            "sectors": ["Materials", "Energy", "Industrials"],
            "top_n_values": [1, 2],
            "horizon_days_values": [5, 10, 15],
            "support": {
                "min_trade_count_for_rank": 75,
                "target_trade_count_for_score": 100,
                "min_trade_count_for_live_section": 75,
                "min_trade_count_for_top_sections": 75,
                "min_distinct_tickers_for_rank": 3,
                "target_distinct_tickers_for_score": 5,
                "min_distinct_tickers_for_top_sections": 3,
                "target_live_match_count_for_score": 2,
                "min_live_match_count_for_top_sections": 1,
            },
            "breadth_thresholds": {
                "sector_pct_above_50_min": [0.40, 0.50, 0.60],
                "sector_pct_above_200_min": [0.40, 0.50, 0.60],
                "sector_median_roc_63_min": [0.00, 0.05],
            },
            "base_filters": {
                "relative_strength_index_vs_spy_min": 75.0,
                "roc_63_min": 0.05,
                "vol_alpha_min": 1.0,
                "rsi_14_min": 40.0,
                "rsi_14_max": 65.0,
            },
            "rank_weights": {
                "relative_strength_index_vs_spy": 0.40,
                "roc_63": 0.25,
                "rsi_14_pullback": 0.20,
                "vol_alpha": 0.15,
            },
        }
        loaded = config.get("sleeve_research", {})
        if not loaded:
            return default
        merged = {**default, **loaded}
        for nested_key in ("support", "breadth_thresholds", "base_filters", "rank_weights"):
            merged[nested_key] = {**default[nested_key], **loaded.get(nested_key, {})}
        return merged

    def _iter_sector_configurations(self, config: dict):
        breadth = config["breadth_thresholds"]
        for top_n, horizon_days, above_50, above_200, median_roc in product(
            config["top_n_values"],
            config["horizon_days_values"],
            breadth["sector_pct_above_50_min"],
            breadth["sector_pct_above_200_min"],
            breadth["sector_median_roc_63_min"],
        ):
            yield {
                "top_n": int(top_n),
                "horizon_days": int(horizon_days),
                "sector_pct_above_50_min": float(above_50),
                "sector_pct_above_200_min": float(above_200),
                "sector_median_roc_63_min": float(median_roc),
            }

    def _evaluate_sector_configuration(
        self,
        *,
        sector_frame: pd.DataFrame,
        sector: str,
        top_n: int,
        horizon_days: int,
        breadth_filters: dict[str, float],
        base_filters: dict[str, float],
        rank_weights: dict[str, float],
        backtest_costs,
        benchmark_context: BenchmarkContext,
        latest_date: pd.Timestamp,
        sweep_service: SweepService,
    ) -> dict[str, object]:
        working = sector_frame.sort_values(["ticker", "date"]).copy()
        working["date"] = pd.to_datetime(working["date"])
        ticker_frames = {
            ticker: group.sort_values("date").reset_index(drop=True)
            for ticker, group in working.groupby("ticker", sort=False)
        }
        ticker_date_index = {
            ticker: {pd.Timestamp(row["date"]): index for index, row in group.iterrows()}
            for ticker, group in ticker_frames.items()
        }
        sector_dates = sorted(working["date"].drop_duplicates().tolist())

        open_positions: list[dict[str, object]] = []
        open_tickers: set[str] = set()
        trades: list[float] = []
        spy_excess_trades: list[float] = []
        sector_excess_trades: list[float] = []
        traded_tickers: set[str] = set()
        trade_counts_by_ticker: dict[str, int] = {}
        daily_return_contributions: dict[pd.Timestamp, list[float]] = {}
        live_candidates = pd.DataFrame()

        for current_date in sector_dates:
            remaining_positions: list[dict[str, object]] = []
            for position in open_positions:
                if position["exit_date"] == current_date:
                    exit_price = float(position["exit_price"])
                    pnl_pct = ((exit_price - float(position["entry_price"])) / float(position["entry_price"])) - backtest_costs.round_trip_cost_fraction
                    trades.append(pnl_pct)
                    spy_benchmark_return = sweep_service._benchmark_return(
                        benchmark_context.price_maps,
                        benchmark_context.spy_ticker,
                        entry_date=position["entry_date"],
                        exit_date=current_date,
                    )
                    if spy_benchmark_return is not None:
                        spy_excess_trades.append(pnl_pct - spy_benchmark_return)
                    if benchmark_context.sector_ticker is not None:
                        sector_benchmark_return = sweep_service._benchmark_return(
                            benchmark_context.price_maps,
                            benchmark_context.sector_ticker,
                            entry_date=position["entry_date"],
                            exit_date=current_date,
                        )
                        if sector_benchmark_return is not None:
                            sector_excess_trades.append(pnl_pct - sector_benchmark_return)
                    traded_tickers.add(str(position["ticker"]))
                    trade_counts_by_ticker[str(position["ticker"])] = trade_counts_by_ticker.get(str(position["ticker"]), 0) + 1
                    open_tickers.discard(str(position["ticker"]))
                else:
                    remaining_positions.append(position)
            open_positions = remaining_positions

            day_frame = working[working["date"] == current_date].copy()
            ranked_candidates = self._rank_candidates_for_day(
                day_frame,
                breadth_filters=breadth_filters,
                base_filters=base_filters,
                rank_weights=rank_weights,
            )
            if current_date == latest_date:
                live_candidates = ranked_candidates.head(top_n).copy()
            available_slots = max(top_n - len(open_positions), 0)
            if available_slots <= 0 or ranked_candidates.empty:
                continue
            for candidate in ranked_candidates.itertuples(index=False):
                if available_slots <= 0:
                    break
                ticker = str(candidate.ticker)
                if ticker in open_tickers:
                    continue
                ticker_frame = ticker_frames[ticker]
                signal_index = ticker_date_index[ticker].get(current_date)
                if signal_index is None or signal_index + 1 >= len(ticker_frame.index):
                    continue
                entry_index = signal_index + 1
                exit_index = entry_index + horizon_days - 1
                if exit_index >= len(ticker_frame.index):
                    continue
                entry_row = ticker_frame.iloc[entry_index]
                exit_row = ticker_frame.iloc[exit_index]
                entry_price = float(entry_row["open"])
                if entry_price <= 0:
                    continue
                self._append_trade_daily_returns(
                    daily_return_contributions,
                    ticker_frame=ticker_frame,
                    entry_index=entry_index,
                    exit_index=exit_index,
                    entry_price=entry_price,
                )
                open_positions.append(
                    {
                        "ticker": ticker,
                        "entry_date": pd.Timestamp(entry_row["date"]),
                        "entry_price": entry_price,
                        "exit_date": pd.Timestamp(exit_row["date"]),
                        "exit_price": float(exit_row["close"]),
                    }
                )
                open_tickers.add(ticker)
                available_slots -= 1

        portfolio_mdd = self._portfolio_mdd_from_daily_returns(daily_return_contributions)
        distinct_tickers_traded = len(traded_tickers)
        max_ticker_trade_share = (
            max(trade_counts_by_ticker.values()) / len(trades)
            if trades and trade_counts_by_ticker
            else 0.0
        )
        if not trades:
            return {
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "alpha_vs_spy": np.nan,
                "alpha_vs_sector": np.nan,
                "mdd": portfolio_mdd,
                "win_rate": 0.0,
                "trade_count": 0,
                "distinct_tickers_traded": 0,
                "max_ticker_trade_share": 0.0,
                "live_match_count": int(len(live_candidates.index)),
                "live_match_tickers": live_candidates["ticker"].tolist() if not live_candidates.empty else [],
            }

        profit_sum = sum(value for value in trades if value > 0)
        loss_sum = abs(sum(value for value in trades if value < 0))
        if loss_sum > 0:
            profit_factor = profit_sum / loss_sum
        elif profit_sum > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        return {
            "expectancy": float(sum(trades) / len(trades)),
            "profit_factor": float(profit_factor),
            "alpha_vs_spy": self._average_or_nan(spy_excess_trades),
            "alpha_vs_sector": self._average_or_nan(sector_excess_trades),
            "mdd": float(portfolio_mdd),
            "win_rate": float(sum(1 for value in trades if value > 0) / len(trades)),
            "trade_count": len(trades),
            "distinct_tickers_traded": int(distinct_tickers_traded),
            "max_ticker_trade_share": float(max_ticker_trade_share),
            "live_match_count": int(len(live_candidates.index)),
            "live_match_tickers": live_candidates["ticker"].tolist() if not live_candidates.empty else [],
        }

    def _rank_candidates_for_day(
        self,
        frame: pd.DataFrame,
        *,
        breadth_filters: dict[str, float],
        base_filters: dict[str, float],
        rank_weights: dict[str, float],
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        breadth_ok = (
            float(frame["sector_pct_above_50"].iloc[0]) >= breadth_filters["sector_pct_above_50_min"]
            and float(frame["sector_pct_above_200"].iloc[0]) >= breadth_filters["sector_pct_above_200_min"]
            and float(frame["sector_median_roc_63"].iloc[0]) >= breadth_filters["sector_median_roc_63_min"]
        )
        if not breadth_ok:
            return frame.iloc[0:0].copy()

        working = frame.copy()
        working = working[working["regime_green"].fillna(False)].copy()
        working = working[working["sma_50_dist"].astype(float) > 0.0].copy()
        working = working[working["sma_200_dist"].astype(float) > 0.0].copy()
        working = working[working["relative_strength_index_vs_spy"].astype(float) >= float(base_filters["relative_strength_index_vs_spy_min"])].copy()
        working = working[working["roc_63"].astype(float) >= float(base_filters["roc_63_min"])].copy()
        working = working[working["vol_alpha"].astype(float) >= float(base_filters["vol_alpha_min"])].copy()
        if "rsi_14_min" in base_filters:
            working = working[working["rsi_14"].astype(float) >= float(base_filters["rsi_14_min"])].copy()
        if "rsi_14_max" in base_filters:
            working = working[working["rsi_14"].astype(float) <= float(base_filters["rsi_14_max"])].copy()
        if working.empty:
            return working

        working["rank_rs"] = working["relative_strength_index_vs_spy"].rank(method="average", pct=True)
        working["rank_roc"] = working["roc_63"].rank(method="average", pct=True)
        working["rank_pullback"] = working["rsi_14"].rank(method="average", pct=True, ascending=False)
        working["rank_volume"] = working["vol_alpha"].rank(method="average", pct=True)
        working["rank_score"] = (
            working["rank_rs"] * float(rank_weights["relative_strength_index_vs_spy"])
            + working["rank_roc"] * float(rank_weights["roc_63"])
            + working["rank_pullback"] * float(rank_weights["rsi_14_pullback"])
            + working["rank_volume"] * float(rank_weights["vol_alpha"])
        )
        return working.sort_values(
            ["rank_score", "relative_strength_index_vs_spy", "md_volume_30d", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def _append_trade_daily_returns(
        self,
        contributions: dict[pd.Timestamp, list[float]],
        *,
        ticker_frame: pd.DataFrame,
        entry_index: int,
        exit_index: int,
        entry_price: float,
    ) -> None:
        previous_price = entry_price
        for index in range(entry_index, exit_index + 1):
            row = ticker_frame.iloc[index]
            current_date = pd.Timestamp(row["date"])
            close_price = float(row["close"])
            daily_return = (close_price - previous_price) / previous_price
            contributions.setdefault(current_date, []).append(daily_return)
            previous_price = close_price

    def _portfolio_mdd_from_daily_returns(self, contributions: dict[pd.Timestamp, list[float]]) -> float:
        if not contributions:
            return 0.0
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for current_date in sorted(contributions):
            daily_values = contributions[current_date]
            daily_return = float(sum(daily_values) / len(daily_values)) if daily_values else 0.0
            equity *= 1.0 + daily_return
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return float(max_drawdown)

    def _average_or_nan(self, values: list[float]) -> float:
        if not values:
            return np.nan
        return float(sum(values) / len(values))

    def _apply_practical_scoring(self, frame: pd.DataFrame, support: dict[str, float | int]) -> pd.DataFrame:
        scored = frame.copy()
        scored["norm_expectancy"] = self._min_max(scored["expectancy"])
        scored["norm_profit_factor"] = self._min_max(scored["profit_factor"])
        scored["norm_mdd"] = self._min_max(scored["mdd"])
        scored["norm_alpha_vs_spy"] = self._min_max(scored["alpha_vs_spy"])
        scored["norm_alpha_vs_sector"] = self._min_max(scored["alpha_vs_sector"])
        min_trade_count_for_rank = max(int(support.get("min_trade_count_for_rank", 0)), 1)
        target_trade_count_for_score = max(int(support.get("target_trade_count_for_score", min_trade_count_for_rank)), 1)
        min_distinct_tickers_for_rank = max(int(support.get("min_distinct_tickers_for_rank", 1)), 1)
        target_distinct_tickers_for_score = max(
            int(support.get("target_distinct_tickers_for_score", min_distinct_tickers_for_rank)),
            1,
        )
        target_live_match_count_for_score = max(int(support.get("target_live_match_count_for_score", 1)), 1)
        scored["trade_support_ratio"] = scored["trade_count"].map(
            lambda count: min(max(int(count), 0), target_trade_count_for_score) / target_trade_count_for_score
        )
        scored["low_support_penalty"] = scored["trade_count"].map(
            lambda count: max(min_trade_count_for_rank - max(int(count), 0), 0) / min_trade_count_for_rank
        )
        scored["distinct_ticker_support_ratio"] = scored["distinct_tickers_traded"].map(
            lambda count: min(max(int(count), 0), target_distinct_tickers_for_score) / target_distinct_tickers_for_score
        )
        scored["low_distinct_ticker_penalty"] = scored["distinct_tickers_traded"].map(
            lambda count: max(min_distinct_tickers_for_rank - max(int(count), 0), 0) / min_distinct_tickers_for_rank
        )
        scored["ticker_concentration_penalty"] = scored["max_ticker_trade_share"].map(
            lambda share: max(float(share) - 0.50, 0.0)
        )
        scored["live_match_ratio"] = scored["live_match_count"].map(
            lambda count: min(max(int(count), 0), target_live_match_count_for_score) / target_live_match_count_for_score
        )
        scored["live_match_bonus"] = scored["live_match_count"].map(
            lambda count: 0.0 if int(count) <= 0 else 0.20 + (min(int(count), 4) - 1) * 0.04
        )
        scored["practical_score"] = (
            scored["norm_expectancy"] * 0.26
            + scored["norm_profit_factor"] * 0.18
            - scored["norm_mdd"] * 0.25
            + scored["norm_alpha_vs_spy"] * 0.10
            + scored["norm_alpha_vs_sector"] * 0.10
            + scored["trade_support_ratio"] * 0.10
            + scored["distinct_ticker_support_ratio"] * 0.08
            + scored["live_match_ratio"] * 0.08
            + scored["live_match_bonus"]
            - scored["low_support_penalty"] * 0.25
            - scored["low_distinct_ticker_penalty"] * 0.18
            - scored["ticker_concentration_penalty"] * 0.15
        )
        return scored

    def _best_per_sector(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return frame.sort_values(
            ["practical_score", "alpha_vs_spy", "expectancy"],
            ascending=[False, False, False],
        ).drop_duplicates(subset=["sector"], keep="first").reset_index(drop=True)

    def _best_per_horizon(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return frame.sort_values(
            ["practical_score", "alpha_vs_spy", "expectancy"],
            ascending=[False, False, False],
        ).drop_duplicates(subset=["horizon_days"], keep="first").reset_index(drop=True)

    def _best_live_supported(self, frame: pd.DataFrame, *, min_trade_count: int, top: int) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return (
            frame[
                (frame["live_match_count"] > 0)
                & frame["trade_count"].notna()
                & (frame["trade_count"] >= max(int(min_trade_count), 0))
            ]
            .sort_values(
                ["practical_score", "alpha_vs_spy", "expectancy"],
                ascending=[False, False, False],
            )
            .head(top)
            .reset_index(drop=True)
        )

    def _supported_for_top_sections(self, frame: pd.DataFrame, *, support: dict[str, float | int]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        min_trade_count = max(int(support.get("min_trade_count_for_top_sections", 0)), 0)
        min_distinct_tickers = max(int(support.get("min_distinct_tickers_for_top_sections", 0)), 0)
        min_live_match_count = max(int(support.get("min_live_match_count_for_top_sections", 0)), 0)
        return frame[
            frame["trade_count"].notna()
            & (frame["trade_count"] >= min_trade_count)
            & frame["distinct_tickers_traded"].notna()
            & (frame["distinct_tickers_traded"] >= min_distinct_tickers)
            & frame["live_match_count"].notna()
            & (frame["live_match_count"] >= min_live_match_count)
        ].copy()

    def _best_supported_per_sector(self, frame: pd.DataFrame, *, support: dict[str, float | int]) -> pd.DataFrame:
        return self._best_per_sector(self._supported_for_top_sections(frame, support=support))

    def _best_supported_live_per_sector(self, frame: pd.DataFrame, *, support: dict[str, float | int]) -> pd.DataFrame:
        supported = self._supported_for_top_sections(frame, support=support)
        if supported.empty:
            return supported
        live_supported = supported[supported["live_match_count"] > 0].copy()
        return self._best_per_sector(live_supported)

    def _best_walk_forward_stable(self, frame: pd.DataFrame, *, top: int) -> pd.DataFrame:
        if frame.empty or "wf_stability_score" not in frame.columns:
            return frame.iloc[0:0].copy()
        return (
            frame[frame["wf_stability_score"].notna()]
            .sort_values(
                ["wf_stability_score", "wf_positive_window_ratio", "practical_score"],
                ascending=[False, False, False],
            )
            .head(top)
            .copy()
        )

    def _best_stable_per_sector(self, frame: pd.DataFrame, *, support: dict[str, float | int]) -> pd.DataFrame:
        supported = self._supported_for_top_sections(frame, support=support)
        if supported.empty or "wf_stability_score" not in supported.columns:
            return supported.iloc[0:0].copy()
        return (
            supported[supported["wf_stability_score"].notna()]
            .sort_values(
                ["wf_stability_score", "wf_positive_window_ratio", "practical_score"],
                ascending=[False, False, False],
            )
            .drop_duplicates(subset=["sector"], keep="first")
            .reset_index(drop=True)
        )

    def _build_walk_forward_stability(
        self,
        *,
        ranked: pd.DataFrame,
        analysis_frame: pd.DataFrame,
        windows: int,
        shortlist_size: int,
        sleeve_config: dict[str, object],
        latest_date: pd.Timestamp,
        sweep_service: SweepService,
        benchmark_price_maps: dict[str, pd.DataFrame],
        backtest_costs,
    ) -> pd.DataFrame:
        shortlist = self._build_walk_forward_shortlist(ranked, shortlist_size=shortlist_size)
        if shortlist.empty:
            return pd.DataFrame(columns=self._sleeve_key_columns())

        stability_rows: list[dict[str, float | int | str]] = []
        for row in shortlist.itertuples(index=False):
            sector_frame = analysis_frame[analysis_frame["sector"] == row.sector].copy()
            if sector_frame.empty:
                continue
            unique_dates = sorted(pd.to_datetime(sector_frame["date"]).drop_duplicates().tolist())
            if len(unique_dates) < max(windows + 1, 4):
                continue
            date_chunks = [
                list(chunk)
                for chunk in np.array_split(np.array(unique_dates, dtype=object), windows + 1)
                if len(chunk) > 0
            ]
            test_chunks = date_chunks[1:]
            if len(test_chunks) < windows:
                continue

            window_metrics: list[dict[str, object]] = []
            for test_dates in test_chunks:
                start_date = pd.Timestamp(test_dates[0])
                end_date = pd.Timestamp(test_dates[-1])
                scoped_frame = sector_frame[
                    (pd.to_datetime(sector_frame["date"]) >= start_date)
                    & (pd.to_datetime(sector_frame["date"]) <= end_date)
                ].copy()
                if scoped_frame.empty:
                    continue
                metrics = self._evaluate_sector_configuration(
                    sector_frame=scoped_frame,
                    sector=str(row.sector),
                    top_n=int(row.top_n),
                    horizon_days=int(row.horizon_days),
                    breadth_filters={
                        "sector_pct_above_50_min": float(row.sector_pct_above_50_min),
                        "sector_pct_above_200_min": float(row.sector_pct_above_200_min),
                        "sector_median_roc_63_min": float(row.sector_median_roc_63_min),
                    },
                    base_filters=sleeve_config["base_filters"],
                    rank_weights=sleeve_config["rank_weights"],
                    backtest_costs=backtest_costs,
                    benchmark_context=BenchmarkContext(
                        spy_ticker="SPY",
                        sector_ticker=benchmark_etf_for_sector(str(row.sector)),
                        price_maps=benchmark_price_maps,
                    ),
                    latest_date=min(latest_date, end_date),
                    sweep_service=sweep_service,
                )
                window_metrics.append(metrics)
            if not window_metrics:
                continue
            stability_rows.append(
                {
                    "sector": str(row.sector),
                    "top_n": int(row.top_n),
                    "horizon_days": int(row.horizon_days),
                    "sector_pct_above_50_min": float(row.sector_pct_above_50_min),
                    "sector_pct_above_200_min": float(row.sector_pct_above_200_min),
                    "sector_median_roc_63_min": float(row.sector_median_roc_63_min),
                    **self._summarize_walk_forward_metrics(window_metrics),
                }
            )
        return pd.DataFrame(stability_rows)

    def _build_walk_forward_shortlist(self, ranked: pd.DataFrame, *, shortlist_size: int) -> pd.DataFrame:
        shortlist_parts = [
            ranked.head(shortlist_size),
            self._best_per_sector(ranked).head(shortlist_size),
            self._best_live_supported(ranked, min_trade_count=1, top=shortlist_size),
            ranked.sort_values(
                ["alpha_vs_sector", "alpha_vs_spy", "practical_score"],
                ascending=[False, False, False],
            ).head(shortlist_size),
        ]
        return (
            pd.concat(shortlist_parts, ignore_index=True)
            .drop_duplicates(subset=self._sleeve_key_columns(), keep="first")
            .head(shortlist_size)
            .reset_index(drop=True)
        )

    def _summarize_walk_forward_metrics(self, window_metrics: list[dict[str, object]]) -> dict[str, float | int]:
        expectancy_values = [float(metrics["expectancy"]) for metrics in window_metrics]
        alpha_values = [
            float(metrics["alpha_vs_spy"])
            for metrics in window_metrics
            if metrics.get("alpha_vs_spy") is not None and np.isfinite(float(metrics["alpha_vs_spy"]))
        ]
        mdd_values = [float(metrics["mdd"]) for metrics in window_metrics]
        trade_counts = [int(metrics["trade_count"]) for metrics in window_metrics]

        positive_window_ratio = (
            sum(1 for value in expectancy_values if value > 0) / len(expectancy_values)
            if expectancy_values
            else 0.0
        )
        positive_alpha_window_ratio = (
            sum(1 for value in alpha_values if value > 0) / len(window_metrics)
            if window_metrics
            else 0.0
        )
        median_expectancy = float(np.median(expectancy_values)) if expectancy_values else 0.0
        worst_expectancy = float(min(expectancy_values)) if expectancy_values else 0.0
        median_alpha_vs_spy = float(np.median(alpha_values)) if alpha_values else np.nan
        worst_mdd = float(max(mdd_values)) if mdd_values else 0.0
        trade_count_min = min(trade_counts) if trade_counts else 0
        return {
            "wf_window_count": len(window_metrics),
            "wf_median_expectancy": median_expectancy,
            "wf_worst_expectancy": worst_expectancy,
            "wf_positive_window_ratio": positive_window_ratio,
            "wf_positive_alpha_window_ratio": positive_alpha_window_ratio,
            "wf_median_alpha_vs_spy": median_alpha_vs_spy,
            "wf_worst_mdd": worst_mdd,
            "wf_trade_count_min": trade_count_min,
            "wf_stability_score": self._score_walk_forward_summary(
                median_expectancy=median_expectancy,
                worst_expectancy=worst_expectancy,
                positive_window_ratio=positive_window_ratio,
                positive_alpha_window_ratio=positive_alpha_window_ratio,
                worst_mdd=worst_mdd,
                trade_count_min=trade_count_min,
            ),
        }

    def _score_walk_forward_summary(
        self,
        *,
        median_expectancy: float,
        worst_expectancy: float,
        positive_window_ratio: float,
        positive_alpha_window_ratio: float,
        worst_mdd: float,
        trade_count_min: int,
    ) -> float:
        expectancy_term = max(min(median_expectancy * 20.0, 0.40), -0.40)
        worst_expectancy_term = max(min(worst_expectancy * 15.0, 0.20), -0.20)
        trade_support_term = min(max(trade_count_min, 0), 20) / 20.0 * 0.10
        return (
            expectancy_term
            + worst_expectancy_term
            + positive_window_ratio * 0.30
            + positive_alpha_window_ratio * 0.15
            + trade_support_term
            - worst_mdd * 0.25
        )

    def _sleeve_key_columns(self) -> list[str]:
        return [
            "sector",
            "top_n",
            "horizon_days",
            "sector_pct_above_50_min",
            "sector_pct_above_200_min",
            "sector_median_roc_63_min",
        ]

    def _min_max(self, series: pd.Series) -> pd.Series:
        normalized = series.astype(float).copy()
        finite_mask = np.isfinite(normalized)
        finite_values = normalized[finite_mask]
        if finite_values.empty:
            return pd.Series([0.0] * len(normalized.index), index=normalized.index)
        if (~finite_mask).any():
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            spread = finite_max - finite_min
            replacement = finite_max + (spread if spread > 0 else 1.0)
            normalized.loc[~finite_mask] = replacement
        min_value = normalized.min()
        max_value = normalized.max()
        if min_value == max_value:
            return pd.Series([0.0] * len(normalized.index), index=normalized.index)
        return (normalized - min_value) / (max_value - min_value)

    def _render_section(self, title: str, frame: pd.DataFrame, *, empty_message: str = "No results.") -> list[str]:
        lines = [f"## {title}", ""]
        if frame.empty:
            lines.append(empty_message)
            lines.append("")
            return lines
        for row in frame.itertuples(index=False):
            lines.append(f"### {row.sector} | top_n={row.top_n} | horizon={row.horizon_days}d")
            lines.append(f"- global_rank: {row.global_rank}")
            lines.append(f"- sector_rank: {row.sector_rank}")
            lines.append(f"- practical_score: {row.practical_score:.6f}")
            lines.append(f"- expectancy: {row.expectancy:.6f}")
            lines.append(f"- profit_factor: {row.profit_factor:.6f}")
            lines.append(
                f"- alpha_vs_spy: {row.alpha_vs_spy:.6f}" if np.isfinite(float(row.alpha_vs_spy)) else "- alpha_vs_spy: unknown"
            )
            lines.append(
                f"- alpha_vs_sector: {row.alpha_vs_sector:.6f}" if np.isfinite(float(row.alpha_vs_sector)) else "- alpha_vs_sector: unknown"
            )
            lines.append(f"- mdd: {row.mdd:.6f}")
            lines.append(f"- win_rate: {row.win_rate:.6f}")
            lines.append(f"- trade_count: {int(row.trade_count)}")
            lines.append(f"- distinct_tickers_traded: {int(row.distinct_tickers_traded)}")
            lines.append(f"- max_ticker_trade_share: {float(row.max_ticker_trade_share):.6f}")
            lines.append(f"- live_match_count: {int(row.live_match_count)}")
            lines.append(
                f"- live_match_tickers: {', '.join(row.live_match_tickers) if row.live_match_tickers else 'none'}"
            )
            if hasattr(row, "trade_support_ratio") and pd.notna(getattr(row, "trade_support_ratio", np.nan)):
                lines.append(f"- trade_support_ratio: {float(row.trade_support_ratio):.6f}")
            if hasattr(row, "low_support_penalty") and pd.notna(getattr(row, "low_support_penalty", np.nan)):
                lines.append(f"- low_support_penalty: {float(row.low_support_penalty):.6f}")
            if hasattr(row, "distinct_ticker_support_ratio") and pd.notna(getattr(row, "distinct_ticker_support_ratio", np.nan)):
                lines.append(f"- distinct_ticker_support_ratio: {float(row.distinct_ticker_support_ratio):.6f}")
            if hasattr(row, "low_distinct_ticker_penalty") and pd.notna(getattr(row, "low_distinct_ticker_penalty", np.nan)):
                lines.append(f"- low_distinct_ticker_penalty: {float(row.low_distinct_ticker_penalty):.6f}")
            if hasattr(row, "ticker_concentration_penalty") and pd.notna(getattr(row, "ticker_concentration_penalty", np.nan)):
                lines.append(f"- ticker_concentration_penalty: {float(row.ticker_concentration_penalty):.6f}")
            if hasattr(row, "wf_stability_score") and pd.notna(getattr(row, "wf_stability_score", np.nan)):
                lines.append(f"- wf_stability_score: {float(row.wf_stability_score):.6f}")
                lines.append(f"- wf_positive_window_ratio: {float(row.wf_positive_window_ratio):.6f}")
                lines.append(f"- wf_positive_alpha_window_ratio: {float(row.wf_positive_alpha_window_ratio):.6f}")
                lines.append(f"- wf_median_expectancy: {float(row.wf_median_expectancy):.6f}")
                lines.append(f"- wf_worst_expectancy: {float(row.wf_worst_expectancy):.6f}")
                lines.append(
                    f"- wf_median_alpha_vs_spy: {float(row.wf_median_alpha_vs_spy):.6f}"
                    if np.isfinite(float(row.wf_median_alpha_vs_spy))
                    else "- wf_median_alpha_vs_spy: unknown"
                )
                lines.append(f"- wf_worst_mdd: {float(row.wf_worst_mdd):.6f}")
                lines.append(f"- wf_trade_count_min: {int(row.wf_trade_count_min)}")
            lines.append(
                f"- breadth_filters: sector_pct_above_50>={row.sector_pct_above_50_min:.2f}, "
                f"sector_pct_above_200>={row.sector_pct_above_200_min:.2f}, "
                f"sector_median_roc_63>={row.sector_median_roc_63_min:.2f}"
            )
            lines.append(f"- base_filters: `{row.base_filters}`")
            lines.append(f"- rank_weights: `{row.rank_weights}`")
            lines.append("")
        return lines
