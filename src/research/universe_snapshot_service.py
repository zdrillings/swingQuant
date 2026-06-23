from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates
from src.utils.strategy import load_active_strategies


OUTCOME_HORIZONS = (1, 3, 5, 10, 20)
SNAPSHOT_REFRESH_COLUMNS = (
    "sub_industry",
    "subindustry_benchmark",
    "relative_strength_index_vs_subindustry",
)
SNAPSHOT_FEATURE_COLUMNS = [
    "md_volume_30d",
    "adj_close",
    "atr_14",
    "relative_strength_index_vs_spy",
    "relative_strength_index_vs_qqq",
    "relative_strength_index_vs_xlk",
    "relative_strength_index_vs_subindustry",
    "roc_63",
    "roc_126",
    "vol_alpha",
    "sma_200_dist",
    "sma_50_dist",
    "rsi_14",
    "days_to_next_earnings",
    "days_since_last_earnings",
    "last_earnings_gap_pct",
    "last_earnings_volume_ratio_20",
    "last_earnings_open_vs_20d_high",
    "close_vs_last_earnings_close",
    "avg_abs_gap_pct_20",
    "max_gap_down_pct_60",
    "distance_above_20d_high",
    "base_range_pct_20",
    "base_atr_contraction_20",
    "base_volume_dryup_ratio_20",
    "breakout_volume_ratio_50",
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
]


@dataclass(frozen=True)
class UniverseSnapshotBackfillReport:
    snapshot_dates_processed: int
    snapshot_dates_skipped: int
    total_rows: int


class UniverseSnapshotBackfillService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("universe_snapshot_backfill")

    def run(
        self,
        *,
        date_from: str,
        date_to: str | None = None,
        skip_existing: bool = False,
    ) -> UniverseSnapshotBackfillReport:
        self.db_manager.initialize()
        strategies = load_active_strategies()
        universe_rows = self.db_manager.list_universe_rows(active_only=True)
        if not universe_rows:
            raise ValueError("Universe is empty. Run `sq sync` first.")

        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")
        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, _ = build_analysis_frame(
            price_history,
            universe_rows,
            earnings_calendar=earnings_calendar,
        )
        if analysis_frame.empty:
            raise ValueError("No analysis frame could be built for historical universe snapshot backfill.")

        snapshot_dates = self._snapshot_dates(
            analysis_frame=analysis_frame,
            universe_tickers=universe_tickers,
            date_from=date_from,
            date_to=date_to,
        )
        if not snapshot_dates:
            raise ValueError("No trading dates matched the requested universe snapshot range.")

        history_context = self._history_context(price_history)
        processed = 0
        skipped = 0
        total_rows = 0
        existing_dates: set[str] = set()
        if skip_existing:
            existing_dates = set(self.db_manager.list_universe_daily_snapshot_dates())

        for snapshot_date in snapshot_dates:
            snapshot_date_str = snapshot_date.strftime("%Y-%m-%d")
            if skip_existing and snapshot_date_str in existing_dates:
                needs_refresh = False
                refresh_probe = getattr(self.db_manager, "universe_daily_snapshot_date_needs_refresh", None)
                if callable(refresh_probe):
                    needs_refresh = bool(
                        refresh_probe(
                            snapshot_date=snapshot_date_str,
                            required_non_null_columns=SNAPSHOT_REFRESH_COLUMNS,
                        )
                    )
                if not needs_refresh:
                    skipped += 1
                    continue
                self.logger.info(
                    "Universe snapshot date=%s marked stale; recomputing despite skip_existing.",
                    snapshot_date_str,
                )
            day_frame = analysis_frame[
                (pd.to_datetime(analysis_frame["date"]).dt.normalize() == snapshot_date)
                & analysis_frame["ticker"].isin(universe_tickers)
            ].copy()
            rows = self._build_rows_for_date(
                snapshot_date=snapshot_date_str,
                day_frame=day_frame,
                strategies=strategies,
                history_context=history_context,
            )
            self.db_manager.replace_universe_daily_snapshots(snapshot_date=snapshot_date_str, rows=rows)
            processed += 1
            total_rows += len(rows)
            if processed == 1 or processed % 20 == 0 or processed == len(snapshot_dates):
                self.logger.info(
                    "Universe snapshot backfill progress: processed=%s/%s current_date=%s rows=%s",
                    processed,
                    len(snapshot_dates),
                    snapshot_date_str,
                    len(rows),
                )

        return UniverseSnapshotBackfillReport(
            snapshot_dates_processed=processed,
            snapshot_dates_skipped=skipped,
            total_rows=total_rows,
        )

    def _snapshot_dates(
        self,
        *,
        analysis_frame: pd.DataFrame,
        universe_tickers: list[str],
        date_from: str,
        date_to: str | None,
    ) -> list[pd.Timestamp]:
        working = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        if working.empty:
            return []
        all_dates = sorted(pd.to_datetime(working["date"]).dt.normalize().drop_duplicates().tolist())
        start = pd.Timestamp(date_from).normalize()
        end = pd.Timestamp(date_to).normalize() if date_to is not None else all_dates[-1]
        return [date_value for date_value in all_dates if start <= date_value <= end]

    def _build_rows_for_date(
        self,
        *,
        snapshot_date: str,
        day_frame: pd.DataFrame,
        strategies: dict,
        history_context: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        if day_frame.empty:
            return []
        passed_slots_by_ticker: dict[str, list[str]] = {}
        for slot, strategy in strategies.items():
            scoped = day_frame if strategy.sector == "ALL" else day_frame.loc[day_frame["sector"] == strategy.sector].copy()
            if scoped.empty:
                continue
            passed = filter_signal_candidates(scoped, strategy.indicators)
            if passed.empty:
                continue
            for ticker in passed["ticker"].astype(str).tolist():
                passed_slots_by_ticker.setdefault(ticker, []).append(str(slot))
        rows: list[dict[str, object]] = []
        for row in day_frame.to_dict(orient="records"):
            ticker = str(row["ticker"])
            sector = str(row.get("sector", ""))
            passed_slots = sorted(set(passed_slots_by_ticker.get(ticker, [])))
            detail_payload = {
                "indicator_details": row.get("indicator_details", {}) or {},
                "regime_green": bool(row.get("regime_green", False)),
            }
            snapshot_row = {
                "ticker": ticker,
                "sector": sector,
                "sub_industry": row.get("sub_industry"),
                "subindustry_benchmark": row.get("subindustry_benchmark"),
                "regime_etf": row.get("regime_etf"),
                "regime_green": bool(row.get("regime_green", False)),
                "passed_any_strategy": bool(passed_slots),
                "strategy_pass_count": len(passed_slots),
                "passed_slots": passed_slots,
                "details": detail_payload,
            }
            for column in SNAPSHOT_FEATURE_COLUMNS:
                snapshot_row[column] = self._optional_float(row.get(column))
            snapshot_row.update(
                self._outcome_payload(
                    snapshot_date=snapshot_date,
                    ticker=ticker,
                    sector=sector,
                    history_context=history_context,
                )
            )
            rows.append(snapshot_row)
        return rows

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

    def _outcome_payload(
        self,
        *,
        snapshot_date: str,
        ticker: str,
        sector: str,
        history_context: dict[str, dict[str, object]],
    ) -> dict[str, float | None]:
        payload: dict[str, float | None] = {}
        ticker_context = history_context.get(ticker)
        if ticker_context is None:
            return payload
        ticker_frame = ticker_context["frame"]
        index = ticker_context["index_by_date"].get(snapshot_date)
        if index is None:
            return payload
        benchmark_ticker = benchmark_etf_for_sector(sector)
        for horizon in OUTCOME_HORIZONS:
            payload[f"fwd_return_{horizon}d"] = self._forward_return(ticker_frame=ticker_frame, index=int(index), horizon=horizon)
            payload[f"alpha_vs_spy_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                snapshot_date=snapshot_date,
                index=int(index),
                horizon=horizon,
                benchmark_ticker="SPY",
            )
            payload[f"alpha_vs_sector_{horizon}d"] = self._alpha_vs_benchmark(
                history_context=history_context,
                ticker_frame=ticker_frame,
                snapshot_date=snapshot_date,
                index=int(index),
                horizon=horizon,
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

    def _forward_return(self, *, ticker_frame: pd.DataFrame, index: int, horizon: int) -> float | None:
        future_index = index + int(horizon)
        if future_index >= len(ticker_frame.index):
            return None
        entry_price = float(ticker_frame.loc[index, "adj_close"])
        future_price = float(ticker_frame.loc[future_index, "adj_close"])
        return (future_price / entry_price) - 1.0

    def _alpha_vs_benchmark(
        self,
        *,
        history_context: dict[str, dict[str, object]],
        ticker_frame: pd.DataFrame,
        snapshot_date: str,
        index: int,
        horizon: int,
        benchmark_ticker: str | None,
    ) -> float | None:
        raw_return = self._forward_return(ticker_frame=ticker_frame, index=index, horizon=horizon)
        if benchmark_ticker in (None, "") or raw_return is None:
            return None
        benchmark_context = history_context.get(str(benchmark_ticker))
        if benchmark_context is None:
            return None
        benchmark_index = benchmark_context["index_by_date"].get(snapshot_date)
        if benchmark_index is None:
            return None
        benchmark_return = self._forward_return(
            ticker_frame=benchmark_context["frame"],
            index=int(benchmark_index),
            horizon=horizon,
        )
        if benchmark_return is None:
            return None
        return raw_return - benchmark_return

    def _excursion(
        self,
        *,
        ticker_frame: pd.DataFrame,
        index: int,
        horizon: int,
        column: str,
        use_max: bool,
    ) -> float | None:
        start_index = index + 1
        end_index = min(index + int(horizon), len(ticker_frame.index) - 1)
        if start_index > end_index:
            return None
        window = ticker_frame.loc[start_index:end_index, column].astype(float)
        if window.empty:
            return None
        entry_price = float(ticker_frame.loc[index, "adj_close"])
        extreme_price = float(window.max() if use_max else window.min())
        return (extreme_price / entry_price) - 1.0

    def _optional_float(self, value) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
