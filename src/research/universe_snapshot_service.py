from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, datetime
import json
import re

import pandas as pd

from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, filter_signal_candidates
from src.utils.strategy import (
    ExitRules,
    entry_stop_price,
    load_active_strategies,
    profit_target_price,
    trailing_stop_price,
)


OUTCOME_HORIZONS = (1, 3, 5, 10, 20)
SNAPSHOT_REFRESH_COLUMNS = (
    "sub_industry",
    "subindustry_benchmark",
    "relative_strength_index_vs_subindustry",
    "rs_vs_spy_5d_change",
    "rs_vs_subindustry_5d_change",
    "rsi_2",
    "ret_1d",
    "ret_5d",
    "close_vs_20d_low",
    "spy_roc_20",
    "spy_roc_5",
    "spy_realized_vol_20",
    "qqq_roc_20",
)
SNAPSHOT_OUTCOME_COLUMNS = tuple(
    column
    for horizon in OUTCOME_HORIZONS
    for column in (
        f"fwd_return_{horizon}d",
        f"alpha_vs_spy_{horizon}d",
        f"alpha_vs_sector_{horizon}d",
    )
) + (
    "alpha_vs_sector_20d_pos",
    "mfe_20d",
    "mae_20d",
    "path_return_20d",
    "path_alpha_vs_sector_20d",
    "path_exit_reason_20d",
)
SNAPSHOT_FEATURE_COLUMNS = [
    "md_volume_30d",
    "adj_close",
    "atr_14",
    "atr_pct_14",
    "atr_pct_14_percentile_252",
    "realized_vol_20_percentile_252",
    "relative_strength_index_vs_spy",
    "relative_strength_index_vs_qqq",
    "relative_strength_index_vs_xlk",
    "relative_strength_index_vs_subindustry",
    "rs_vs_spy_5d_change",
    "rs_vs_qqq_5d_change",
    "rs_vs_xlk_5d_change",
    "rs_vs_subindustry_5d_change",
    "rs_vs_subindustry_10d_change",
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
    "dollar_volume_ratio_20_60",
    "volume_percentile_60",
    "distance_from_52w_high",
    "days_since_52w_high",
    "rsi_2",
    "ret_1d",
    "ret_5d",
    "close_vs_20d_low",
    "sector_pct_above_50",
    "sector_pct_above_200",
    "sector_median_roc_63",
    "spy_roc_20",
    "spy_roc_5",
    "spy_realized_vol_20",
    "qqq_roc_20",
    "analyst_target_upside",
    "analyst_target_range_pct",
    "analyst_count",
    "analyst_recommendation_score",
    "analyst_eps_revision_breadth",
    "analyst_upgrade_downgrade_score",
    "analyst_snapshot_age_days",
    "analyst_revision_snapshot_age_days",
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
        strategies = {
            slot: strategy
            for slot, strategy in load_active_strategies().items()
            if getattr(strategy, "scan_enabled", True)
        }
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
        analyst_context = self._analyst_context()
        analyst_revision_context = self._analyst_revision_context()
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
                    required_columns = self._required_refresh_columns_for_snapshot(
                        snapshot_date=snapshot_date_str,
                        history_context=history_context,
                        analyst_context=analyst_context,
                        analyst_revision_context=analyst_revision_context,
                    )
                    needs_refresh = bool(
                        refresh_probe(
                            snapshot_date=snapshot_date_str,
                            required_non_null_columns=required_columns,
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
                strategies=self._strategies_effective_on(strategies, snapshot_date_str),
                path_label_strategies=strategies,
                history_context=history_context,
                analyst_context=analyst_context,
                analyst_revision_context=analyst_revision_context,
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
        path_label_strategies: dict | None = None,
        analyst_context: dict[str, pd.DataFrame] | None = None,
        analyst_revision_context: dict[str, pd.DataFrame] | None = None,
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
                self._analyst_feature_payload(
                    snapshot_date=snapshot_date,
                    ticker=ticker,
                    adj_close=snapshot_row.get("adj_close"),
                    analyst_context=analyst_context or {},
                    analyst_revision_context=analyst_revision_context or {},
                )
            )
            snapshot_row.update(
                self._outcome_payload(
                    snapshot_date=snapshot_date,
                    ticker=ticker,
                    sector=sector,
                    history_context=history_context,
                    exit_rules=self._exit_rules_for_snapshot(
                        strategies=path_label_strategies or strategies,
                        passed_slots=passed_slots,
                        sector=sector,
                    ),
                )
            )
            rows.append(snapshot_row)
        return rows

    def _strategies_effective_on(self, strategies: dict, snapshot_date: str) -> dict:
        effective: dict = {}
        for slot, strategy in strategies.items():
            promoted_at = getattr(strategy, "promoted_at", None)
            promoted_date = self._promoted_date(promoted_at)
            if promoted_date is None or promoted_date <= pd.Timestamp(snapshot_date).date():
                effective[slot] = strategy
        return effective

    def _promoted_date(self, value) -> date_type | None:
        if value in (None, ""):
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return pd.Timestamp(value).date()
            except Exception:
                return None

    def _exit_rules_for_snapshot(self, *, strategies: dict, passed_slots: list[str], sector: str) -> ExitRules | None:
        for slot in passed_slots:
            strategy = strategies.get(str(slot))
            if strategy is not None:
                return getattr(strategy, "exit_rules", None)
        for strategy in strategies.values():
            strategy_sector = str(getattr(strategy, "sector", ""))
            if strategy_sector == "ALL" or strategy_sector == str(sector):
                return getattr(strategy, "exit_rules", None)
        return None

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

    def _analyst_context(self) -> dict[str, pd.DataFrame]:
        loader = getattr(self.db_manager, "load_analyst_snapshots", None)
        if not callable(loader):
            return {}
        frame = loader()
        if frame.empty:
            return {}
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        return {
            str(ticker): group.sort_values("snapshot_date").reset_index(drop=True)
            for ticker, group in working.groupby(working["ticker"].astype(str).str.upper(), sort=False)
        }

    def _analyst_revision_context(self) -> dict[str, pd.DataFrame]:
        loader = getattr(self.db_manager, "load_analyst_revision_snapshots", None)
        if not callable(loader):
            return {}
        frame = loader()
        if frame.empty:
            return {}
        working = frame.copy()
        working["snapshot_date"] = pd.to_datetime(working["snapshot_date"]).dt.normalize()
        return {
            str(ticker): group.sort_values("snapshot_date").reset_index(drop=True)
            for ticker, group in working.groupby(working["ticker"].astype(str).str.upper(), sort=False)
        }

    def _analyst_feature_payload(
        self,
        *,
        snapshot_date: str,
        ticker: str,
        adj_close: float | None,
        analyst_context: dict[str, pd.DataFrame],
        analyst_revision_context: dict[str, pd.DataFrame],
    ) -> dict[str, float | None]:
        payload = {
            "analyst_target_upside": None,
            "analyst_target_range_pct": None,
            "analyst_count": None,
            "analyst_recommendation_score": None,
            "analyst_eps_revision_breadth": None,
            "analyst_upgrade_downgrade_score": None,
            "analyst_snapshot_age_days": None,
            "analyst_revision_snapshot_age_days": None,
        }
        snapshot_ts = pd.Timestamp(snapshot_date).normalize()
        analyst_row = self._latest_point_in_time_row(analyst_context.get(str(ticker).upper()), snapshot_ts)
        if analyst_row is not None:
            target_mean = self._optional_float(analyst_row.get("target_mean"))
            target_low = self._optional_float(analyst_row.get("target_low"))
            target_high = self._optional_float(analyst_row.get("target_high"))
            close = self._optional_float(adj_close)
            if target_mean is not None and close is not None and close > 0:
                payload["analyst_target_upside"] = (float(target_mean) / float(close)) - 1.0
            if target_low is not None and target_high is not None and close is not None and close > 0:
                payload["analyst_target_range_pct"] = (float(target_high) - float(target_low)) / float(close)
            payload["analyst_count"] = self._optional_float(analyst_row.get("analyst_count"))
            payload["analyst_recommendation_score"] = self._recommendation_score(analyst_row.get("recommendation"))
            payload["analyst_snapshot_age_days"] = float((snapshot_ts - pd.Timestamp(analyst_row["snapshot_date"]).normalize()).days)

        revision_row = self._latest_point_in_time_row(analyst_revision_context.get(str(ticker).upper()), snapshot_ts)
        if revision_row is not None:
            payload["analyst_eps_revision_breadth"] = self._eps_revision_breadth(
                self._json_records(revision_row.get("eps_revisions_json"))
            )
            payload["analyst_upgrade_downgrade_score"] = self._upgrade_downgrade_score(
                self._json_records(revision_row.get("upgrades_downgrades_json"))
            )
            payload["analyst_revision_snapshot_age_days"] = float((snapshot_ts - pd.Timestamp(revision_row["snapshot_date"]).normalize()).days)
        return payload

    def _latest_point_in_time_row(self, frame: pd.DataFrame | None, snapshot_ts: pd.Timestamp):
        if frame is None or frame.empty:
            return None
        eligible = frame.loc[pd.to_datetime(frame["snapshot_date"]).dt.normalize() <= snapshot_ts]
        if eligible.empty:
            return None
        return eligible.sort_values("snapshot_date").iloc[-1]

    def _recommendation_score(self, recommendation: object) -> float | None:
        if recommendation in (None, "") or pd.isna(recommendation):
            return None
        weights = {
            "strong buy": 2.0,
            "buy": 1.0,
            "hold": 0.0,
            "sell": -1.0,
            "strong sell": -2.0,
        }
        total = 0.0
        count = 0.0
        text = str(recommendation).lower()
        for raw_count, raw_label in re.findall(r"(\d+)\s+([a-z ]+?)(?=,|$)", text):
            label = raw_label.strip()
            if label not in weights:
                continue
            weight = float(raw_count)
            total += weight * weights[label]
            count += weight
        return total / count if count > 0 else None

    def _json_records(self, value: object) -> list[dict]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [record for record in parsed if isinstance(record, dict)] if isinstance(parsed, list) else []

    def _eps_revision_breadth(self, records: list[dict]) -> float | None:
        up_total = 0.0
        down_total = 0.0
        for record in records:
            for key, value in record.items():
                normalized = str(key).lower()
                numeric = self._optional_float(value)
                if numeric is None:
                    continue
                if "up" in normalized:
                    up_total += float(numeric)
                elif "down" in normalized:
                    down_total += float(numeric)
        total = up_total + down_total
        return (up_total - down_total) / total if total > 0 else None

    def _upgrade_downgrade_score(self, records: list[dict]) -> float | None:
        if not records:
            return None
        score = 0.0
        observations = 0
        for record in records:
            text = " ".join(str(value).lower() for value in record.values())
            if "upgrade" in text or "initiated" in text or "raised" in text:
                score += 1.0
                observations += 1
            elif "downgrade" in text or "lowered" in text or "cut" in text:
                score -= 1.0
                observations += 1
        return score / observations if observations else None

    def _required_refresh_columns_for_snapshot(
        self,
        *,
        snapshot_date: str,
        history_context: dict[str, dict[str, object]],
        analyst_context: dict[str, pd.DataFrame],
        analyst_revision_context: dict[str, pd.DataFrame],
    ) -> tuple[str, ...]:
        required = list(SNAPSHOT_REFRESH_COLUMNS)
        snapshot_ts = pd.Timestamp(snapshot_date).normalize()
        if any(self._latest_point_in_time_row(frame, snapshot_ts) is not None for frame in analyst_context.values()):
            required.extend(
                [
                    "analyst_target_upside",
                    "analyst_target_range_pct",
                    "analyst_count",
                    "analyst_recommendation_score",
                    "analyst_snapshot_age_days",
                ]
            )
        if any(self._latest_point_in_time_row(frame, snapshot_ts) is not None for frame in analyst_revision_context.values()):
            required.extend(
                [
                    "analyst_eps_revision_breadth",
                    "analyst_upgrade_downgrade_score",
                    "analyst_revision_snapshot_age_days",
                ]
            )
        spy_context = history_context.get("SPY")
        if spy_context is None:
            return tuple(required)
        index = spy_context["index_by_date"].get(snapshot_date)
        if index is None:
            return tuple(required)
        available_forward_sessions = len(spy_context["frame"].index) - int(index) - 1
        for horizon in OUTCOME_HORIZONS:
            if available_forward_sessions < horizon:
                continue
            required.extend(
                [
                    f"fwd_return_{horizon}d",
                    f"alpha_vs_spy_{horizon}d",
                    f"alpha_vs_sector_{horizon}d",
                ]
            )
        if available_forward_sessions >= 20:
            required.extend(
                [
                    "alpha_vs_sector_20d_pos",
                    "mfe_20d",
                    "mae_20d",
                    "path_return_20d",
                    "path_alpha_vs_sector_20d",
                    "path_exit_reason_20d",
                ]
            )
        return tuple(required)

    def _outcome_payload(
        self,
        *,
        snapshot_date: str,
        ticker: str,
        sector: str,
        history_context: dict[str, dict[str, object]],
        exit_rules: ExitRules | None = None,
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
        alpha_20d = payload.get("alpha_vs_sector_20d")
        if alpha_20d is None:
            payload["alpha_vs_sector_20d_pos"] = None
        else:
            payload["alpha_vs_sector_20d_pos"] = 1 if float(alpha_20d) > 0.02 else 0
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
        path_outcome = self._path_outcome(
            ticker_frame=ticker_frame,
            index=int(index),
            horizon=20,
            exit_rules=exit_rules,
        )
        payload["path_return_20d"] = path_outcome["return"]
        payload["path_exit_reason_20d"] = path_outcome["exit_reason"]
        benchmark_ticker = benchmark_etf_for_sector(sector)
        benchmark_return = self._path_benchmark_return(
            history_context=history_context,
            snapshot_date=snapshot_date,
            horizon=int(path_outcome["holding_days"]) if path_outcome["holding_days"] is not None else 20,
            benchmark_ticker=benchmark_ticker,
        )
        if path_outcome["return"] is None or benchmark_return is None:
            payload["path_alpha_vs_sector_20d"] = None
        else:
            payload["path_alpha_vs_sector_20d"] = float(path_outcome["return"]) - float(benchmark_return)
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

    def _path_outcome(
        self,
        *,
        ticker_frame: pd.DataFrame,
        index: int,
        horizon: int,
        exit_rules: ExitRules | None,
    ) -> dict[str, float | str | int | None]:
        if exit_rules is None:
            return {"return": None, "exit_reason": None, "holding_days": None}
        entry_price = self._optional_float(ticker_frame.loc[index, "adj_close"])
        if entry_price is None or entry_price <= 0:
            return {"return": None, "exit_reason": None, "holding_days": None}
        entry_atr = self._optional_float(ticker_frame.loc[index].get("atr_14"))
        start_index = int(index) + 1
        max_horizon = min(int(horizon), int(exit_rules.time_limit_days or horizon))
        end_index = min(int(index) + max_horizon, len(ticker_frame.index) - 1)
        if start_index > end_index:
            return {"return": None, "exit_reason": None, "holding_days": None}
        max_price_seen = float(entry_price)
        last_close = None
        held_days = 0
        for forward_index in range(start_index, end_index + 1):
            row = ticker_frame.loc[forward_index]
            open_price = self._optional_float(row.get("open"))
            high = self._optional_float(row.get("high"))
            low = self._optional_float(row.get("low"))
            close = self._optional_float(row.get("adj_close"))
            if close is None:
                close = self._optional_float(row.get("close"))
            if open_price is None:
                open_price = close
            if high is None or low is None or close is None:
                continue
            held_days += 1
            max_price_seen = max(max_price_seen, float(high))
            last_close = float(close)
            hard_stop = entry_stop_price(entry_price=float(entry_price), exit_rules=exit_rules)
            try:
                stop_price = trailing_stop_price(
                    max_price_seen=max_price_seen,
                    entry_atr=entry_atr,
                    exit_rules=exit_rules,
                )
            except ValueError:
                stop_price = None
            try:
                target_price = profit_target_price(
                    entry_price=float(entry_price),
                    entry_atr=entry_atr,
                    exit_rules=exit_rules,
                )
            except ValueError:
                target_price = None
            if hard_stop is not None and float(low) <= float(hard_stop):
                exit_price = float(open_price) if open_price is not None and float(open_price) < float(hard_stop) else float(hard_stop)
                reason = "hard_stop_gap" if exit_price < float(hard_stop) else "hard_stop"
                return {"return": (exit_price / float(entry_price)) - 1.0, "exit_reason": reason, "holding_days": held_days}
            if stop_price is not None and float(low) <= float(stop_price):
                return {"return": (float(stop_price) / float(entry_price)) - 1.0, "exit_reason": "trailing_stop", "holding_days": held_days}
            if target_price is not None and float(high) >= float(target_price):
                return {"return": (float(target_price) / float(entry_price)) - 1.0, "exit_reason": "profit_target", "holding_days": held_days}
            if held_days >= int(exit_rules.time_limit_days):
                return {"return": (float(close) / float(entry_price)) - 1.0, "exit_reason": "time_limit", "holding_days": held_days}
        if last_close is None:
            return {"return": None, "exit_reason": None, "holding_days": None}
        return {"return": (float(last_close) / float(entry_price)) - 1.0, "exit_reason": "time_limit", "holding_days": held_days}

    def _path_benchmark_return(
        self,
        *,
        history_context: dict[str, dict[str, object]],
        snapshot_date: str,
        horizon: int,
        benchmark_ticker: str | None,
    ) -> float | None:
        if benchmark_ticker in (None, ""):
            return None
        benchmark_context = history_context.get(str(benchmark_ticker))
        if benchmark_context is None:
            return None
        benchmark_index = benchmark_context["index_by_date"].get(snapshot_date)
        if benchmark_index is None:
            return None
        return self._forward_return(
            ticker_frame=benchmark_context["frame"],
            index=int(benchmark_index),
            horizon=horizon,
        )

    def _optional_float(self, value) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
