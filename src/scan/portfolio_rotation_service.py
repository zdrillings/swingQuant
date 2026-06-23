from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd

from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


@dataclass(frozen=True)
class PortfolioRotationReport:
    output_path: str
    scan_dates: int
    trades: int
    final_equity: float
    total_return: float
    benchmark_return: float | None
    max_drawdown: float


@dataclass
class RotationPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_cash_outlay: float
    entry_fee: float
    sector: str
    model_alpha: float
    pre_opportunity: float
    entry_rank: int


class PortfolioRotationService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("portfolio_rotation")

    def run(
        self,
        *,
        target_positions: int = 6,
        max_hold_days: int = 20,
        min_pre_opportunity: float = 0.40,
        min_model_alpha: float = 0.0,
        initial_equity: float = 1.0,
        walk_forward: bool = False,
        horizon_days: int = 20,
        generated_at: str | None = None,
        model_name: str | None = None,
        eligible_universe_mode: str | None = None,
        model_scope: str | None = None,
        transaction_cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
        cooldown_days: int = 0,
        reinvest_gains: bool = True,
        max_new_entries_per_scan: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> PortfolioRotationReport:
        self.db_manager.initialize()
        transaction_cost_rate = max(float(transaction_cost_bps), 0.0) / 10_000.0
        slippage_rate = max(float(slippage_bps), 0.0) / 10_000.0
        candidates = self.db_manager.load_scan_candidates()
        if candidates.empty:
            raise ValueError("No scan snapshots found. Run `sq scan` or `sq scan-backfill` first.")
        walk_forward_context: dict[str, object] | None = None
        if walk_forward:
            candidates, walk_forward_context = self._apply_walk_forward_predictions(
                candidates,
                horizon_days=int(horizon_days),
                generated_at=generated_at,
                model_name=model_name,
                eligible_universe_mode=eligible_universe_mode,
                model_scope=model_scope,
            )
        candidates = self._prepare_candidates(
            candidates,
            min_pre_opportunity=float(min_pre_opportunity),
            min_model_alpha=float(min_model_alpha),
        )
        candidates = self._filter_date_window(candidates, date_from=date_from, date_to=date_to)
        if candidates.empty:
            raise ValueError("No candidates matched the portfolio rotation target filter.")

        tickers = sorted(set(candidates["ticker"].astype(str)).union({"SPY"}))
        history = self.db_manager.load_price_history(tickers)
        if history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")
        price_maps = self._build_price_maps(history)
        self._supplement_scan_prices(price_maps, candidates)

        scan_dates = sorted(pd.to_datetime(candidates["scan_date"]).dt.normalize().drop_duplicates().tolist())
        scan_date_set = {pd.Timestamp(date) for date in scan_dates}
        timeline = self._build_timeline(price_maps, start=scan_dates[0], end=scan_dates[-1])
        cash = float(initial_equity)
        fixed_slot_budget = float(initial_equity) / float(target_positions) if int(target_positions) > 0 else float(initial_equity)
        positions: list[RotationPosition] = []
        last_exit_dates: dict[str, pd.Timestamp] = {}
        equity_curve: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []

        for scan_date in timeline:
            current_prices = self._prices_for_date(price_maps, scan_date)
            if not current_prices:
                continue
            equity = cash + sum(
                self._mark_position_value(
                    position,
                    current_prices=current_prices,
                    transaction_cost_rate=transaction_cost_rate,
                    slippage_rate=slippage_rate,
                )
                for position in positions
            )
            positions, exited_cash, exits = self._exit_positions(
                positions=positions,
                scan_date=scan_date,
                current_prices=current_prices,
                max_hold_days=int(max_hold_days),
                transaction_cost_rate=transaction_cost_rate,
                slippage_rate=slippage_rate,
            )
            cash += exited_cash
            trade_rows.extend(exits)
            for exit_row in exits:
                last_exit_dates[str(exit_row["ticker"])] = scan_date
            exited_today = {str(row["ticker"]) for row in exits}
            equity = cash + sum(
                self._mark_position_value(
                    position,
                    current_prices=current_prices,
                    transaction_cost_rate=transaction_cost_rate,
                    slippage_rate=slippage_rate,
                )
                for position in positions
            )

            empty_slots = max(0, int(target_positions) - len(positions))
            if empty_slots > 0 and cash > 0:
                if scan_date not in scan_date_set:
                    equity_curve.append(
                        {
                            "scan_date": scan_date,
                            "equity": equity,
                            "cash": cash,
                            "positions": len(positions),
                            "holdings": ", ".join(position.ticker for position in positions),
                        }
                    )
                    continue
                day_candidates = candidates[pd.to_datetime(candidates["scan_date"]).dt.normalize() == scan_date].copy()
                held = {position.ticker for position in positions}
                cooldown_blocked = self._cooldown_blocked_tickers(
                    last_exit_dates=last_exit_dates,
                    scan_date=scan_date,
                    cooldown_days=int(cooldown_days),
                )
                blocked = held.union(exited_today).union(cooldown_blocked)
                ranked = day_candidates[~day_candidates["ticker"].astype(str).isin(blocked)].sort_values(
                    ["model_predicted_alpha", "pre_penalty_opportunity_score", "ticker"],
                    ascending=[False, False, True],
                )
                budget_per_slot = equity / float(target_positions) if bool(reinvest_gains) else fixed_slot_budget
                opened_today = 0
                for entry_rank, row in enumerate(ranked.itertuples(index=False), start=1):
                    if len(positions) >= int(target_positions) or cash <= 0:
                        break
                    if max_new_entries_per_scan is not None and opened_today >= int(max_new_entries_per_scan):
                        break
                    ticker = str(row.ticker)
                    price = current_prices.get(ticker)
                    if price is None or price <= 0:
                        continue
                    allocation = min(float(budget_per_slot), cash)
                    if allocation <= 0:
                        continue
                    execution_price = price * (1.0 + slippage_rate)
                    gross_purchase = allocation / (1.0 + transaction_cost_rate)
                    entry_fee = allocation - gross_purchase
                    shares = gross_purchase / execution_price
                    cash -= allocation
                    positions.append(
                        RotationPosition(
                            ticker=ticker,
                            entry_date=scan_date,
                            entry_price=execution_price,
                            shares=shares,
                            entry_cash_outlay=allocation,
                            entry_fee=entry_fee,
                            sector=str(getattr(row, "sector", "")),
                            model_alpha=float(row.model_predicted_alpha),
                            pre_opportunity=float(row.pre_penalty_opportunity_score),
                            entry_rank=int(entry_rank),
                        )
                    )
                    opened_today += 1

            equity = cash + sum(
                self._mark_position_value(
                    position,
                    current_prices=current_prices,
                    transaction_cost_rate=transaction_cost_rate,
                    slippage_rate=slippage_rate,
                )
                for position in positions
            )
            equity_curve.append(
                {
                    "scan_date": scan_date,
                    "equity": equity,
                    "cash": cash,
                    "positions": len(positions),
                    "holdings": ", ".join(position.ticker for position in positions),
                }
            )

        if equity_curve:
            final_prices = self._prices_for_date(price_maps, pd.Timestamp(equity_curve[-1]["scan_date"]))
            final_date = pd.Timestamp(equity_curve[-1]["scan_date"])
        else:
            final_prices = {}
            final_date = timeline[-1] if timeline else scan_dates[-1]
        for position in positions:
            price = final_prices.get(position.ticker, position.entry_price)
            trade_rows.append(
                self._trade_row(
                    position,
                    exit_date=final_date,
                    raw_exit_price=price,
                    reason="open_at_end",
                    transaction_cost_rate=transaction_cost_rate,
                    slippage_rate=slippage_rate,
                )
            )

        curve = pd.DataFrame(equity_curve)
        final_equity = float(curve["equity"].iloc[-1]) if not curve.empty else float(initial_equity)
        total_return = (final_equity / float(initial_equity)) - 1.0 if initial_equity else 0.0
        benchmark_return = self._benchmark_return(price_maps, curve)
        max_drawdown = self._max_drawdown(curve["equity"]) if not curve.empty else 0.0
        trades = pd.DataFrame(trade_rows)
        rolling_windows = self._rolling_window_returns(curve, windows=(20, 60))
        quarterly_returns = self._calendar_period_returns(curve, period="Q")

        friction_enabled = any(
            [
                float(transaction_cost_bps) > 0.0,
                float(slippage_bps) > 0.0,
                int(cooldown_days) > 0,
            ]
        )
        output_stem = "portfolio_rotation_walk_forward" if walk_forward_context is not None else "portfolio_rotation"
        if friction_enabled:
            output_stem = f"{output_stem}_friction"
        capital_constraints_enabled = (not bool(reinvest_gains)) or max_new_entries_per_scan is not None
        if capital_constraints_enabled:
            output_stem = f"{output_stem}_capital"
        report_path = self.db_manager.paths.reports_dir / f"{output_stem}.md"
        curve_path = self.db_manager.paths.reports_dir / f"{output_stem}_equity.csv"
        trades_path = self.db_manager.paths.reports_dir / f"{output_stem}_trades.csv"
        rolling_path = self.db_manager.paths.reports_dir / f"{output_stem}_rolling_windows.csv"
        quarterly_path = self.db_manager.paths.reports_dir / f"{output_stem}_quarterly_returns.csv"
        curve.to_csv(curve_path, index=False)
        trades.to_csv(trades_path, index=False)
        rolling_windows.to_csv(rolling_path, index=False)
        quarterly_returns.to_csv(quarterly_path, index=False)
        report_path.write_text(
            self._render_report(
                curve=curve,
                trades=trades,
                rolling_windows=rolling_windows,
                quarterly_returns=quarterly_returns,
                curve_path=curve_path,
                trades_path=trades_path,
                rolling_path=rolling_path,
                quarterly_path=quarterly_path,
                target_positions=int(target_positions),
                max_hold_days=int(max_hold_days),
                min_pre_opportunity=float(min_pre_opportunity),
                min_model_alpha=float(min_model_alpha),
                walk_forward_context=walk_forward_context,
                transaction_cost_bps=float(transaction_cost_bps),
                slippage_bps=float(slippage_bps),
                cooldown_days=int(cooldown_days),
                reinvest_gains=bool(reinvest_gains),
                max_new_entries_per_scan=max_new_entries_per_scan,
                date_from=date_from,
                date_to=date_to,
                eligible_scan_dates=len(scan_dates),
                initial_equity=float(initial_equity),
                final_equity=final_equity,
                total_return=total_return,
                benchmark_return=benchmark_return,
                max_drawdown=max_drawdown,
            ),
            encoding="utf-8",
        )
        return PortfolioRotationReport(
            output_path=str(report_path),
            scan_dates=len([date for date in scan_dates if date in set(curve["scan_date"])]) if not curve.empty else 0,
            trades=len(trades.index),
            final_equity=final_equity,
            total_return=total_return,
            benchmark_return=benchmark_return,
            max_drawdown=max_drawdown,
        )

    def _apply_walk_forward_predictions(
        self,
        candidates: pd.DataFrame,
        *,
        horizon_days: int,
        generated_at: str | None,
        model_name: str | None,
        eligible_universe_mode: str | None,
        model_scope: str | None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if not hasattr(self.db_manager, "load_shortlist_model_predictions"):
            raise ValueError("Database manager does not support shortlist model predictions.")
        run = self._resolve_shortlist_model_run(
            horizon_days=int(horizon_days),
            generated_at=generated_at,
            model_name=model_name,
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
        )
        resolved_generated_at = str(run.get("generated_at"))
        resolved_model_name = str(model_name or run.get("champion_model"))
        resolved_eligible_universe_mode = eligible_universe_mode or run.get("eligible_universe_mode")
        resolved_model_scope = model_scope or run.get("model_scope")
        predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=resolved_generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=resolved_eligible_universe_mode,
            model_scope=resolved_model_scope,
            dataset_split="oos",
            model_name=resolved_model_name,
        )
        if predictions.empty:
            raise ValueError(
                "No walk-forward out-of-sample shortlist predictions found for "
                f"generated_at={resolved_generated_at}, model_name={resolved_model_name}."
            )
        working = candidates.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        working["ticker"] = working["ticker"].astype(str)
        predictions = predictions.copy()
        predictions["scan_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.normalize()
        predictions["ticker"] = predictions["ticker"].astype(str)
        prediction_columns = ["scan_date", "ticker", "predicted_alpha", "model_name", "generated_at"]
        merged = working.drop(
            columns=["model_predicted_alpha", "model_name", "model_generated_at", "selection_source"],
            errors="ignore",
        ).merge(
            predictions[prediction_columns],
            on=["scan_date", "ticker"],
            how="inner",
        )
        if merged.empty:
            first_date = pd.to_datetime(predictions["scan_date"]).min().date()
            last_date = pd.to_datetime(predictions["scan_date"]).max().date()
            raise ValueError(
                "No scan candidates matched walk-forward prediction dates/tickers "
                f"({first_date} to {last_date})."
            )
        merged = merged.rename(columns={"predicted_alpha": "model_predicted_alpha"})
        merged["selection_source"] = "shortlist_model_oos"
        merged["model_generated_at"] = merged["generated_at"]
        merged = merged.drop(columns=["generated_at"], errors="ignore")
        return merged, {
            "mode": "walk_forward_oos",
            "generated_at": resolved_generated_at,
            "model_name": resolved_model_name,
            "eligible_universe_mode": resolved_eligible_universe_mode,
            "model_scope": resolved_model_scope,
            "horizon_days": int(horizon_days),
            "prediction_dates": int(predictions["scan_date"].nunique()),
            "prediction_start": str(pd.to_datetime(predictions["scan_date"]).min().date()),
            "prediction_end": str(pd.to_datetime(predictions["scan_date"]).max().date()),
        }

    def _resolve_shortlist_model_run(
        self,
        *,
        horizon_days: int,
        generated_at: str | None,
        model_name: str | None,
        eligible_universe_mode: str | None,
        model_scope: str | None,
    ) -> dict[str, object]:
        if not hasattr(self.db_manager, "load_shortlist_model_runs"):
            raise ValueError("Database manager does not support shortlist model runs.")
        runs = self.db_manager.load_shortlist_model_runs(
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            limit=50,
        )
        if runs.empty:
            raise ValueError("No shortlist model runs found. Run `sq shortlist-model` first.")
        if generated_at is not None:
            runs = runs[runs["generated_at"].astype(str) == str(generated_at)].copy()
        if model_name is not None and "champion_model" in runs.columns:
            runs = runs[runs["champion_model"].astype(str) == str(model_name)].copy()
        if runs.empty:
            raise ValueError("No shortlist model run matched the requested walk-forward filters.")
        return runs.sort_values("generated_at", ascending=False).iloc[0].to_dict()

    def _prepare_candidates(
        self,
        frame: pd.DataFrame,
        *,
        min_pre_opportunity: float,
        min_model_alpha: float,
    ) -> pd.DataFrame:
        working = frame.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        if "model_predicted_alpha" not in working.columns:
            working["model_predicted_alpha"] = working.get("selection_score")
        working["model_predicted_alpha"] = pd.to_numeric(working["model_predicted_alpha"], errors="coerce")
        if "overlap_penalty" not in working.columns:
            working["overlap_penalty"] = 0.0
        working["pre_penalty_opportunity_score"] = (
            pd.to_numeric(working["opportunity_score"], errors="coerce")
            + pd.to_numeric(working["overlap_penalty"], errors="coerce").fillna(0.0)
        )
        if "details_json" in working.columns:
            details_pre = working["details_json"].apply(self._pre_opportunity_from_details)
            working["pre_penalty_opportunity_score"] = details_pre.combine_first(working["pre_penalty_opportunity_score"])
        filtered = working[
            (working["pre_penalty_opportunity_score"] >= float(min_pre_opportunity))
            & (working["model_predicted_alpha"] > float(min_model_alpha))
        ].copy()
        return filtered.reset_index(drop=True)

    def _filter_date_window(self, frame: pd.DataFrame, *, date_from: str | None, date_to: str | None) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        working = frame.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        if date_from not in (None, ""):
            working = working[working["scan_date"] >= pd.Timestamp(date_from).normalize()].copy()
        if date_to not in (None, ""):
            working = working[working["scan_date"] <= pd.Timestamp(date_to).normalize()].copy()
        return working.reset_index(drop=True)

    def _pre_opportunity_from_details(self, value) -> float | None:
        if value is None or value == "" or pd.isna(value):
            return None
        try:
            details = json.loads(str(value))
        except Exception:
            return None
        value = details.get("pre_penalty_opportunity_score") if isinstance(details, dict) else None
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _build_price_maps(self, history: pd.DataFrame) -> dict[str, dict[pd.Timestamp, float]]:
        working = history.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
        maps: dict[str, dict[pd.Timestamp, float]] = {}
        for ticker, group in working.dropna(subset=["adj_close"]).groupby("ticker", sort=False):
            maps[str(ticker)] = {
                pd.Timestamp(row.date).normalize(): float(row.adj_close)
                for row in group.itertuples(index=False)
            }
        return maps

    def _supplement_scan_prices(self, price_maps: dict[str, dict[pd.Timestamp, float]], candidates: pd.DataFrame) -> None:
        if candidates.empty or "adj_close" not in candidates.columns:
            return
        working = candidates[["scan_date", "ticker", "adj_close"]].copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"], errors="coerce").dt.normalize()
        working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
        for row in working.dropna(subset=["scan_date", "adj_close"]).itertuples(index=False):
            ticker = str(row.ticker)
            price_maps.setdefault(ticker, {})[pd.Timestamp(row.scan_date)] = float(row.adj_close)

    def _build_timeline(
        self,
        price_maps: dict[str, dict[pd.Timestamp, float]],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        start_key = pd.Timestamp(start).normalize()
        end_key = pd.Timestamp(end).normalize()
        dates = {
            pd.Timestamp(date).normalize()
            for prices in price_maps.values()
            for date in prices.keys()
            if start_key <= pd.Timestamp(date).normalize() <= end_key
        }
        return sorted(dates)

    def _benchmark_return(self, price_maps: dict[str, dict[pd.Timestamp, float]], curve: pd.DataFrame) -> float | None:
        if curve.empty or "SPY" not in price_maps:
            return None
        dates = pd.to_datetime(curve["scan_date"]).dt.normalize().tolist()
        start_price = next((price_maps["SPY"].get(pd.Timestamp(date)) for date in dates if price_maps["SPY"].get(pd.Timestamp(date))), None)
        end_price = next(
            (price_maps["SPY"].get(pd.Timestamp(date)) for date in reversed(dates) if price_maps["SPY"].get(pd.Timestamp(date))),
            None,
        )
        if not start_price or not end_price:
            return None
        return (float(end_price) / float(start_price)) - 1.0

    def _prices_for_date(self, price_maps: dict[str, dict[pd.Timestamp, float]], scan_date: pd.Timestamp) -> dict[str, float]:
        date_key = pd.Timestamp(scan_date).normalize()
        return {
            ticker: prices[date_key]
            for ticker, prices in price_maps.items()
            if date_key in prices
        }

    def _mark_position_value(
        self,
        position: RotationPosition,
        *,
        current_prices: dict[str, float],
        transaction_cost_rate: float,
        slippage_rate: float,
    ) -> float:
        raw_price = current_prices.get(position.ticker, position.entry_price)
        execution_price = float(raw_price) * (1.0 - float(slippage_rate))
        gross_value = position.shares * execution_price
        return gross_value * (1.0 - float(transaction_cost_rate))

    def _cooldown_blocked_tickers(
        self,
        *,
        last_exit_dates: dict[str, pd.Timestamp],
        scan_date: pd.Timestamp,
        cooldown_days: int,
    ) -> set[str]:
        if int(cooldown_days) <= 0:
            return set()
        blocked: set[str] = set()
        for ticker, exit_date in last_exit_dates.items():
            elapsed = int(len(pd.bdate_range(pd.Timestamp(exit_date), pd.Timestamp(scan_date))) - 1)
            if elapsed <= int(cooldown_days):
                blocked.add(str(ticker))
        return blocked

    def _exit_positions(
        self,
        *,
        positions: list[RotationPosition],
        scan_date: pd.Timestamp,
        current_prices: dict[str, float],
        max_hold_days: int,
        transaction_cost_rate: float,
        slippage_rate: float,
    ) -> tuple[list[RotationPosition], float, list[dict[str, object]]]:
        remaining: list[RotationPosition] = []
        cash = 0.0
        trades: list[dict[str, object]] = []
        for position in positions:
            held_days = int(len(pd.bdate_range(position.entry_date, scan_date)) - 1)
            price = current_prices.get(position.ticker)
            if price is not None and held_days >= int(max_hold_days):
                trade = self._trade_row(
                    position,
                    exit_date=scan_date,
                    raw_exit_price=price,
                    reason="max_hold_days",
                    transaction_cost_rate=transaction_cost_rate,
                    slippage_rate=slippage_rate,
                )
                cash += float(trade["exit_net_proceeds"])
                trades.append(trade)
            else:
                remaining.append(position)
        return remaining, cash, trades

    def _trade_row(
        self,
        position: RotationPosition,
        *,
        exit_date: pd.Timestamp,
        raw_exit_price: float,
        reason: str,
        transaction_cost_rate: float,
        slippage_rate: float,
    ) -> dict[str, object]:
        exit_price = float(raw_exit_price) * (1.0 - float(slippage_rate))
        gross_proceeds = position.shares * exit_price
        exit_fee = gross_proceeds * float(transaction_cost_rate)
        net_proceeds = gross_proceeds - exit_fee
        return {
            "ticker": position.ticker,
            "sector": position.sector,
            "entry_date": position.entry_date,
            "exit_date": exit_date,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_cash_outlay": position.entry_cash_outlay,
            "exit_net_proceeds": net_proceeds,
            "entry_fee": position.entry_fee,
            "exit_fee": exit_fee,
            "return": (float(net_proceeds) / float(position.entry_cash_outlay)) - 1.0 if position.entry_cash_outlay else math.nan,
            "holding_days": int(len(pd.bdate_range(position.entry_date, exit_date)) - 1),
            "model_alpha": position.model_alpha,
            "pre_opportunity": position.pre_opportunity,
            "entry_rank": position.entry_rank,
            "exit_reason": reason,
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        values = pd.to_numeric(equity, errors="coerce").dropna()
        if values.empty:
            return 0.0
        running_max = values.cummax()
        drawdown = (values / running_max) - 1.0
        return abs(float(drawdown.min()))

    def _rolling_window_returns(self, curve: pd.DataFrame, *, windows: tuple[int, ...]) -> pd.DataFrame:
        if curve.empty:
            return pd.DataFrame(columns=["window_days", "start_date", "end_date", "return"])
        working = curve.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        working["equity"] = pd.to_numeric(working["equity"], errors="coerce")
        working = working.dropna(subset=["scan_date", "equity"]).sort_values("scan_date").reset_index(drop=True)
        rows: list[dict[str, object]] = []
        for window in windows:
            step_count = int(window)
            if step_count <= 0 or len(working.index) <= step_count:
                continue
            for end_index in range(step_count, len(working.index)):
                start_index = end_index - step_count
                start_equity = float(working.loc[start_index, "equity"])
                end_equity = float(working.loc[end_index, "equity"])
                if start_equity <= 0.0:
                    continue
                rows.append(
                    {
                        "window_days": step_count,
                        "start_date": pd.Timestamp(working.loc[start_index, "scan_date"]).date().isoformat(),
                        "end_date": pd.Timestamp(working.loc[end_index, "scan_date"]).date().isoformat(),
                        "return": (end_equity / start_equity) - 1.0,
                    }
                )
        return pd.DataFrame(rows, columns=["window_days", "start_date", "end_date", "return"])

    def _calendar_period_returns(self, curve: pd.DataFrame, *, period: str = "Q") -> pd.DataFrame:
        if curve.empty:
            return pd.DataFrame(columns=["period", "start_date", "end_date", "return"])
        working = curve.copy()
        working["scan_date"] = pd.to_datetime(working["scan_date"]).dt.normalize()
        working["equity"] = pd.to_numeric(working["equity"], errors="coerce")
        working = working.dropna(subset=["scan_date", "equity"]).sort_values("scan_date").reset_index(drop=True)
        working["period"] = working["scan_date"].dt.to_period(period).astype(str)
        rows: list[dict[str, object]] = []
        for period_label, group in working.groupby("period", sort=True):
            if group.empty:
                continue
            start_equity = float(group.iloc[0]["equity"])
            end_equity = float(group.iloc[-1]["equity"])
            if start_equity <= 0.0:
                continue
            rows.append(
                {
                    "period": str(period_label),
                    "start_date": pd.Timestamp(group.iloc[0]["scan_date"]).date().isoformat(),
                    "end_date": pd.Timestamp(group.iloc[-1]["scan_date"]).date().isoformat(),
                    "return": (end_equity / start_equity) - 1.0,
                }
            )
        return pd.DataFrame(rows, columns=["period", "start_date", "end_date", "return"])

    def _return_summary(self, returns: pd.Series) -> dict[str, float | int | str]:
        series = pd.to_numeric(returns, errors="coerce").dropna()
        if series.empty:
            return {
                "windows": 0,
                "mean": math.nan,
                "median": math.nan,
                "p25": math.nan,
                "p75": math.nan,
                "p05": math.nan,
                "p95": math.nan,
                "hit_rate": math.nan,
                "worst": math.nan,
                "best": math.nan,
            }
        return {
            "windows": int(len(series.index)),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p05": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
            "hit_rate": float(series.gt(0.0).mean()),
            "worst": float(series.min()),
            "best": float(series.max()),
        }

    def _render_rolling_scorecard(self, rolling_windows: pd.DataFrame) -> list[str]:
        lines = ["### Rolling Portfolio Windows"]
        if rolling_windows.empty:
            return lines + ["- observations: 0", ""]
        working = rolling_windows.copy()
        working["return"] = pd.to_numeric(working["return"], errors="coerce")
        for window in sorted(working["window_days"].dropna().astype(int).unique()):
            scoped = working[working["window_days"].astype(int) == int(window)].copy()
            summary = self._return_summary(scoped["return"])
            if int(summary["windows"]) <= 0:
                lines.append(f"- {window}d: observations=0")
                continue
            worst_row = scoped.sort_values("return", ascending=True).iloc[0]
            lines.append(
                f"- {window}d: windows={int(summary['windows'])}, "
                f"mean={self._format_pct(summary['mean'])}, median={self._format_pct(summary['median'])}, "
                f"p25_p75={self._format_pct(summary['p25'])} to {self._format_pct(summary['p75'])}, "
                f"p05_p95={self._format_pct(summary['p05'])} to {self._format_pct(summary['p95'])}, "
                f"hit_rate={self._format_pct(summary['hit_rate'])}, "
                f"worst={self._format_pct(summary['worst'])} ({worst_row['start_date']} to {worst_row['end_date']}), "
                f"best={self._format_pct(summary['best'])}"
            )
        lines.append("")
        return lines

    def _render_quarterly_scorecard(self, quarterly_returns: pd.DataFrame) -> list[str]:
        lines = ["### Calendar Quarter Returns"]
        if quarterly_returns.empty:
            return lines + ["- observations: 0", ""]
        working = quarterly_returns.copy()
        working["return"] = pd.to_numeric(working["return"], errors="coerce")
        summary = self._return_summary(working["return"])
        if int(summary["windows"]) <= 0:
            return lines + ["- observations: 0", ""]
        worst_row = working.sort_values("return", ascending=True).iloc[0]
        best_row = working.sort_values("return", ascending=False).iloc[0]
        lines.append(
            f"- quarters={int(summary['windows'])}, "
            f"mean={self._format_pct(summary['mean'])}, median={self._format_pct(summary['median'])}, "
            f"p25_p75={self._format_pct(summary['p25'])} to {self._format_pct(summary['p75'])}, "
            f"hit_rate={self._format_pct(summary['hit_rate'])}, "
            f"worst={self._format_pct(summary['worst'])} ({worst_row['period']}), "
            f"best={self._format_pct(summary['best'])} ({best_row['period']})"
        )
        lines.append("")
        return lines

    def _format_pct(self, value) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(numeric):
            return "n/a"
        return f"{numeric:.2%}"

    def _render_report(
        self,
        *,
        curve: pd.DataFrame,
        trades: pd.DataFrame,
        rolling_windows: pd.DataFrame,
        quarterly_returns: pd.DataFrame,
        curve_path,
        trades_path,
        rolling_path,
        quarterly_path,
        target_positions: int,
        max_hold_days: int,
        min_pre_opportunity: float,
        min_model_alpha: float,
        walk_forward_context: dict[str, object] | None,
        transaction_cost_bps: float,
        slippage_bps: float,
        cooldown_days: int,
        reinvest_gains: bool,
        max_new_entries_per_scan: int | None,
        date_from: str | None,
        date_to: str | None,
        eligible_scan_dates: int,
        initial_equity: float,
        final_equity: float,
        total_return: float,
        benchmark_return: float | None,
        max_drawdown: float,
    ) -> str:
        lines = [
            "# Portfolio Rotation Backtest",
            "",
            "## Rules",
            f"- target_positions: {target_positions}",
            f"- max_hold_days: {max_hold_days}",
            f"- min_pre_penalty_opportunity: {min_pre_opportunity:.2f}",
            f"- min_model_alpha: {min_model_alpha:.4f}",
            f"- transaction_cost_bps: {transaction_cost_bps:.2f}",
            f"- slippage_bps: {slippage_bps:.2f}",
            f"- cooldown_days: {cooldown_days}",
            f"- reinvest_gains: {str(bool(reinvest_gains)).lower()}",
            f"- max_new_entries_per_scan: {max_new_entries_per_scan if max_new_entries_per_scan is not None else 'unlimited'}",
            f"- date_from: {date_from or 'all'}",
            f"- date_to: {date_to or 'all'}",
            "- sell_rule: max_hold_days only in v1",
            "- replacement_rule: fill empty slots from best unheld targets; no churn of existing holdings",
        ]
        if walk_forward_context is not None:
            lines.extend(
                [
                    f"- ranking_mode: {walk_forward_context['mode']}",
                    f"- ranking_model: {walk_forward_context['model_name']}",
                    f"- ranking_generated_at: {walk_forward_context['generated_at']}",
                    f"- ranking_prediction_dates: {walk_forward_context['prediction_start']} to {walk_forward_context['prediction_end']}",
                ]
            )
        else:
            lines.append("- ranking_mode: retrospective_persisted_scan_scores")
        caveat = (
            "- Walk-forward mode uses persisted out-of-sample shortlist predictions, but still depends on the current stored scan snapshots and v1 portfolio mechanics."
            if walk_forward_context is not None
            else "- This is not a walk-forward live simulation; it applies persisted scan model scores across historical scan snapshots."
        )
        exit_caveat = (
            "- The v1 exit model only uses max holding days, so it does not yet reflect monitor sell signals, stops, RSI exits, or earnings exits."
            if (float(transaction_cost_bps) > 0.0 or float(slippage_bps) > 0.0)
            else "- The v1 exit model only uses max holding days, so it does not yet reflect monitor sell signals, stops, RSI exits, earnings exits, or transaction costs."
        )
        lines.extend(
            [
                "",
                "## Caveats",
                caveat,
                exit_caveat,
                "- Total return is fully compounded through six equal-weight slots; high 20-day trade returns compound quickly.",
                "",
                "## Primary Scorecard",
                "",
            ]
        )
        lines.extend(self._render_rolling_scorecard(rolling_windows))
        lines.extend(self._render_quarterly_scorecard(quarterly_returns))
        lines.extend(
            [
                "",
                "## Summary",
                f"- initial_equity: {initial_equity:.4f}",
                f"- final_equity: {final_equity:.4f}",
                f"- total_return: {total_return:.2%}",
                f"- spy_return: {benchmark_return:.2%}" if benchmark_return is not None else "- spy_return: n/a",
                f"- max_drawdown: {max_drawdown:.2%}",
                f"- eligible_scan_dates: {eligible_scan_dates}",
                f"- portfolio_valuation_dates: {len(curve.index)}",
                f"- completed_or_open_trades: {len(trades.index)}",
                f"- equity_csv: {curve_path}",
                f"- trades_csv: {trades_path}",
                f"- rolling_windows_csv: {rolling_path}",
                f"- quarterly_returns_csv: {quarterly_path}",
            ]
        )
        if not curve.empty:
            lines.extend(
                [
                    f"- avg_positions: {float(curve['positions'].mean()):.2f}",
                    f"- pct_days_fully_invested: {(curve['positions'].eq(target_positions).mean()):.2%}",
                ]
            )
        if not trades.empty:
            closed = trades[trades["exit_reason"] != "open_at_end"].copy()
            lines.extend(
                [
                    f"- closed_trades: {len(closed.index)}",
                    f"- trade_win_rate: {(closed['return'].gt(0).mean() if not closed.empty else 0.0):.2%}",
                    f"- mean_trade_return: {(closed['return'].mean() if not closed.empty else 0.0):.2%}",
                    f"- median_trade_return: {(closed['return'].median() if not closed.empty else 0.0):.2%}",
                    f"- avg_holding_days: {(closed['holding_days'].mean() if not closed.empty else 0.0):.1f}",
                ]
            )
            if not closed.empty:
                lines.extend(["", "## Closed Trade Returns by Entry Month"])
                monthly = closed.copy()
                monthly["entry_month"] = pd.to_datetime(monthly["entry_date"]).dt.to_period("M").astype(str)
                grouped = monthly.groupby("entry_month", sort=True)["return"].agg(["count", "mean", "median"])
                win_rate = monthly.groupby("entry_month", sort=True)["return"].apply(lambda series: series.gt(0).mean())
                for month, row in grouped.iterrows():
                    lines.append(
                        f"- {month}: trades={int(row['count'])}, mean={float(row['mean']):+.2%}, "
                        f"median={float(row['median']):+.2%}, win_rate={float(win_rate.loc[month]):.2%}"
                    )

                lines.extend(["", "## Best Closed Trades"])
                for row in closed.sort_values("return", ascending=False).head(10).to_dict("records"):
                    lines.append(
                        f"- {row['ticker']}: {pd.Timestamp(row['entry_date']).date()} -> {pd.Timestamp(row['exit_date']).date()} "
                        f"return={float(row['return']):+.2%}, model_alpha={float(row['model_alpha']):+.2%}, "
                        f"pre_opp={float(row['pre_opportunity']):.2f}"
                    )

                lines.extend(["", "## Worst Closed Trades"])
                for row in closed.sort_values("return", ascending=True).head(10).to_dict("records"):
                    lines.append(
                        f"- {row['ticker']}: {pd.Timestamp(row['entry_date']).date()} -> {pd.Timestamp(row['exit_date']).date()} "
                        f"return={float(row['return']):+.2%}, model_alpha={float(row['model_alpha']):+.2%}, "
                        f"pre_opp={float(row['pre_opportunity']):.2f}"
                    )
            lines.extend(["", "## Recent Trades"])
            for row in trades.tail(20).to_dict("records"):
                lines.append(
                    f"- {row['ticker']}: {pd.Timestamp(row['entry_date']).date()} -> {pd.Timestamp(row['exit_date']).date()} "
                    f"return={float(row['return']):+.2%} reason={row['exit_reason']}"
                )
        if not curve.empty:
            lines.extend(["", "## Recent Portfolio States"])
            for row in curve.tail(10).itertuples(index=False):
                lines.append(
                    f"- {pd.Timestamp(row.scan_date).date()}: equity={float(row.equity):.4f}, "
                    f"positions={int(row.positions)}, holdings={row.holdings}"
                )
        return "\n".join(lines) + "\n"
