from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import itertools
import json
import math
import operator

import pandas as pd

from src.settings import load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import BacktestResultRow, DatabaseManager
from src.utils.logging import get_logger
from src.utils.signal_engine import build_analysis_frame
from src.utils.strategy import DEFAULT_EXIT_RULES


@dataclass(frozen=True)
class SweepReport:
    combinations: int
    inserted_results: int


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + (step / 10):
        values.append(round(current, 10))
        current += step
    return values


class SweepService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("sweep")

    def run(self) -> SweepReport:
        try:
            import polars as pl
        except ModuleNotFoundError as exc:
            raise RuntimeError("polars is required for `sq sweep`. Install project dependencies first.") from exc

        self.db_manager.initialize()
        universe_rows = self.db_manager.list_research_universe(limit=250)
        if not universe_rows:
            raise ValueError("Universe is empty or liquidity metrics are unavailable. Run `sq sync` first.")

        universe_tickers = [row["ticker"] for row in universe_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")

        analysis_frame, feature_columns = build_analysis_frame(price_history, universe_rows)
        analysis_frame = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        if analysis_frame.empty:
            raise ValueError("No analysis frame could be built for the selected universe.")

        pl_frame = pl.from_dicts(analysis_frame.to_dict(orient="records"))
        unique_dates = sorted(pl_frame.select("date").unique().to_series().to_list())
        if len(unique_dates) < 2:
            raise ValueError("At least two distinct dates are required for sweep validation.")
        split_index = min(max(int(len(unique_dates) * 0.7), 1), len(unique_dates) - 1)
        validation_start = unique_dates[split_index]
        validation_frame = pl_frame.filter(pl.col("date") >= validation_start)

        grid = self._build_parameter_grid(load_feature_config())
        sectors = sorted({row["sector"] for row in universe_rows})
        next_strategy_id = self.db_manager.next_strategy_id()
        result_rows: list[BacktestResultRow] = []

        for params in grid:
            for sector in ["ALL", *sectors]:
                scoped_frame = validation_frame
                if sector != "ALL":
                    scoped_frame = validation_frame.filter(pl.col("sector") == sector)
                metrics = self._run_backtest(scoped_frame, params)
                params_json = json.dumps(
                    {
                        "indicators": params,
                        "exit_rules": {
                            "trailing_stop_pct": DEFAULT_EXIT_RULES.trailing_stop_pct,
                            "profit_target_pct": DEFAULT_EXIT_RULES.profit_target_pct,
                            "time_limit_days": DEFAULT_EXIT_RULES.time_limit_days,
                        },
                        "sector": sector,
                        "scope_size": 250,
                    },
                    sort_keys=True,
                )
                result_rows.append(
                    BacktestResultRow(
                        strategy_id=next_strategy_id,
                        params_json=params_json,
                        norm_score=None,
                        profit_factor=metrics["profit_factor"],
                        expectancy=metrics["expectancy"],
                        mdd=metrics["mdd"],
                        win_rate=metrics["win_rate"],
                    )
                )
                next_strategy_id += 1

        inserted = self.db_manager.insert_backtest_results(result_rows)
        return SweepReport(combinations=len(grid), inserted_results=inserted)

    def _build_parameter_grid(self, config: dict) -> list[dict[str, float]]:
        grid_values = []
        for name, definition in config.get("sweep_grid", {}).items():
            values = _frange(float(definition["min"]), float(definition["max"]), float(definition["step"]))
            grid_values.append((name, values))

        combinations = []
        for combination in itertools.product(*[values for _, values in grid_values]):
            combinations.append({name: value for (name, _), value in zip(grid_values, combination, strict=True)})
        return combinations

    def _run_backtest(self, validation_frame, params: dict[str, float]) -> dict[str, float]:
        import polars as pl

        if validation_frame.is_empty():
            return {"profit_factor": 0.0, "expectancy": 0.0, "mdd": 0.0, "win_rate": 0.0}

        conditions = []
        for name, value in params.items():
            feature_name = name[:-4]
            if name.endswith("_min"):
                conditions.append(pl.col(feature_name) >= value)
            elif name.endswith("_max"):
                conditions.append(pl.col(feature_name) <= value)
        signal_expr = reduce(operator.and_, conditions)

        prepared = (
            validation_frame.sort(["ticker", "date"])
            .with_columns(
                signal=signal_expr,
                next_open=pl.col("open").shift(-1).over("ticker"),
                next_date=pl.col("date").shift(-1).over("ticker"),
            )
        )

        trades: list[float] = []
        equity_curve = [1.0]

        for _, ticker_frame in prepared.partition_by("ticker", as_dict=True).items():
            position = None
            for row in ticker_frame.to_dicts():
                if position is not None:
                    position["max_price"] = max(position["max_price"], float(row["high"]))
                    stop_price = position["max_price"] * (1 - DEFAULT_EXIT_RULES.trailing_stop_pct)
                    profit_target_price = position["entry_price"] * (1 + DEFAULT_EXIT_RULES.profit_target_pct)
                    held_days = position["held_days"]
                    exit_price = None
                    if not bool(row["regime_green"]):
                        exit_price = float(row["close"])
                    elif float(row["low"]) <= stop_price:
                        exit_price = stop_price
                    elif float(row["high"]) >= profit_target_price:
                        exit_price = profit_target_price
                    elif float(row["rsi_2"]) > 90:
                        exit_price = float(row["close"])
                    elif held_days > DEFAULT_EXIT_RULES.time_limit_days:
                        exit_price = float(row["close"])

                    if exit_price is not None:
                        pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
                        trades.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None
                    else:
                        position["held_days"] += 1

                if position is None and bool(row["signal"]) and bool(row["regime_green"]) and row["next_open"] is not None:
                    position = {
                        "entry_price": float(row["next_open"]),
                        "max_price": float(row["next_open"]),
                        "held_days": 1,
                    }

        if not trades:
            return {"profit_factor": 0.0, "expectancy": 0.0, "mdd": 0.0, "win_rate": 0.0}

        profit_sum = sum(value for value in trades if value > 0)
        loss_sum = abs(sum(value for value in trades if value < 0))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else profit_sum if profit_sum > 0 else 0.0
        expectancy = sum(trades) / len(trades)
        win_rate = sum(1 for value in trades if value > 0) / len(trades)
        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - value) / peak)
        return {
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "mdd": float(max_drawdown),
            "win_rate": float(win_rate),
        }
