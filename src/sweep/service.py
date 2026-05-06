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
from src.utils.strategy import (
    DEFAULT_EXIT_RULES,
    ExitRules,
    SIGNAL_SCORE_MIN_KEY,
    load_signal_model_config,
    profit_target_price,
    split_signal_indicators,
    trailing_stop_price,
)


@dataclass(frozen=True)
class SweepReport:
    combinations: int
    inserted_results: int


@dataclass(frozen=True)
class BacktestCostModel:
    slippage_bps_per_side: float
    commission_bps_per_side: float

    @property
    def round_trip_cost_fraction(self) -> float:
        return ((self.slippage_bps_per_side + self.commission_bps_per_side) * 2.0) / 10_000.0


EXIT_RULE_GRID_KEYS = {
    "trailing_stop_pct",
    "profit_target_pct",
    "trailing_stop_atr_mult",
    "profit_target_atr_mult",
    "time_limit_days",
}


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + (step / 10):
        values.append(round(current, 10))
        current += step
    return values


def _optional_finite_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


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

        feature_config = load_feature_config()
        grid = self._build_parameter_grid(feature_config)
        backtest_costs = self._load_backtest_costs(feature_config)
        sectors = sorted({row["sector"] for row in universe_rows})
        run_id = self.db_manager.next_run_id()
        next_strategy_id = self.db_manager.next_strategy_id()
        result_rows: list[BacktestResultRow] = []
        total_runs = len(grid) * (len(sectors) + 1)
        completed_runs = 0

        self.logger.info(
            "Starting sweep: run_id=%s parameter_combinations=%s sectors=%s total_runs=%s",
            run_id,
            len(grid),
            len(sectors) + 1,
            total_runs,
        )
        self.logger.info(
            "Backtest costs: slippage_bps_per_side=%s commission_bps_per_side=%s round_trip_cost_pct=%.4f",
            backtest_costs.slippage_bps_per_side,
            backtest_costs.commission_bps_per_side,
            backtest_costs.round_trip_cost_fraction * 100.0,
        )

        for strategy_params in grid:
            for sector in ["ALL", *sectors]:
                scoped_frame = validation_frame
                if sector != "ALL":
                    scoped_frame = validation_frame.filter(pl.col("sector") == sector)
                metrics = self._run_backtest(
                    scoped_frame,
                    strategy_params["indicators"],
                    strategy_params["exit_rules"],
                    backtest_costs,
                )
                params_json = json.dumps(
                    {
                        "indicators": strategy_params["indicators"],
                        "exit_rules": {
                            "trailing_stop_pct": strategy_params["exit_rules"]["trailing_stop_pct"],
                            "profit_target_pct": strategy_params["exit_rules"]["profit_target_pct"],
                            "time_limit_days": strategy_params["exit_rules"]["time_limit_days"],
                            "trailing_stop_atr_mult": strategy_params["exit_rules"].get("trailing_stop_atr_mult"),
                            "profit_target_atr_mult": strategy_params["exit_rules"].get("profit_target_atr_mult"),
                        },
                        "backtest_costs": {
                            "slippage_bps_per_side": backtest_costs.slippage_bps_per_side,
                            "commission_bps_per_side": backtest_costs.commission_bps_per_side,
                        },
                        "sector": sector,
                        "scope_size": 250,
                    },
                    sort_keys=True,
                )
                result_rows.append(
                    BacktestResultRow(
                        run_id=run_id,
                        strategy_id=next_strategy_id,
                        params_json=params_json,
                        norm_score=None,
                        profit_factor=metrics["profit_factor"],
                        expectancy=metrics["expectancy"],
                        mdd=metrics["mdd"],
                        win_rate=metrics["win_rate"],
                        trade_count=metrics["trade_count"],
                    )
                )
                next_strategy_id += 1
                completed_runs += 1
                if completed_runs == total_runs or completed_runs % 25 == 0:
                    self.logger.info(
                        "Sweep progress: completed_runs=%s total_runs=%s current_sector=%s current_params=%s",
                        completed_runs,
                        total_runs,
                        sector,
                        strategy_params,
                    )

        inserted = self.db_manager.insert_backtest_results(result_rows)
        self.logger.info(
            "Sweep finished: run_id=%s inserted_results=%s parameter_combinations=%s total_runs=%s",
            run_id,
            inserted,
            len(grid),
            total_runs,
        )
        return SweepReport(combinations=len(grid), inserted_results=inserted)

    def _build_parameter_grid(self, config: dict) -> list[dict[str, dict[str, float | None]]]:
        grid_values = []
        for name, definition in config.get("sweep_grid", {}).items():
            values = _frange(float(definition["min"]), float(definition["max"]), float(definition["step"]))
            grid_values.append((name, values))

        combinations = []
        for combination in itertools.product(*[values for _, values in grid_values]):
            flat_params = {name: value for (name, _), value in zip(grid_values, combination, strict=True)}
            indicators = {
                name: value
                for name, value in flat_params.items()
                if name not in EXIT_RULE_GRID_KEYS
            }
            trailing_stop_pct = (
                float(flat_params["trailing_stop_pct"])
                if "trailing_stop_pct" in flat_params
                else DEFAULT_EXIT_RULES.trailing_stop_pct
            )
            profit_target_pct = (
                float(flat_params["profit_target_pct"])
                if "profit_target_pct" in flat_params
                else DEFAULT_EXIT_RULES.profit_target_pct
            )
            trailing_stop_atr_mult = (
                float(flat_params["trailing_stop_atr_mult"])
                if "trailing_stop_atr_mult" in flat_params
                else DEFAULT_EXIT_RULES.trailing_stop_atr_mult
            )
            profit_target_atr_mult = (
                float(flat_params["profit_target_atr_mult"])
                if "profit_target_atr_mult" in flat_params
                else DEFAULT_EXIT_RULES.profit_target_atr_mult
            )
            if trailing_stop_atr_mult is not None and "trailing_stop_pct" not in flat_params:
                trailing_stop_pct = None
            if profit_target_atr_mult is not None and "profit_target_pct" not in flat_params:
                profit_target_pct = None
            exit_rules = {
                "trailing_stop_pct": trailing_stop_pct,
                "profit_target_pct": profit_target_pct,
                "time_limit_days": int(flat_params.get("time_limit_days", DEFAULT_EXIT_RULES.time_limit_days)),
                "trailing_stop_atr_mult": trailing_stop_atr_mult,
                "profit_target_atr_mult": profit_target_atr_mult,
            }
            combinations.append({"indicators": indicators, "exit_rules": exit_rules})
        return combinations

    def _load_backtest_costs(self, config: dict) -> BacktestCostModel:
        cost_config = config.get("backtest_costs", {})
        return BacktestCostModel(
            slippage_bps_per_side=float(cost_config.get("slippage_bps_per_side", 0.0)),
            commission_bps_per_side=float(cost_config.get("commission_bps_per_side", 0.0)),
        )

    def _run_backtest(
        self,
        validation_frame,
        indicators: dict[str, float],
        exit_rules: dict[str, float],
        backtest_costs: BacktestCostModel,
    ) -> dict[str, float]:
        import polars as pl

        if validation_frame.is_empty():
            return {"profit_factor": 0.0, "expectancy": 0.0, "mdd": 0.0, "win_rate": 0.0, "trade_count": 0}

        signal_expr = self._build_signal_expression(pl, indicators)

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
        resolved_exit_rules = ExitRules(
            trailing_stop_pct=(
                float(exit_rules["trailing_stop_pct"])
                if exit_rules.get("trailing_stop_pct") is not None
                else None
            ),
            profit_target_pct=(
                float(exit_rules["profit_target_pct"])
                if exit_rules.get("profit_target_pct") is not None
                else None
            ),
            time_limit_days=int(exit_rules["time_limit_days"]),
            trailing_stop_atr_mult=(
                float(exit_rules["trailing_stop_atr_mult"])
                if exit_rules.get("trailing_stop_atr_mult") is not None
                else None
            ),
            profit_target_atr_mult=(
                float(exit_rules["profit_target_atr_mult"])
                if exit_rules.get("profit_target_atr_mult") is not None
                else None
            ),
        )

        for _, ticker_frame in prepared.partition_by("ticker", as_dict=True).items():
            position = None
            for row in ticker_frame.to_dicts():
                if position is not None:
                    position["max_price"] = max(position["max_price"], float(row["high"]))
                    stop_price = trailing_stop_price(
                        max_price_seen=position["max_price"],
                        entry_atr=position["entry_atr"],
                        exit_rules=resolved_exit_rules,
                    )
                    target_price = profit_target_price(
                        entry_price=position["entry_price"],
                        entry_atr=position["entry_atr"],
                        exit_rules=resolved_exit_rules,
                    )
                    held_days = position["held_days"]
                    exit_price = None
                    if not bool(row["regime_green"]):
                        exit_price = float(row["close"])
                    elif float(row["low"]) <= stop_price:
                        exit_price = stop_price
                    elif float(row["high"]) >= target_price:
                        exit_price = target_price
                    elif float(row["rsi_2"]) > 90:
                        exit_price = float(row["close"])
                    elif held_days > int(exit_rules["time_limit_days"]):
                        exit_price = float(row["close"])

                    if exit_price is not None:
                        pnl_pct = ((exit_price - position["entry_price"]) / position["entry_price"]) - backtest_costs.round_trip_cost_fraction
                        trades.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None
                    else:
                        position["held_days"] += 1

                if position is None and bool(row["signal"]) and bool(row["regime_green"]) and row["next_open"] is not None:
                    entry_atr = _optional_finite_float(row.get("atr_14"))
                    if resolved_exit_rules.trailing_stop_atr_mult is not None and entry_atr is None:
                        continue
                    if resolved_exit_rules.profit_target_atr_mult is not None and entry_atr is None:
                        continue
                    position = {
                        "entry_price": float(row["next_open"]),
                        "max_price": float(row["next_open"]),
                        "held_days": 1,
                        "entry_atr": entry_atr,
                    }

        if not trades:
            return {"profit_factor": 0.0, "expectancy": 0.0, "mdd": 0.0, "win_rate": 0.0, "trade_count": 0}

        profit_sum = sum(value for value in trades if value > 0)
        loss_sum = abs(sum(value for value in trades if value < 0))
        if loss_sum > 0:
            profit_factor = profit_sum / loss_sum
        elif profit_sum > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
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
            "trade_count": len(trades),
        }

    def _build_signal_expression(self, pl_module, params: dict[str, float]):
        config = load_signal_model_config()
        hard_filters, score_components, pass_score = split_signal_indicators(params)

        hard_filter_expressions = []
        for name, value in hard_filters.items():
            feature_name = name[:-4] if name.endswith(("_min", "_max")) else name
            if name.endswith("_min"):
                hard_filter_expressions.append(pl_module.col(feature_name) >= value)
            elif name.endswith("_max"):
                hard_filter_expressions.append(pl_module.col(feature_name) <= value)
            else:
                hard_filter_expressions.append(pl_module.col(feature_name) == value)

        score_expressions = []
        for name, value in score_components.items():
            feature_name = name[:-4] if name.endswith(("_min", "_max")) else name
            span = config["score_spans"].get(name)
            if span is None or span <= 0:
                if name.endswith("_min"):
                    score_expressions.append(
                        pl_module.when(pl_module.col(feature_name) >= value).then(10.0).otherwise(0.0)
                    )
                elif name.endswith("_max"):
                    score_expressions.append(
                        pl_module.when(pl_module.col(feature_name) <= value).then(10.0).otherwise(0.0)
                    )
                else:
                    score_expressions.append(
                        pl_module.when(pl_module.col(feature_name) == value).then(10.0).otherwise(0.0)
                    )
                continue

            if name.endswith("_max"):
                raw_score = ((value + span) - pl_module.col(feature_name)) / span * 10.0
            elif name.endswith("_min"):
                raw_score = (pl_module.col(feature_name) - (value - span)) / span * 10.0
            else:
                raw_score = pl_module.when(pl_module.col(feature_name) == value).then(10.0).otherwise(0.0)
            if isinstance(raw_score, pl_module.Expr):
                score_expressions.append(raw_score.clip(0.0, 10.0))
            else:
                score_expressions.append(raw_score)

        total_score_expr = (
            reduce(operator.add, score_expressions)
            if score_expressions
            else pl_module.lit(0.0)
        )
        base_signal = total_score_expr >= params.get(SIGNAL_SCORE_MIN_KEY, pass_score)
        if hard_filter_expressions:
            return reduce(operator.and_, hard_filter_expressions) & base_signal
        return base_signal
