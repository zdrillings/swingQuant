from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import math

from src.settings import load_feature_config, load_production_strategies, load_production_strategy


@dataclass(frozen=True)
class ExitRules:
    trailing_stop_pct: float | None
    profit_target_pct: float | None
    time_limit_days: int
    trailing_stop_atr_mult: float | None = None
    profit_target_atr_mult: float | None = None
    exit_before_earnings_days: int | None = None


@dataclass(frozen=True)
class ProductionStrategy:
    strategy_id: int
    promoted_at: str
    indicators: dict[str, float]
    exit_rules: ExitRules
    slot: str = "default"
    sector: str = "ALL"


@dataclass(frozen=True)
class TradeStrategyResolution:
    strategy: ProductionStrategy | None
    source: str


DEFAULT_EXIT_RULES = ExitRules(
    trailing_stop_pct=0.05,
    profit_target_pct=0.12,
    time_limit_days=20,
)

SIGNAL_SCORE_MIN_KEY = "signal_score_min"


def load_active_strategy() -> ProductionStrategy:
    data = load_production_strategy()
    return _production_strategy_from_payload(data, slot="default")


@lru_cache(maxsize=1)
def load_active_strategies() -> dict[str, ProductionStrategy]:
    try:
        data = load_production_strategies()
        strategies_payload = data.get("strategies", {})
        if not strategies_payload:
            raise ValueError("production_strategies.json does not contain any strategies")
        return {
            str(slot): _production_strategy_from_payload(payload, slot=str(slot))
            for slot, payload in strategies_payload.items()
        }
    except Exception:
        strategy = load_active_strategy()
        return {strategy.slot: strategy}


def load_active_strategy_for_slot(slot: str) -> ProductionStrategy:
    strategies = load_active_strategies()
    if slot not in strategies:
        available = ", ".join(sorted(strategies))
        raise ValueError(f"Active strategy slot '{slot}' was not found. Available slots: {available}")
    return strategies[slot]


def clear_strategy_caches() -> None:
    load_signal_model_config.cache_clear()
    load_active_strategies.cache_clear()


def resolve_trade_strategy(
    *,
    trade,
    strategies: dict[str, ProductionStrategy],
    sector_map: dict[str, str],
    backtest_lookup=None,
) -> TradeStrategyResolution:
    strategy_slot = _trade_row_value(trade, "strategy_slot")
    if strategy_slot and strategy_slot in strategies:
        return TradeStrategyResolution(strategy=strategies[strategy_slot], source="active_slot")

    strategy_id = _trade_row_value(trade, "strategy_id")
    if strategy_id is not None:
        for strategy in strategies.values():
            if strategy.strategy_id == int(strategy_id):
                return TradeStrategyResolution(strategy=strategy, source="active_strategy_id")
        if callable(backtest_lookup):
            historical_row = backtest_lookup(int(strategy_id))
            if historical_row is not None:
                return TradeStrategyResolution(
                    strategy=production_strategy_from_backtest_result(
                        historical_row,
                        slot=str(strategy_slot) if strategy_slot not in (None, "") else "legacy",
                    ),
                    source="historical_strategy_id",
                )

    ticker = _trade_row_value(trade, "ticker")
    ticker_sector = sector_map.get(str(ticker)) if ticker is not None else None
    exact_matches = [strategy for strategy in strategies.values() if strategy.sector == ticker_sector]
    if len(exact_matches) == 1:
        return TradeStrategyResolution(strategy=exact_matches[0], source="active_sector_match")

    from src.utils.regime import regime_etf_for_sector

    ticker_regime = regime_etf_for_sector(ticker_sector or "")
    regime_matches = [
        strategy for strategy in strategies.values()
        if regime_etf_for_sector(strategy.sector) == ticker_regime
    ]
    if len(regime_matches) == 1:
        return TradeStrategyResolution(strategy=regime_matches[0], source="active_regime_match")

    all_matches = [strategy for strategy in strategies.values() if strategy.sector == "ALL"]
    if len(all_matches) == 1:
        return TradeStrategyResolution(strategy=all_matches[0], source="active_all_match")
    if len(strategies) == 1:
        return TradeStrategyResolution(strategy=next(iter(strategies.values())), source="sole_active_strategy")
    return TradeStrategyResolution(strategy=None, source="unresolved")


def _production_strategy_from_payload(data: dict, *, slot: str) -> ProductionStrategy:
    exit_rules = data.get("exit_rules", {})
    return ProductionStrategy(
        strategy_id=int(data["strategy_id"]),
        slot=slot,
        sector=str(data.get("sector", "ALL")),
        promoted_at=str(data["promoted_at"]),
        indicators={str(key): float(value) for key, value in data["indicators"].items()},
        exit_rules=ExitRules(
            trailing_stop_pct=float(exit_rules["trailing_stop_pct"]) if exit_rules.get("trailing_stop_pct") is not None else None,
            profit_target_pct=float(exit_rules["profit_target_pct"]) if exit_rules.get("profit_target_pct") is not None else None,
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
            exit_before_earnings_days=(
                int(exit_rules["exit_before_earnings_days"])
                if exit_rules.get("exit_before_earnings_days") is not None
                else None
            ),
        ),
    )


def build_production_strategy_payload(
    *,
    strategy_id: int,
    sector: str,
    indicators: dict[str, float],
    exit_rules: ExitRules,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "sector": sector,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "indicators": indicators,
        "exit_rules": {
            "trailing_stop_pct": exit_rules.trailing_stop_pct,
            "profit_target_pct": exit_rules.profit_target_pct,
            "time_limit_days": exit_rules.time_limit_days,
            "trailing_stop_atr_mult": exit_rules.trailing_stop_atr_mult,
            "profit_target_atr_mult": exit_rules.profit_target_atr_mult,
            "exit_before_earnings_days": exit_rules.exit_before_earnings_days,
        },
    }


def production_strategy_from_backtest_result(
    row,
    *,
    slot: str | None = None,
) -> ProductionStrategy:
    params = json.loads(row["params_json"])
    exit_rules = params.get("exit_rules", {})
    return ProductionStrategy(
        strategy_id=int(row["strategy_id"]),
        slot=slot or str(params.get("slot", "legacy")),
        sector=str(params.get("sector", "ALL")),
        promoted_at=datetime.now().isoformat(timespec="seconds"),
        indicators={str(key): float(value) for key, value in params.get("indicators", {}).items()},
        exit_rules=ExitRules(
            trailing_stop_pct=float(exit_rules["trailing_stop_pct"]) if exit_rules.get("trailing_stop_pct") is not None else None,
            profit_target_pct=float(exit_rules["profit_target_pct"]) if exit_rules.get("profit_target_pct") is not None else None,
            time_limit_days=int(exit_rules.get("time_limit_days", DEFAULT_EXIT_RULES.time_limit_days)),
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
            exit_before_earnings_days=(
                int(exit_rules["exit_before_earnings_days"])
                if exit_rules.get("exit_before_earnings_days") is not None
                else None
            ),
        ),
    )


def uses_atr_trailing_stop(exit_rules: ExitRules) -> bool:
    return exit_rules.trailing_stop_atr_mult is not None


def uses_atr_profit_target(exit_rules: ExitRules) -> bool:
    return exit_rules.profit_target_atr_mult is not None


def trailing_stop_price(*, max_price_seen: float, entry_atr: float | None, exit_rules: ExitRules) -> float:
    if uses_atr_trailing_stop(exit_rules):
        if entry_atr is None or not math.isfinite(entry_atr) or entry_atr <= 0:
            raise ValueError("ATR exit rules require a positive entry_atr value")
        return max_price_seen - (entry_atr * float(exit_rules.trailing_stop_atr_mult))
    if exit_rules.trailing_stop_pct is None or exit_rules.trailing_stop_pct <= 0:
        raise ValueError("Percent trailing stop requires a positive trailing_stop_pct")
    return max_price_seen * (1 - exit_rules.trailing_stop_pct)


def profit_target_price(*, entry_price: float, entry_atr: float | None, exit_rules: ExitRules) -> float:
    if uses_atr_profit_target(exit_rules):
        if entry_atr is None or not math.isfinite(entry_atr) or entry_atr <= 0:
            raise ValueError("ATR exit rules require a positive entry_atr value")
        return entry_price + (entry_atr * float(exit_rules.profit_target_atr_mult))
    if exit_rules.profit_target_pct is None or exit_rules.profit_target_pct <= 0:
        raise ValueError("Percent profit target requires a positive profit_target_pct")
    return entry_price * (1 + exit_rules.profit_target_pct)


def stop_risk_per_share(*, price: float, entry_atr: float | None, exit_rules: ExitRules) -> float:
    if price <= 0:
        raise ValueError("price must be positive")
    if uses_atr_trailing_stop(exit_rules):
        if entry_atr is None or not math.isfinite(entry_atr) or entry_atr <= 0:
            raise ValueError("ATR exit rules require a positive entry_atr value")
        return entry_atr * float(exit_rules.trailing_stop_atr_mult)
    if exit_rules.trailing_stop_pct is None or exit_rules.trailing_stop_pct <= 0:
        raise ValueError("Percent trailing stop requires a positive trailing_stop_pct")
    return price * exit_rules.trailing_stop_pct


@lru_cache(maxsize=1)
def load_signal_model_config() -> dict:
    config = load_feature_config().get("signal_model", {})
    return {
        "default_pass_score": float(config.get("default_pass_score", 25.0)),
        "hard_filter_indicators": tuple(str(value) for value in config.get("hard_filter_indicators", [])),
        "score_spans": {
            str(name): float(value)
            for name, value in config.get("score_spans", {}).items()
        },
        "score_weights": {
            str(name): float(value)
            for name, value in config.get("score_weights", {}).items()
        },
    }


def split_signal_indicators(indicators: dict[str, float]) -> tuple[dict[str, float], dict[str, float], float]:
    config = load_signal_model_config()
    hard_filter_names = set(config["hard_filter_indicators"])
    hard_filters: dict[str, float] = {}
    score_components: dict[str, float] = {}
    pass_score = float(indicators.get(SIGNAL_SCORE_MIN_KEY, config["default_pass_score"]))
    for name, value in indicators.items():
        if name == SIGNAL_SCORE_MIN_KEY:
            continue
        if name in hard_filter_names:
            hard_filters[name] = float(value)
        else:
            score_components[name] = float(value)
    return hard_filters, score_components, pass_score


def feature_name_for_indicator(indicator_name: str) -> str:
    if indicator_name.endswith("_min") or indicator_name.endswith("_max"):
        return indicator_name[:-4]
    return indicator_name


def coerce_signal_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _trade_row_value(row, key: str):
    if hasattr(row, "keys"):
        return row[key] if key in row.keys() else None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return None


def indicator_condition(indicator_name: str, actual_value: float, threshold_value: float) -> bool:
    if not math.isfinite(actual_value):
        return False
    if indicator_name.endswith("_min"):
        return actual_value >= threshold_value
    if indicator_name.endswith("_max"):
        return actual_value <= threshold_value
    return actual_value == threshold_value


def indicator_score(indicator_name: str, actual_value: float, threshold_value: float) -> float:
    config = load_signal_model_config()
    if not math.isfinite(actual_value):
        return 0.0
    weight = float(config["score_weights"].get(indicator_name, 1.0))
    max_score = 10.0 * max(weight, 0.0)
    span = config["score_spans"].get(indicator_name)
    if span is None or span <= 0:
        return max_score if indicator_condition(indicator_name, actual_value, threshold_value) else 0.0
    if indicator_name.endswith("_max"):
        raw_score = ((threshold_value + span) - actual_value) / span * max_score
    elif indicator_name.endswith("_min"):
        raw_score = (actual_value - (threshold_value - span)) / span * max_score
    else:
        raw_score = max_score if actual_value == threshold_value else 0.0
    return max(0.0, min(max_score, raw_score))


def evaluate_signal_gate(
    indicators: dict[str, float],
    latest_row: dict,
) -> tuple[bool, dict[str, dict[str, float | bool | str]], float]:
    details: dict[str, dict[str, float | bool | str]] = {}
    hard_filters, score_components, pass_score = split_signal_indicators(indicators)
    hard_filters_passed = True
    total_score = 0.0

    for threshold_name, threshold_value in hard_filters.items():
        feature_name = feature_name_for_indicator(threshold_name)
        actual_value = coerce_signal_value(latest_row.get(feature_name, float("nan")))
        condition = indicator_condition(threshold_name, actual_value, threshold_value)
        details[threshold_name] = {
            "feature": feature_name,
            "actual": actual_value,
            "threshold": float(threshold_value),
            "passed": condition,
            "score": 10.0 if condition else 0.0,
            "mode": "hard_filter",
        }
        hard_filters_passed = hard_filters_passed and condition

    for threshold_name, threshold_value in score_components.items():
        feature_name = feature_name_for_indicator(threshold_name)
        actual_value = coerce_signal_value(latest_row.get(feature_name, float("nan")))
        condition = indicator_condition(threshold_name, actual_value, threshold_value)
        score = indicator_score(threshold_name, actual_value, threshold_value)
        total_score += score
        details[threshold_name] = {
            "feature": feature_name,
            "actual": actual_value,
            "threshold": float(threshold_value),
            "passed": condition,
            "score": score,
            "mode": "score_component",
        }

    signal_passed = hard_filters_passed and total_score >= pass_score
    details[SIGNAL_SCORE_MIN_KEY] = {
        "feature": "signal_score",
        "actual": total_score,
        "threshold": pass_score,
        "passed": signal_passed,
        "score": total_score,
        "mode": "signal_score_min",
        "hard_filters_passed": hard_filters_passed,
    }
    return signal_passed, details, total_score


def evaluate_indicator_gate(indicators: dict[str, float], latest_row: dict) -> tuple[bool, dict[str, dict[str, float | bool | str]]]:
    passed, details, _ = evaluate_signal_gate(indicators, latest_row)
    return passed, details
