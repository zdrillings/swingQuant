from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import math

from src.settings import load_feature_config, load_production_strategies, load_production_strategy


@dataclass(frozen=True)
class ExitRules:
    trailing_stop_pct: float | None
    profit_target_pct: float | None
    time_limit_days: int
    trailing_stop_atr_mult: float | None = None
    profit_target_atr_mult: float | None = None


@dataclass(frozen=True)
class ProductionStrategy:
    strategy_id: int
    promoted_at: str
    indicators: dict[str, float]
    exit_rules: ExitRules
    slot: str = "default"
    sector: str = "ALL"


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
        },
    }


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
    span = config["score_spans"].get(indicator_name)
    if span is None or span <= 0:
        return 10.0 if indicator_condition(indicator_name, actual_value, threshold_value) else 0.0
    if indicator_name.endswith("_max"):
        raw_score = ((threshold_value + span) - actual_value) / span * 10.0
    elif indicator_name.endswith("_min"):
        raw_score = (actual_value - (threshold_value - span)) / span * 10.0
    else:
        raw_score = 10.0 if actual_value == threshold_value else 0.0
    return max(0.0, min(10.0, raw_score))


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
