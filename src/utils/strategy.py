from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.settings import load_production_strategy


@dataclass(frozen=True)
class ExitRules:
    trailing_stop_pct: float
    profit_target_pct: float
    time_limit_days: int


@dataclass(frozen=True)
class ProductionStrategy:
    strategy_id: int
    promoted_at: str
    indicators: dict[str, float]
    exit_rules: ExitRules


DEFAULT_EXIT_RULES = ExitRules(
    trailing_stop_pct=0.05,
    profit_target_pct=0.12,
    time_limit_days=20,
)


def load_active_strategy() -> ProductionStrategy:
    data = load_production_strategy()
    exit_rules = data.get("exit_rules", {})
    return ProductionStrategy(
        strategy_id=int(data["strategy_id"]),
        promoted_at=str(data["promoted_at"]),
        indicators={str(key): float(value) for key, value in data["indicators"].items()},
        exit_rules=ExitRules(
            trailing_stop_pct=float(exit_rules["trailing_stop_pct"]),
            profit_target_pct=float(exit_rules["profit_target_pct"]),
            time_limit_days=int(exit_rules["time_limit_days"]),
        ),
    )


def build_production_strategy_payload(
    *,
    strategy_id: int,
    indicators: dict[str, float],
    exit_rules: ExitRules,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "indicators": indicators,
        "exit_rules": {
            "trailing_stop_pct": exit_rules.trailing_stop_pct,
            "profit_target_pct": exit_rules.profit_target_pct,
            "time_limit_days": exit_rules.time_limit_days,
        },
    }


def evaluate_indicator_gate(indicators: dict[str, float], latest_row: dict) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    details: dict[str, dict[str, float | bool]] = {}
    passed = True
    for threshold_name, threshold_value in indicators.items():
        if threshold_name.endswith("_min"):
            feature_name = threshold_name[:-4]
            actual_value = float(latest_row[feature_name])
            condition = actual_value >= float(threshold_value)
        elif threshold_name.endswith("_max"):
            feature_name = threshold_name[:-4]
            actual_value = float(latest_row[feature_name])
            condition = actual_value <= float(threshold_value)
        else:
            feature_name = threshold_name
            actual_value = float(latest_row[feature_name])
            condition = actual_value == float(threshold_value)
        details[threshold_name] = {
            "feature": feature_name,
            "actual": actual_value,
            "threshold": float(threshold_value),
            "passed": condition,
        }
        passed = passed and condition
    return passed, details
