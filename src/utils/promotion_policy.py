from __future__ import annotations

from dataclasses import dataclass

from src.settings import load_feature_config


@dataclass(frozen=True)
class PromotionPolicy:
    min_profit_factor: float | None
    min_expectancy: float | None
    min_trade_count: int | None
    max_mdd: float | None


def load_promotion_policy() -> PromotionPolicy:
    policy = load_feature_config().get("promotion_policy", {})
    return PromotionPolicy(
        min_profit_factor=float(policy["min_profit_factor"]) if policy.get("min_profit_factor") is not None else None,
        min_expectancy=float(policy["min_expectancy"]) if policy.get("min_expectancy") is not None else None,
        min_trade_count=int(policy["min_trade_count"]) if policy.get("min_trade_count") is not None else None,
        max_mdd=float(policy["max_mdd"]) if policy.get("max_mdd") is not None else None,
    )


def promotion_policy_violations(
    *,
    profit_factor: float,
    expectancy: float,
    mdd: float,
    trade_count: int | None,
    policy: PromotionPolicy | None = None,
) -> list[str]:
    resolved_policy = policy or load_promotion_policy()
    violations: list[str] = []
    if resolved_policy.min_profit_factor is not None and profit_factor < resolved_policy.min_profit_factor:
        violations.append(
            f"profit_factor {profit_factor:.6f} < {resolved_policy.min_profit_factor:.6f}"
        )
    if resolved_policy.min_expectancy is not None and expectancy < resolved_policy.min_expectancy:
        violations.append(
            f"expectancy {expectancy:.6f} < {resolved_policy.min_expectancy:.6f}"
        )
    if resolved_policy.min_trade_count is not None:
        if trade_count is None or trade_count < resolved_policy.min_trade_count:
            violations.append(
                f"trade_count {trade_count if trade_count is not None else 'unknown'} < {resolved_policy.min_trade_count}"
            )
    if resolved_policy.max_mdd is not None and mdd > resolved_policy.max_mdd:
        violations.append(
            f"mdd {mdd:.6f} > {resolved_policy.max_mdd:.6f}"
        )
    return violations


def passes_promotion_policy(
    *,
    profit_factor: float,
    expectancy: float,
    mdd: float,
    trade_count: int | None,
    policy: PromotionPolicy | None = None,
) -> bool:
    return not promotion_policy_violations(
        profit_factor=profit_factor,
        expectancy=expectancy,
        mdd=mdd,
        trade_count=trade_count,
        policy=policy,
    )
