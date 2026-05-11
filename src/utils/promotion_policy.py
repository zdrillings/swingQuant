from __future__ import annotations

from dataclasses import dataclass

from src.settings import load_feature_config


@dataclass(frozen=True)
class PromotionPolicy:
    min_profit_factor: float | None
    min_expectancy: float | None
    min_trade_count: int | None
    max_mdd: float | None
    walk_forward_windows: int | None = None
    walk_forward_min_window_count: int | None = None
    walk_forward_min_positive_window_ratio: float | None = None
    walk_forward_min_positive_alpha_window_ratio: float | None = None
    walk_forward_min_median_expectancy: float | None = None
    walk_forward_max_worst_mdd: float | None = None
    walk_forward_min_trade_count_min: int | None = None


def load_promotion_policy() -> PromotionPolicy:
    policy = load_feature_config().get("promotion_policy", {})
    walk_forward = policy.get("walk_forward", {})
    return PromotionPolicy(
        min_profit_factor=float(policy["min_profit_factor"]) if policy.get("min_profit_factor") is not None else None,
        min_expectancy=float(policy["min_expectancy"]) if policy.get("min_expectancy") is not None else None,
        min_trade_count=int(policy["min_trade_count"]) if policy.get("min_trade_count") is not None else None,
        max_mdd=float(policy["max_mdd"]) if policy.get("max_mdd") is not None else None,
        walk_forward_windows=(
            int(walk_forward["windows"])
            if walk_forward.get("windows") is not None
            else None
        ),
        walk_forward_min_window_count=(
            int(walk_forward["min_window_count"])
            if walk_forward.get("min_window_count") is not None
            else None
        ),
        walk_forward_min_positive_window_ratio=(
            float(walk_forward["min_positive_window_ratio"])
            if walk_forward.get("min_positive_window_ratio") is not None
            else None
        ),
        walk_forward_min_positive_alpha_window_ratio=(
            float(walk_forward["min_positive_alpha_window_ratio"])
            if walk_forward.get("min_positive_alpha_window_ratio") is not None
            else None
        ),
        walk_forward_min_median_expectancy=(
            float(walk_forward["min_median_expectancy"])
            if walk_forward.get("min_median_expectancy") is not None
            else None
        ),
        walk_forward_max_worst_mdd=(
            float(walk_forward["max_worst_mdd"])
            if walk_forward.get("max_worst_mdd") is not None
            else None
        ),
        walk_forward_min_trade_count_min=(
            int(walk_forward["min_trade_count_min"])
            if walk_forward.get("min_trade_count_min") is not None
            else None
        ),
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


def walk_forward_policy_enabled(policy: PromotionPolicy | None = None) -> bool:
    resolved_policy = policy or load_promotion_policy()
    return any(
        value is not None
        for value in [
            resolved_policy.walk_forward_min_window_count,
            resolved_policy.walk_forward_min_positive_window_ratio,
            resolved_policy.walk_forward_min_positive_alpha_window_ratio,
            resolved_policy.walk_forward_min_median_expectancy,
            resolved_policy.walk_forward_max_worst_mdd,
            resolved_policy.walk_forward_min_trade_count_min,
        ]
    )


def walk_forward_policy_violations(
    *,
    window_count: int | None,
    positive_window_ratio: float | None,
    positive_alpha_window_ratio: float | None,
    median_expectancy: float | None,
    worst_mdd: float | None,
    trade_count_min: int | None,
    policy: PromotionPolicy | None = None,
) -> list[str]:
    resolved_policy = policy or load_promotion_policy()
    violations: list[str] = []
    if (
        resolved_policy.walk_forward_min_window_count is not None
        and (window_count is None or window_count < resolved_policy.walk_forward_min_window_count)
    ):
        violations.append(
            f"wf_window_count {window_count if window_count is not None else 'unknown'} < {resolved_policy.walk_forward_min_window_count}"
        )
    if (
        resolved_policy.walk_forward_min_positive_window_ratio is not None
        and (
            positive_window_ratio is None
            or positive_window_ratio < resolved_policy.walk_forward_min_positive_window_ratio
        )
    ):
        violations.append(
            f"wf_positive_window_ratio {positive_window_ratio if positive_window_ratio is not None else 'unknown'} < {resolved_policy.walk_forward_min_positive_window_ratio:.6f}"
        )
    if (
        resolved_policy.walk_forward_min_positive_alpha_window_ratio is not None
        and (
            positive_alpha_window_ratio is None
            or positive_alpha_window_ratio < resolved_policy.walk_forward_min_positive_alpha_window_ratio
        )
    ):
        violations.append(
            "wf_positive_alpha_window_ratio "
            f"{positive_alpha_window_ratio if positive_alpha_window_ratio is not None else 'unknown'} "
            f"< {resolved_policy.walk_forward_min_positive_alpha_window_ratio:.6f}"
        )
    if (
        resolved_policy.walk_forward_min_median_expectancy is not None
        and (median_expectancy is None or median_expectancy < resolved_policy.walk_forward_min_median_expectancy)
    ):
        violations.append(
            f"wf_median_expectancy {median_expectancy if median_expectancy is not None else 'unknown'} < {resolved_policy.walk_forward_min_median_expectancy:.6f}"
        )
    if (
        resolved_policy.walk_forward_max_worst_mdd is not None
        and (worst_mdd is None or worst_mdd > resolved_policy.walk_forward_max_worst_mdd)
    ):
        violations.append(
            f"wf_worst_mdd {worst_mdd if worst_mdd is not None else 'unknown'} > {resolved_policy.walk_forward_max_worst_mdd:.6f}"
        )
    if (
        resolved_policy.walk_forward_min_trade_count_min is not None
        and (trade_count_min is None or trade_count_min < resolved_policy.walk_forward_min_trade_count_min)
    ):
        violations.append(
            f"wf_trade_count_min {trade_count_min if trade_count_min is not None else 'unknown'} < {resolved_policy.walk_forward_min_trade_count_min}"
        )
    return violations
