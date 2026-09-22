from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.settings import RuntimeSettings
from src.utils.strategy import ExitRules, stop_risk_per_share


@dataclass(frozen=True)
class HeuristicSizingPolicy:
    vol_target_daily_pct: float = 0.025
    vol_scale_floor: float = 0.50
    portfolio_stop_risk_cap_pct: float = 0.04


def compute_position_size(
    *,
    price: float,
    exit_rules: ExitRules,
    settings: RuntimeSettings,
    entry_atr: float | None = None,
) -> int:
    if settings.total_capital is None or settings.risk_per_trade is None:
        raise ValueError("TOTAL_CAPITAL and RISK_PER_TRADE must be set in .env")
    risk_per_share = stop_risk_per_share(price=price, entry_atr=entry_atr, exit_rules=exit_rules)
    risk_dollars = settings.total_capital * settings.risk_per_trade
    return math.floor(risk_dollars / risk_per_share)


def vol_target_multiplier(
    *,
    atr_pct_14,
    policy: HeuristicSizingPolicy,
) -> float:
    try:
        atr_pct = float(atr_pct_14)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(atr_pct) or atr_pct <= 0.0:
        return 1.0
    raw = float(policy.vol_target_daily_pct) / atr_pct
    return max(float(policy.vol_scale_floor), min(raw, 1.0))


def compute_heuristic_position_size(
    *,
    price: float,
    exit_rules: ExitRules,
    settings: RuntimeSettings,
    sizing_policy: HeuristicSizingPolicy,
    atr_pct_14,
    entry_atr: float | None = None,
) -> tuple[int, float]:
    base_shares = compute_position_size(
        price=price,
        exit_rules=exit_rules,
        settings=settings,
        entry_atr=entry_atr,
    )
    multiplier = vol_target_multiplier(atr_pct_14=atr_pct_14, policy=sizing_policy)
    shares = math.floor(float(base_shares) * float(multiplier))
    return max(int(shares), 0), float(multiplier)


def stop_risk_dollars(
    *,
    shares: int,
    price: float,
    exit_rules: ExitRules,
    entry_atr: float | None = None,
) -> float:
    if int(shares) <= 0:
        return 0.0
    risk_per_share = stop_risk_per_share(price=price, entry_atr=entry_atr, exit_rules=exit_rules)
    return float(shares) * float(risk_per_share)


def apply_portfolio_stop_risk_cap(
    candidates: pd.DataFrame,
    *,
    settings: RuntimeSettings,
    sizing_policy: HeuristicSizingPolicy,
    score_column: str = "signal_score",
) -> pd.DataFrame:
    if candidates.empty or settings.total_capital is None:
        return candidates.copy()
    cap_dollars = float(settings.total_capital) * float(sizing_policy.portfolio_stop_risk_cap_pct)
    if cap_dollars <= 0.0:
        return candidates.iloc[0:0].copy()
    working = candidates.copy()
    if "stop_risk_dollars" not in working.columns:
        return working
    working["_stop_risk_for_cap"] = pd.to_numeric(working["stop_risk_dollars"], errors="coerce").fillna(0.0)
    working["_score_for_risk_cap"] = pd.to_numeric(working.get(score_column, 0.0), errors="coerce").fillna(0.0)
    while not working.empty and float(working["_stop_risk_for_cap"].sum()) > cap_dollars:
        weakest_index = working.sort_values(
            ["_score_for_risk_cap", "opportunity_score", "ticker"],
            ascending=[True, True, False],
        ).index[0]
        working = working.drop(index=weakest_index)
    return working.drop(columns=["_stop_risk_for_cap", "_score_for_risk_cap"], errors="ignore")
