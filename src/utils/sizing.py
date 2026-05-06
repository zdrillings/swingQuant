from __future__ import annotations

import math

from src.settings import RuntimeSettings
from src.utils.strategy import ExitRules, stop_risk_per_share


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
