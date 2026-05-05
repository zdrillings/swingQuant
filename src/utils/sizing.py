from __future__ import annotations

import math

from src.settings import RuntimeSettings


def compute_position_size(
    *,
    price: float,
    trailing_stop_pct: float,
    settings: RuntimeSettings,
) -> int:
    if settings.total_capital is None or settings.risk_per_trade is None:
        raise ValueError("TOTAL_CAPITAL and RISK_PER_TRADE must be set in .env")
    if price <= 0 or trailing_stop_pct <= 0:
        raise ValueError("price and trailing_stop_pct must be positive")
    risk_dollars = settings.total_capital * settings.risk_per_trade
    return math.floor(risk_dollars / (price * trailing_stop_pct))
