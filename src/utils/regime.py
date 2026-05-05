from __future__ import annotations

QQQ_REGIME_SECTORS = {
    "Information Technology",
    "Communication Services",
}


def regime_etf_for_sector(sector: str) -> str:
    return "QQQ" if sector in QQQ_REGIME_SECTORS else "SPY"

