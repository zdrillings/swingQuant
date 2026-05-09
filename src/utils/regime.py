from __future__ import annotations

QQQ_REGIME_SECTORS = {
    "Information Technology",
    "Communication Services",
}

SECTOR_BENCHMARK_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def regime_etf_for_sector(sector: str) -> str:
    return "QQQ" if sector in QQQ_REGIME_SECTORS else "SPY"


def benchmark_etf_for_sector(sector: str) -> str:
    return SECTOR_BENCHMARK_ETFS.get(sector, "SPY")
