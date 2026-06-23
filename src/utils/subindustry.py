from __future__ import annotations

from src.utils.regime import benchmark_etf_for_sector


SUBINDUSTRY_BENCHMARK_ETFS = {
    "Semiconductors": "SMH",
    "Semiconductor Materials & Equipment": "SMH",
    "Application Software": "IGV",
    "Systems Software": "IGV",
    "Internet Services & Infrastructure": "SKYY",
}


def benchmark_etf_for_sub_industry(sector: str | None, sub_industry: str | None) -> str:
    normalized_sub_industry = str(sub_industry or "").strip()
    if normalized_sub_industry:
        mapped = SUBINDUSTRY_BENCHMARK_ETFS.get(normalized_sub_industry)
        if mapped:
            return mapped
    return benchmark_etf_for_sector(str(sector or "").strip())
