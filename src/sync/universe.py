from __future__ import annotations

from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

from src.utils.db_manager import UniverseRow


WIKIPEDIA_INDEX_SOURCES = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
)
WIKIPEDIA_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def _find_symbol_column(frame: pd.DataFrame) -> str:
    for candidate in ("Symbol", "Ticker symbol", "Ticker"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find ticker column in table columns: {list(frame.columns)}")


def _find_sector_column(frame: pd.DataFrame) -> str:
    for candidate in ("GICS Sector", "Sector"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find sector column in table columns: {list(frame.columns)}")


def _fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": WIKIPEDIA_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def scrape_sp_universe() -> list[UniverseRow]:
    seen: dict[str, UniverseRow] = {}

    for url in WIKIPEDIA_INDEX_SOURCES:
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html))
        matching_table = None
        for frame in tables:
            if any(column in frame.columns for column in ("Symbol", "Ticker symbol", "Ticker")):
                matching_table = frame
                break
        if matching_table is None:
            raise ValueError(f"Unable to locate ticker table on page: {url}")

        symbol_column = _find_symbol_column(matching_table)
        sector_column = _find_sector_column(matching_table)

        for _, row in matching_table.iterrows():
            ticker = normalize_ticker(str(row[symbol_column]))
            sector = str(row[sector_column]).strip()
            if ticker and ticker not in seen:
                seen[ticker] = UniverseRow(ticker=ticker, sector=sector, is_active=True)

    return sorted(seen.values(), key=lambda member: member.ticker)
