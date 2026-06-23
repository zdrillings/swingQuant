from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.sync.universe import scrape_sp_universe
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


@dataclass(frozen=True)
class RefreshUniverseReport:
    refreshed_rows: int


class RefreshUniverseService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        universe_loader: Callable[[], list] = scrape_sp_universe,
    ) -> None:
        self.db_manager = db_manager
        self.universe_loader = universe_loader
        self.logger = get_logger("refresh_universe")

    def run(self) -> RefreshUniverseReport:
        self.db_manager.initialize()
        members = self.universe_loader()
        refreshed_rows = self.db_manager.refresh_universe_metadata(members)
        self.logger.info("Refreshed universe metadata for %s tickers", refreshed_rows)
        return RefreshUniverseReport(refreshed_rows=refreshed_rows)
