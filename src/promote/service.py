from __future__ import annotations

import json

from src.utils.db_manager import DatabaseManager
from src.utils.strategy import ExitRules, build_production_strategy_payload


class PromoteService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(self, *, row_id: int) -> str:
        self.db_manager.initialize()
        row = self.db_manager.get_backtest_result(row_id)
        if row is None:
            raise ValueError(f"Backtest result {row_id} was not found.")

        params = json.loads(row["params_json"])
        indicators = {key: float(value) for key, value in params["indicators"].items()}
        exit_rules_payload = params.get("exit_rules", {})
        payload = build_production_strategy_payload(
            strategy_id=int(row["strategy_id"]),
            indicators=indicators,
            exit_rules=ExitRules(
                trailing_stop_pct=float(exit_rules_payload["trailing_stop_pct"]),
                profit_target_pct=float(exit_rules_payload["profit_target_pct"]),
                time_limit_days=int(exit_rules_payload["time_limit_days"]),
            ),
        )
        self.db_manager.paths.production_strategy_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return f"Strategy {row_id} promoted. production_strategy.json updated."
