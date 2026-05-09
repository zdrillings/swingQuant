from __future__ import annotations

import json

from src.utils.db_manager import DatabaseManager
from src.utils.promotion_policy import promotion_policy_violations
from src.utils.strategy import ExitRules, build_production_strategy_payload, clear_strategy_caches


class PromoteService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(self, *, row_id: int, slot: str | None = None) -> str:
        self.db_manager.initialize()
        row = self.db_manager.get_backtest_result(row_id)
        if row is None:
            raise ValueError(f"Backtest result {row_id} was not found.")
        self._validate_promotion_quality(row)

        params = json.loads(row["params_json"])
        indicators = {key: float(value) for key, value in params["indicators"].items()}
        exit_rules_payload = params.get("exit_rules", {})
        sector = str(params.get("sector", "ALL"))
        resolved_slot = slot or self._default_slot_for_sector(sector)
        payload = build_production_strategy_payload(
            strategy_id=int(row["strategy_id"]),
            sector=sector,
            indicators=indicators,
            exit_rules=ExitRules(
                trailing_stop_pct=(
                    float(exit_rules_payload["trailing_stop_pct"])
                    if exit_rules_payload.get("trailing_stop_pct") is not None
                    else None
                ),
                profit_target_pct=(
                    float(exit_rules_payload["profit_target_pct"])
                    if exit_rules_payload.get("profit_target_pct") is not None
                    else None
                ),
                time_limit_days=int(exit_rules_payload["time_limit_days"]),
                trailing_stop_atr_mult=(
                    float(exit_rules_payload["trailing_stop_atr_mult"])
                    if exit_rules_payload.get("trailing_stop_atr_mult") is not None
                    else None
                ),
                profit_target_atr_mult=(
                    float(exit_rules_payload["profit_target_atr_mult"])
                    if exit_rules_payload.get("profit_target_atr_mult") is not None
                    else None
                ),
            ),
        )
        strategies_path = self.db_manager.paths.production_strategies_path or (
            self.db_manager.paths.root_dir / "production_strategies.json"
        )
        strategies_document = {"strategies": {}}
        if strategies_path.exists():
            strategies_document = json.loads(
                strategies_path.read_text(encoding="utf-8")
            )
        strategies_document.setdefault("strategies", {})
        strategies_document["strategies"][resolved_slot] = payload
        strategies_path.write_text(
            json.dumps(strategies_document, indent=2),
            encoding="utf-8",
        )
        self.db_manager.paths.production_strategy_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        clear_strategy_caches()
        return (
            f"Strategy {row_id} promoted into slot '{resolved_slot}'. "
            "production_strategies.json updated."
        )

    def _validate_promotion_quality(self, row) -> None:
        violations = promotion_policy_violations(
            profit_factor=float(row["profit_factor"]),
            expectancy=float(row["expectancy"]),
            mdd=float(row["mdd"]),
            trade_count=int(row["trade_count"]) if row["trade_count"] is not None else None,
        )
        if violations:
            joined = "; ".join(violations)
            raise ValueError(f"Backtest result {row['id']} does not satisfy promotion policy: {joined}")

    def _default_slot_for_sector(self, sector: str) -> str:
        if not sector or sector == "ALL":
            return "default"
        normalized = sector.lower().replace("&", "and")
        return "_".join(normalized.split())
