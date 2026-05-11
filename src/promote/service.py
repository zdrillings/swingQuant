from __future__ import annotations

import json
import pandas as pd

from src.evaluate.service import EvaluateService
from src.utils.db_manager import DatabaseManager
from src.utils.promotion_policy import (
    load_promotion_policy,
    promotion_policy_violations,
    walk_forward_policy_enabled,
    walk_forward_policy_violations,
)
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
                exit_before_earnings_days=(
                    int(exit_rules_payload["exit_before_earnings_days"])
                    if exit_rules_payload.get("exit_before_earnings_days") is not None
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
        policy = load_promotion_policy()
        violations = promotion_policy_violations(
            profit_factor=float(row["profit_factor"]),
            expectancy=float(row["expectancy"]),
            mdd=float(row["mdd"]),
            trade_count=int(row["trade_count"]) if row["trade_count"] is not None else None,
            policy=policy,
        )
        if not violations and walk_forward_policy_enabled(policy):
            walk_forward_metrics = self._compute_walk_forward_metrics(row)
            violations.extend(
                walk_forward_policy_violations(
                    window_count=walk_forward_metrics.get("wf_window_count"),
                    positive_window_ratio=walk_forward_metrics.get("wf_positive_window_ratio"),
                    positive_alpha_window_ratio=walk_forward_metrics.get("wf_positive_alpha_window_ratio"),
                    median_expectancy=walk_forward_metrics.get("wf_median_expectancy"),
                    worst_mdd=walk_forward_metrics.get("wf_worst_mdd"),
                    trade_count_min=walk_forward_metrics.get("wf_trade_count_min"),
                    policy=policy,
                )
            )
        if violations:
            joined = "; ".join(violations)
            raise ValueError(f"Backtest result {row['id']} does not satisfy promotion policy: {joined}")

    def _compute_walk_forward_metrics(self, row) -> dict[str, float | int]:
        policy = load_promotion_policy()
        params = json.loads(row["params_json"])
        frame = pd.DataFrame(
            [
                {
                    "id": int(row["id"]),
                    "run_id": int(row["run_id"]) if row["run_id"] is not None else None,
                    "strategy_id": int(row["strategy_id"]),
                    "params_json": row["params_json"],
                    "sector": str(params.get("sector", "ALL")),
                    "profit_factor": float(row["profit_factor"]),
                    "expectancy": float(row["expectancy"]),
                    "alpha_vs_spy": float(row["alpha_vs_spy"]) if row["alpha_vs_spy"] is not None else float("nan"),
                    "alpha_vs_sector": float(row["alpha_vs_sector"]) if row["alpha_vs_sector"] is not None else float("nan"),
                    "mdd": float(row["mdd"]),
                    "win_rate": float(row["win_rate"]),
                    "trade_count": int(row["trade_count"]) if row["trade_count"] is not None else None,
                    "norm_score": float(row["norm_score"]) if row["norm_score"] is not None else 0.0,
                    "practical_score": float(row["norm_score"]) if row["norm_score"] is not None else 0.0,
                    "live_match_count": 0,
                    "promotion_policy_passed": True,
                }
            ]
        )
        metrics = EvaluateService(self.db_manager)._build_walk_forward_stability(
            ranked=frame,
            top=1,
            shortlist_size=1,
            windows=policy.walk_forward_windows or 5,
        )
        if metrics.empty:
            return {}
        return {
            key: metrics.iloc[0][key]
            for key in [
                "wf_window_count",
                "wf_median_expectancy",
                "wf_worst_expectancy",
                "wf_positive_window_ratio",
                "wf_positive_alpha_window_ratio",
                "wf_median_alpha_vs_spy",
                "wf_worst_mdd",
                "wf_trade_count_min",
                "wf_stability_score",
            ]
            if key in metrics.columns
        }

    def _default_slot_for_sector(self, sector: str) -> str:
        if not sector or sector == "ALL":
            return "default"
        normalized = sector.lower().replace("&", "and")
        return "_".join(normalized.split())
