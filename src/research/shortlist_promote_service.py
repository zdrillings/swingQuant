from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.research.shortlist_universe import normalize_eligible_universe_mode, normalize_model_scope
from src.settings import get_settings
from src.utils.db_manager import DatabaseManager


@dataclass(frozen=True)
class ShortlistPromoteReport:
    config_path: str
    production_model_name: str
    production_eligible_universe_mode: str
    production_model_scope: str
    production_xgboost_config: str


class ShortlistPromoteService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def run(
        self,
        *,
        model_name: str,
        eligible_universe_mode: str,
        model_scope: str,
        xgboost_config: str = "baseline",
        horizon_days: int = 20,
    ) -> ShortlistPromoteReport:
        self.db_manager.initialize()
        eligible_universe_mode = normalize_eligible_universe_mode(eligible_universe_mode)
        model_scope = normalize_model_scope(model_scope)
        selected_model_name = str(model_name).strip()
        selected_xgboost_config = str(xgboost_config or "baseline").strip().lower()
        if not selected_model_name:
            raise ValueError("model_name is required.")

        runs = self.db_manager.load_shortlist_model_runs(
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            xgboost_config=selected_xgboost_config,
            limit=1,
        )
        if runs.empty:
            raise ValueError(
                f"No shortlist model runs found for eligible_universe_mode={eligible_universe_mode} "
                f"and model_scope={model_scope} and xgboost_config={selected_xgboost_config}."
            )
        latest_run = runs.iloc[0]
        generated_at = str(latest_run["generated_at"])
        predictions = self.db_manager.load_shortlist_model_predictions(
            generated_at=generated_at,
            horizon_days=int(horizon_days),
            eligible_universe_mode=eligible_universe_mode,
            model_scope=model_scope,
            dataset_split="live",
            model_name=selected_model_name,
        )
        if predictions.empty:
            raise ValueError(
                f"No live predictions found for model_name={selected_model_name}, "
                f"eligible_universe_mode={eligible_universe_mode}, model_scope={model_scope}."
            )

        config_path = get_settings().paths.config_path
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        scan_policy = config.setdefault("scan_policy", {})
        shortlist_model = scan_policy.setdefault("shortlist_model", {})
        shortlist_model["production_model_name"] = selected_model_name
        shortlist_model["production_eligible_universe_mode"] = eligible_universe_mode
        shortlist_model["production_model_scope"] = model_scope
        shortlist_model["production_xgboost_config"] = selected_xgboost_config
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        return ShortlistPromoteReport(
            config_path=str(config_path),
            production_model_name=selected_model_name,
            production_eligible_universe_mode=eligible_universe_mode,
            production_model_scope=model_scope,
            production_xgboost_config=selected_xgboost_config,
        )
