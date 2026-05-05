from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.features import build_feature_frame, chronological_split
from src.settings import load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger


@dataclass(frozen=True)
class ResearchResult:
    feature_importance: pd.DataFrame
    train_rows: int
    validation_rows: int

    def render_console_table(self) -> str:
        header = "Feature Importance (Information Gain)"
        lines = [header, "-" * len(header), f"{'Feature':<24} {'Gain':>12}"]
        for row in self.feature_importance.itertuples(index=False):
            lines.append(f"{row.feature:<24} {row.gain:>12.6f}")
        return "\n".join(lines)


class ResearchService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("research")

    def run(self) -> ResearchResult:
        self.db_manager.initialize()
        research_rows = self.db_manager.list_research_universe(limit=250)
        if not research_rows:
            raise ValueError("Universe is empty or liquidity metrics are unavailable. Run `sq sync` first.")

        universe_tickers = [row["ticker"] for row in research_rows]
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS))
        price_history = self.db_manager.load_price_history(tickers)
        feature_config = load_feature_config()

        research_history = price_history[price_history["ticker"].isin(tickers)].copy()
        feature_frame, feature_columns = build_feature_frame(research_history, feature_config)
        feature_frame = feature_frame[feature_frame["ticker"].isin(universe_tickers)].copy()
        train_frame, validation_frame = chronological_split(feature_frame, train_ratio=0.7)

        importance = self._train_xgboost(
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_columns=feature_columns,
        )
        return ResearchResult(
            feature_importance=importance,
            train_rows=len(train_frame.index),
            validation_rows=len(validation_frame.index),
        )

    def _train_xgboost(
        self,
        *,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise RuntimeError("xgboost is required for `sq research`. Install project dependencies first.") from exc

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(
            train_frame[feature_columns],
            train_frame["success"],
            eval_set=[(validation_frame[feature_columns], validation_frame["success"])],
            verbose=False,
        )
        importance_by_feature = model.get_booster().get_score(importance_type="gain")

        rows = [
            {"feature": feature, "gain": float(importance_by_feature.get(feature, 0.0))}
            for feature in feature_columns
        ]
        return pd.DataFrame(rows).sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)
