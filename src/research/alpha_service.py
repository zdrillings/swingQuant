from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from src.settings import load_feature_config
from src.sync.service import REFERENCE_TICKERS
from src.utils.db_manager import DatabaseManager
from src.utils.logging import get_logger
from src.utils.regime import benchmark_etf_for_sector
from src.utils.signal_engine import build_analysis_frame, latest_snapshot


@dataclass(frozen=True)
class AlphaResearchReport:
    output_path: str
    train_rows: int
    validation_rows: int


class AlphaResearchService:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.logger = get_logger("alpha_research")

    def run(
        self,
        *,
        top: int = 10,
        sector: str | None = None,
        horizon_days: int = 10,
        benchmark: str = "sector",
    ) -> AlphaResearchReport:
        self.db_manager.initialize()
        research_rows = self.db_manager.list_research_universe(limit=250)
        if not research_rows:
            raise ValueError("Universe is empty or liquidity metrics are unavailable. Run `sq sync` first.")

        if benchmark not in {"sector", "spy"}:
            raise ValueError("benchmark must be 'sector' or 'spy'")

        if sector is not None:
            research_rows = [row for row in research_rows if row["sector"] == sector]
            if not research_rows:
                raise ValueError(f"No research-universe rows found for sector={sector}")

        universe_tickers = [row["ticker"] for row in research_rows]
        sector_benchmarks = {
            benchmark_etf_for_sector(row["sector"])
            for row in research_rows
            if benchmark_etf_for_sector(row["sector"]) is not None
        }
        tickers = sorted(set(universe_tickers).union(REFERENCE_TICKERS).union(sector_benchmarks))
        price_history = self.db_manager.load_price_history(tickers)
        if price_history.empty:
            raise ValueError("Historical prices are unavailable. Run `sq sync` first.")

        earnings_loader = getattr(self.db_manager, "load_earnings_calendar", None)
        earnings_calendar = earnings_loader(universe_tickers) if callable(earnings_loader) else pd.DataFrame()
        analysis_frame, feature_columns = build_analysis_frame(
            price_history,
            research_rows,
            earnings_calendar=earnings_calendar,
        )
        analysis_frame = analysis_frame[analysis_frame["ticker"].isin(universe_tickers)].copy()
        if analysis_frame.empty:
            raise ValueError("No analysis frame could be built for alpha research.")

        benchmark_returns = self._build_benchmark_future_returns(price_history, horizon_days=horizon_days)
        labeled = self._build_labeled_frame(
            analysis_frame,
            feature_columns=feature_columns,
            benchmark_returns=benchmark_returns,
            benchmark=benchmark,
            horizon_days=horizon_days,
        )
        if labeled.empty:
            raise ValueError("No labeled rows were produced for alpha research.")

        train_frame, validation_frame = self._chronological_split(labeled, train_ratio=0.7)
        if train_frame.empty or validation_frame.empty:
            raise ValueError("Alpha research requires both train and validation rows.")

        feature_set = self._feature_columns(labeled, feature_columns)
        model, importance = self._fit_regressor(
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_columns=feature_set,
        )
        validation_predictions = model.predict(validation_frame[feature_set])
        validation_metrics = self._validation_metrics(
            actual=validation_frame["forward_excess_return"].to_numpy(dtype=float),
            predicted=np.asarray(validation_predictions, dtype=float),
        )
        live_predictions = self._score_latest_snapshot(
            model=model,
            frame=analysis_frame,
            feature_columns=feature_set,
            top=top,
            sector=sector,
        )
        report_path = self.db_manager.paths.reports_dir / "alpha_research.md"
        lines = [
            "# Alpha Research",
            "",
            f"- sector: {sector or 'ALL'}",
            f"- horizon_days: {horizon_days}",
            f"- benchmark: {benchmark}",
            f"- train_rows: {len(train_frame.index)}",
            f"- validation_rows: {len(validation_frame.index)}",
            f"- validation_rmse: {validation_metrics['rmse']:.6f}",
            f"- validation_mae: {validation_metrics['mae']:.6f}",
            f"- validation_directional_accuracy: {validation_metrics['directional_accuracy']:.6f}",
            f"- validation_correlation: {validation_metrics['correlation']:.6f}",
            "",
        ]
        lines.extend(self._render_predictions("Top Predicted Live Excess Return Candidates", live_predictions))
        lines.extend(self._render_importance(importance))
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return AlphaResearchReport(
            output_path=str(report_path),
            train_rows=len(train_frame.index),
            validation_rows=len(validation_frame.index),
        )

    def _build_benchmark_future_returns(self, price_history: pd.DataFrame, *, horizon_days: int) -> pd.DataFrame:
        benchmark_frame = price_history.copy()
        benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
        benchmark_frame = benchmark_frame.sort_values(["ticker", "date"]).copy()
        benchmark_frame["future_adj_close"] = benchmark_frame.groupby("ticker")["adj_close"].shift(-horizon_days)
        benchmark_frame["benchmark_return"] = (benchmark_frame["future_adj_close"] / benchmark_frame["adj_close"]) - 1.0
        return benchmark_frame[["ticker", "date", "benchmark_return"]].dropna().reset_index(drop=True)

    def _build_labeled_frame(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        benchmark_returns: pd.DataFrame,
        benchmark: str,
        horizon_days: int,
    ) -> pd.DataFrame:
        working = frame.sort_values(["ticker", "date"]).copy()
        working["future_adj_close"] = working.groupby("ticker")["adj_close"].shift(-horizon_days)
        working["forward_return"] = (working["future_adj_close"] / working["adj_close"]) - 1.0
        if benchmark == "sector":
            working["benchmark_ticker"] = working["sector"].map(benchmark_etf_for_sector).fillna("SPY")
        else:
            working["benchmark_ticker"] = "SPY"
        benchmark_map = benchmark_returns.rename(columns={"ticker": "benchmark_ticker"})
        working = working.merge(
            benchmark_map,
            on=["benchmark_ticker", "date"],
            how="left",
        )
        working["forward_excess_return"] = working["forward_return"] - working["benchmark_return"]
        feature_set = self._feature_columns(working, feature_columns)
        required_columns = ["ticker", "date", "sector", "forward_excess_return"] + feature_set
        return working[required_columns].dropna().reset_index(drop=True)

    def _feature_columns(self, frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
        extras = [
            "regime_green",
            "spy_regime_green",
            "qqq_regime_green",
            "sector_pct_above_50",
            "sector_pct_above_200",
            "sector_median_roc_63",
            "md_volume_30d",
        ]
        columns = []
        for name in feature_columns + extras:
            if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]):
                columns.append(name)
        return sorted(set(columns))

    def _chronological_split(self, frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(pd.to_datetime(frame["date"]).drop_duplicates().tolist())
        split_index = int(len(unique_dates) * train_ratio)
        split_index = min(max(split_index, 1), len(unique_dates) - 1)
        train_dates = set(unique_dates[:split_index])
        validation_dates = set(unique_dates[split_index:])
        return (
            frame[frame["date"].isin(train_dates)].copy(),
            frame[frame["date"].isin(validation_dates)].copy(),
        )

    def _fit_regressor(
        self,
        *,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        feature_columns: list[str],
    ):
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError("xgboost is required for `sq alpha-research`. Install project dependencies first.") from exc

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(
            train_frame[feature_columns],
            train_frame["forward_excess_return"],
            eval_set=[(validation_frame[feature_columns], validation_frame["forward_excess_return"])],
            verbose=False,
        )
        importance_by_feature = model.get_booster().get_score(importance_type="gain")
        rows = [
            {"feature": feature, "gain": float(importance_by_feature.get(feature, 0.0))}
            for feature in feature_columns
        ]
        importance = pd.DataFrame(rows).sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)
        return model, importance

    def _validation_metrics(self, *, actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        residual = predicted - actual
        rmse = sqrt(float(np.mean(np.square(residual)))) if len(residual) else 0.0
        mae = float(np.mean(np.abs(residual))) if len(residual) else 0.0
        directional_accuracy = float(np.mean(np.sign(predicted) == np.sign(actual))) if len(actual) else 0.0
        correlation = float(pd.Series(predicted).corr(pd.Series(actual))) if len(actual) > 1 else 0.0
        if not np.isfinite(correlation):
            correlation = 0.0
        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
            "correlation": correlation,
        }

    def _score_latest_snapshot(
        self,
        *,
        model,
        frame: pd.DataFrame,
        feature_columns: list[str],
        top: int,
        sector: str | None,
    ) -> pd.DataFrame:
        snapshot = latest_snapshot(frame)
        if sector is not None:
            snapshot = snapshot[snapshot["sector"] == sector].copy()
        if snapshot.empty:
            return snapshot
        working = snapshot.dropna(subset=feature_columns).copy()
        if working.empty:
            return working
        working["predicted_forward_excess_return"] = model.predict(working[feature_columns])
        return working.sort_values(
            ["predicted_forward_excess_return", "md_volume_30d", "ticker"],
            ascending=[False, False, True],
        ).head(top).reset_index(drop=True)

    def _render_predictions(self, title: str, frame: pd.DataFrame) -> list[str]:
        lines = [f"## {title}", ""]
        if frame.empty:
            lines.append("No predictions.")
            lines.append("")
            return lines
        for row in frame.itertuples(index=False):
            lines.append(f"### {row.ticker}")
            lines.append(f"- sector: {row.sector}")
            lines.append(f"- predicted_forward_excess_return: {float(row.predicted_forward_excess_return):.6f}")
            if "adj_close" in frame.columns:
                lines.append(f"- adj_close: {float(row.adj_close):.2f}")
            lines.append(f"- md_volume_30d: {float(row.md_volume_30d):.0f}")
            lines.append(f"- chart: https://www.tradingview.com/chart/?symbol={row.ticker}")
            lines.append("")
        return lines

    def _render_importance(self, frame: pd.DataFrame) -> list[str]:
        lines = ["## Feature Importance", ""]
        for row in frame.head(20).itertuples(index=False):
            lines.append(f"- {row.feature}: {float(row.gain):.6f}")
        lines.append("")
        return lines
