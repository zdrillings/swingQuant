from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.check_shortlist_oos_reproducibility import _evaluate, _rolling_window_summaries
from src.cli import build_parser
from src.research.shortlist_bakeoff_service import MODEL_FEATURE_COLUMNS
from src.research.shortlist_model_service import ShortlistModelService
from src.research.shortlist_universe import filter_eligible_universe
from src.settings import AppPaths
from src.utils.shortlist_runtime import _passes_runtime_promotion_gate, load_live_shortlist_model_context
from src.utils.performance_metrics import annualized_sharpe


class ShortlistModelServiceTests(unittest.TestCase):
    def _gate_row(
        self,
        model: str,
        *,
        hit_rate: float = 0.55,
        universe_hit_rate: float = 0.50,
        mean_target: float = 0.03,
        universe_mean_target: float = 0.01,
        beat_universe_rate: float = 0.60,
        spearman: float = 0.02,
        top_ticker_date_rate: float = 0.20,
    ) -> dict:
        return {
            "model": model,
            "hit_rate": hit_rate,
            "universe_hit_rate": universe_hit_rate,
            "hit_rate_excess": hit_rate - universe_hit_rate,
            "mean_target": mean_target,
            "universe_mean_target": universe_mean_target,
            "mean_target_excess": mean_target - universe_mean_target,
            "beat_universe_rate": beat_universe_rate,
            "spearman": spearman,
            "top_ticker_date_rate": top_ticker_date_rate,
        }

    def test_oos_reproducibility_helpers_apply_costs_and_acceptance_windows(self) -> None:
        rows = []
        for date_index, snapshot_date in enumerate(pd.bdate_range("2026-01-02", periods=65)):
            rows.extend(
                [
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": "AAA",
                        "model_name": "ridge_model",
                        "predicted_alpha": 0.30,
                        "alpha_vs_sector_20d": 0.02 + date_index * 0.0001,
                    },
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": "BBB",
                        "model_name": "ridge_model",
                        "predicted_alpha": 0.20,
                        "alpha_vs_sector_20d": 0.01,
                    },
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": "CCC",
                        "model_name": "ridge_model",
                        "predicted_alpha": 0.10,
                        "alpha_vs_sector_20d": -0.02,
                    },
                ]
            )
        frame = pd.DataFrame(rows)

        summary = _evaluate(frame, target_column="alpha_vs_sector_20d", cost_fraction=0.001)
        windows = dict(
            _rolling_window_summaries(
                frame,
                model_name="ridge_model",
                target_column="alpha_vs_sector_20d",
                cost_fraction=0.001,
            )
        )

        self.assertAlmostEqual(summary["mean_target"], summary["gross_mean_target"] - 0.001, places=9)
        self.assertIn("ridge_model_20d", windows)
        self.assertIn("ridge_model_40d", windows)
        self.assertIn("ridge_model_60d", windows)
        self.assertIn("ridge_model_last_1fold", windows)
        self.assertIn("ridge_model_last_3fold", windows)
        self.assertEqual(windows["ridge_model_20d"]["dates"], 20)

    def test_twenty_day_basket_sharpe_uses_horizon_frequency(self) -> None:
        values = pd.Series([0.02, 0.01, -0.01, 0.03, 0.00])

        horizon_sharpe = annualized_sharpe(values, periods_per_year=252 / 20)
        daily_sharpe = annualized_sharpe(values, periods_per_year=252)

        self.assertLess(horizon_sharpe, daily_sharpe / 3.0)

    def test_shortlist_evaluation_winsorizes_extreme_acceptance_targets(self) -> None:
        service = ShortlistModelService(db_manager=object())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": pd.Timestamp("2026-01-02"),
                    "ticker": "AAA",
                    "sector": "Energy",
                    "predicted_alpha": 2.0,
                    "alpha_vs_sector_20d": 5.0,
                },
                {
                    "snapshot_date": pd.Timestamp("2026-01-02"),
                    "ticker": "BBB",
                    "sector": "Energy",
                    "predicted_alpha": 1.0,
                    "alpha_vs_sector_20d": -5.0,
                },
            ]
        )

        summary = service._evaluate_predictions(
            predictions=frame,
            top_n=2,
            target_column="alpha_vs_sector_20d",
            model_name="ridge_model",
        )

        self.assertAlmostEqual(summary["gross_mean_target"], 0.0)

    def test_service_acceptance_windows_use_fold_labels_and_full_oos(self) -> None:
        service = ShortlistModelService(db_manager=object())
        rows = []
        for date_index, snapshot_date in enumerate(pd.bdate_range("2026-01-02", periods=65)):
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "ticker": "AAA",
                    "sector": "Energy",
                    "predicted_alpha": 0.30,
                    "alpha_vs_sector_20d": 0.02 + date_index * 0.0001,
                }
            )
        summaries = service._rolling_window_summaries(
            predictions=pd.DataFrame(rows),
            target_column="alpha_vs_sector_20d",
            model_name="ridge_model",
            top_n=1,
            windows=service._promotion_recent_windows(horizon_days=60),
            fold_windows=service._promotion_fold_windows(horizon_days=60),
            fold_size=20,
            include_full_oos=True,
        )
        windows = {str(row["model"]): row for row in summaries.to_dict(orient="records")}

        self.assertEqual(service._promotion_recent_windows(horizon_days=60), ())
        self.assertEqual(service._promotion_fold_windows(horizon_days=60), (1, 3))
        self.assertIn("ridge_model_last_fold", windows)
        self.assertIn("ridge_model_trailing_3folds", windows)
        self.assertIn("ridge_model_full_oos", windows)
        self.assertEqual(windows["ridge_model_last_fold"]["dates"], 20)
        self.assertEqual(windows["ridge_model_trailing_3folds"]["dates"], 60)
        self.assertEqual(windows["ridge_model_full_oos"]["dates"], 65)

    def test_regime_matching_enabled_false_disables_matching(self) -> None:
        service = ShortlistModelService(db_manager=object())

        with patch(
            "src.research.shortlist_model_service.load_feature_config",
            return_value={"scan_policy": {"shortlist_model": {"regime_matching_enabled": False}}},
        ):
            self.assertEqual(service._load_regime_matching_mode(), "off")

    def test_regime_conditional_score_flip_uses_lagged_reversal_classification(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=8)

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    [
                        {"snapshot_date": dates[0], "classification": "neutral"},
                        {"snapshot_date": dates[2], "classification": "reversal"},
                        {"snapshot_date": dates[5], "classification": "trending"},
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return [snapshot_date.strftime("%Y-%m-%d") for snapshot_date in dates]

        service = ShortlistModelService(db_manager=FakeDB())
        frame = pd.DataFrame(
            [
                {"snapshot_date": dates[3], "ticker": "AAA", "predicted_alpha": 0.10},
                {"snapshot_date": dates[5], "ticker": "BBB", "predicted_alpha": 0.20},
                {"snapshot_date": dates[7], "ticker": "CCC", "predicted_alpha": 0.30},
            ]
        )

        flipped = service._apply_regime_conditional_score_flip(frame, horizon_sessions=2)

        self.assertEqual(flipped.loc[0, "regime_classification"], "neutral")
        self.assertFalse(bool(flipped.loc[0, "regime_flip_applied"]))
        self.assertEqual(flipped.loc[1, "regime_classification"], "reversal")
        self.assertTrue(bool(flipped.loc[1, "regime_flip_applied"]))
        self.assertAlmostEqual(float(flipped.loc[1, "predicted_alpha"]), -0.20)
        self.assertAlmostEqual(float(flipped.loc[1, "raw_predicted_alpha"]), 0.20)
        self.assertEqual(flipped.loc[2, "regime_classification"], "trending")
        self.assertFalse(bool(flipped.loc[2, "regime_flip_applied"]))

        matched_frame = frame.iloc[[1]].copy()
        matched_frame["regime_matched_training_applied"] = True
        matched = service._apply_regime_conditional_score_flip(
            matched_frame,
            horizon_sessions=2,
            fallback_only=True,
        )
        self.assertFalse(bool(matched.loc[1, "regime_flip_applied"]))
        self.assertAlmostEqual(float(matched.loc[1, "predicted_alpha"]), 0.20)

    def test_walk_forward_predictions_use_regime_matched_training_rows(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=80)

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    [
                        {
                            "snapshot_date": snapshot_date,
                            "classification": "reversal" if index < 70 else "trending",
                        }
                        for index, snapshot_date in enumerate(dates)
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return [snapshot_date.strftime("%Y-%m-%d") for snapshot_date in dates]

        service = ShortlistModelService(db_manager=FakeDB())
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        observed_train_dates: list[pd.Timestamp] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            test_frame = kwargs["test_frame"]
            observed_train_dates.extend(sorted(pd.to_datetime(train_frame["snapshot_date"]).drop_duplicates().tolist()))
            scored = test_frame.copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        stats = {
            "attempted_folds": 0,
            "matched_folds": 0,
            "fallback_folds": 0,
            "unknown_folds": 0,
            "live_matched": 0,
            "live_fallback": 0,
        }
        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=65,
                max_train_dates=70,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=3,
                label_horizon_dates=0,
                regime_matching_mode="train_only",
                min_regime_train_dates=120,
                regime_matching_stats=stats,
            )

        self.assertIsNotNone(predictions)
        self.assertEqual(stats["matched_folds"], 2)
        self.assertTrue(bool(predictions["regime_matched_training_applied"].any()))
        self.assertEqual(observed_train_dates[:65], list(dates[:65]))

    def test_regime_matching_uses_density_floor_before_max_train_cap(self) -> None:
        dates = list(pd.bdate_range("2026-01-02", periods=130))
        service = ShortlistModelService(db_manager=object())
        regime_by_date = {
            pd.Timestamp(date_value).normalize(): "reversal" if index < 65 else "trending"
            for index, date_value in enumerate(dates)
        }
        stats = {
            "attempted_folds": 0,
            "matched_folds": 0,
            "fallback_folds": 0,
            "unknown_folds": 0,
            "live_matched": 0,
            "live_fallback": 0,
        }

        selected, matched, test_regime, feature_screen_dates = service._regime_matched_train_dates(
            train_date_window=dates[:65],
            test_dates=[dates[64]],
            min_train_dates=120,
            max_train_dates=20,
            regime_by_date=regime_by_date,
            mode="train_and_flip",
            stats=stats,
        )

        self.assertTrue(matched)
        self.assertEqual(test_regime, "reversal")
        self.assertEqual(stats["matched_folds"], 1)
        self.assertEqual(selected, dates[45:65])
        self.assertEqual(feature_screen_dates, dates[:65])

    def test_regime_matching_density_floor_falls_back_below_sixty_same_regime_dates(self) -> None:
        dates = list(pd.bdate_range("2026-01-02", periods=80))
        service = ShortlistModelService(db_manager=object())
        regime_by_date = {
            pd.Timestamp(date_value).normalize(): "reversal" if index < 59 else "trending"
            for index, date_value in enumerate(dates)
        }
        stats = {
            "attempted_folds": 0,
            "matched_folds": 0,
            "fallback_folds": 0,
            "unknown_folds": 0,
            "live_matched": 0,
            "live_fallback": 0,
        }

        selected, matched, test_regime, feature_screen_dates = service._regime_matched_train_dates(
            train_date_window=dates[:59],
            test_dates=[dates[58]],
            min_train_dates=120,
            max_train_dates=20,
            regime_by_date=regime_by_date,
            mode="train_and_flip",
            stats=stats,
        )

        self.assertFalse(matched)
        self.assertEqual(test_regime, "reversal")
        self.assertEqual(stats["fallback_folds"], 1)
        self.assertEqual(selected, dates[39:59])
        self.assertEqual(feature_screen_dates, dates[:59])

    def test_regime_transition_purge_modes_remove_expected_training_dates(self) -> None:
        dates = list(pd.bdate_range("2026-01-02", periods=8))
        service = ShortlistModelService(db_manager=object())
        regime_by_date = {
            pd.Timestamp(date_value).normalize(): regime
            for date_value, regime in zip(
                dates,
                ["neutral", "neutral", "reversal", "reversal", "reversal", "neutral", "neutral", "neutral"],
            )
        }

        strict = service._purge_regime_transition_dates(
            dates[:4],
            all_dates=dates,
            horizon_sessions=3,
            regime_by_date=regime_by_date,
            mode="strict",
        )
        majority = service._purge_regime_transition_dates(
            dates[:4],
            all_dates=dates,
            horizon_sessions=3,
            regime_by_date=regime_by_date,
            mode="majority",
        )
        off = service._purge_regime_transition_dates(
            dates[:4],
            all_dates=dates,
            horizon_sessions=3,
            regime_by_date=regime_by_date,
            mode="off",
        )

        self.assertEqual(strict, [])
        self.assertEqual(majority, [dates[2]])
        self.assertEqual(off, dates[:4])

    def test_contamination_decomposition_does_not_change_gate_outcome(self) -> None:
        dates = list(pd.bdate_range("2026-05-01", periods=6))

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    {
                        "snapshot_date": dates,
                        "classification": ["neutral", "neutral", "neutral", "reversal", "reversal", "reversal"],
                    }
                )

            def list_universe_daily_snapshot_dates(self):
                return dates

        service = ShortlistModelService(db_manager=FakeDB())
        summaries = pd.DataFrame(
            [
                dict(
                    self._gate_row("xgboost_model_last_fold", hit_rate=0.75, universe_hit_rate=0.50, mean_target=0.09, universe_mean_target=0.01, beat_universe_rate=0.75, spearman=0.10, top_ticker_date_rate=0.25),
                    dates=2,
                    avg_pick_count=2.0,
                    gross_mean_target=0.10,
                    positive_date_rate=0.75,
                    ge_2pct_rate=0.75,
                    ge_5pct_rate=0.50,
                ),
                dict(
                    self._gate_row("xgboost_model_full_oos", hit_rate=0.75, universe_hit_rate=0.50, mean_target=0.09, universe_mean_target=0.01, beat_universe_rate=0.75, spearman=0.10, top_ticker_date_rate=0.25),
                    dates=6,
                    avg_pick_count=2.0,
                    gross_mean_target=0.10,
                    positive_date_rate=0.75,
                    ge_2pct_rate=0.75,
                    ge_5pct_rate=0.50,
                ),
            ]
        )
        gate = {
            "enabled": True,
            "min_recent_1fold_hit_rate_excess": 0.02,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "max_recent_1fold_top_ticker_date_rate": 0.40,
        }
        predictions = pd.DataFrame(
            [
                {
                    "snapshot_date": date_value,
                    "ticker": f"AAA{index}",
                    "model_name": "xgboost_model",
                    "predicted_alpha": 0.1,
                    "alpha_vs_sector_20d": 0.1,
                }
                for index, date_value in enumerate(dates)
            ]
        )

        before = service._model_passes_promotion_gate(
            model_name="xgboost_model",
            acceptance_summaries=summaries,
            promotion_gate=gate,
            required_recent_windows=(),
            required_fold_windows=(1,),
        )
        lines = service._render_regime_contamination_decomposition(
            predictions=predictions,
            summaries=summaries,
            horizon_sessions=20,
            fold_size=2,
            required_recent_windows=(),
            required_fold_windows=(1,),
        )
        after = service._model_passes_promotion_gate(
            model_name="xgboost_model",
            acceptance_summaries=summaries,
            promotion_gate=gate,
            required_recent_windows=(),
            required_fold_windows=(1,),
        )

        self.assertTrue(before)
        self.assertEqual(before, after)
        self.assertIn("diagnostic only", "\n".join(lines))
        self.assertIn("xgboost_model_last_fold", "\n".join(lines))

    def test_promotion_gate_uses_excess_hit_and_mean_floors(self) -> None:
        service = ShortlistModelService(db_manager=object())
        gate = {
            "enabled": True,
            "min_recent_1fold_hit_rate_excess": 0.02,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "max_recent_1fold_top_ticker_date_rate": 0.40,
        }

        at_base = pd.DataFrame(
            [
                self._gate_row("ridge_model_last_fold", hit_rate=0.43, universe_hit_rate=0.43, mean_target=0.01, universe_mean_target=0.01),
                self._gate_row("ridge_model_full_oos"),
            ]
        )
        self.assertFalse(
            service._model_passes_promotion_gate(
                model_name="ridge_model",
                acceptance_summaries=at_base,
                promotion_gate=gate,
                required_recent_windows=(),
                required_fold_windows=(1,),
            )
        )

        boundary = pd.DataFrame(
            [
                self._gate_row("ridge_model_last_fold", hit_rate=0.45, universe_hit_rate=0.43, mean_target=0.01, universe_mean_target=0.01),
                self._gate_row("ridge_model_full_oos"),
            ]
        )
        self.assertTrue(
            service._model_passes_promotion_gate(
                model_name="ridge_model",
                acceptance_summaries=boundary,
                promotion_gate=gate,
                required_recent_windows=(),
                required_fold_windows=(1,),
            )
        )

        negative_mean_excess = pd.DataFrame(
            [
                self._gate_row("ridge_model_last_fold", hit_rate=0.46, universe_hit_rate=0.43, mean_target=0.00, universe_mean_target=0.01),
                self._gate_row("ridge_model_full_oos"),
            ]
        )
        self.assertFalse(
            service._model_passes_promotion_gate(
                model_name="ridge_model",
                acceptance_summaries=negative_mean_excess,
                promotion_gate=gate,
                required_recent_windows=(),
                required_fold_windows=(1,),
            )
        )

    def test_promotion_gate_fails_closed_without_universe_excess_columns(self) -> None:
        service = ShortlistModelService(db_manager=object())
        summaries = pd.DataFrame(
            [
                {"model": "ridge_model_last_fold", "hit_rate": 0.90, "mean_target": 0.20, "beat_universe_rate": 1.0, "spearman": 0.50},
                {"model": "ridge_model_full_oos", "hit_rate": 0.90, "mean_target": 0.20, "beat_universe_rate": 1.0, "spearman": 0.50},
            ]
        )
        gate = {
            "enabled": True,
            "min_recent_1fold_hit_rate_excess": 0.02,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "max_recent_1fold_top_ticker_date_rate": 0.40,
        }

        self.assertFalse(
            service._model_passes_promotion_gate(
                model_name="ridge_model",
                acceptance_summaries=summaries,
                promotion_gate=gate,
                required_recent_windows=(),
                required_fold_windows=(1,),
            )
        )

    def test_runtime_gate_mirrors_excess_promotion_gate(self) -> None:
        metrics = {
            60: {
                "hit_rate": 0.45,
                "universe_hit_rate": 0.43,
                "hit_rate_excess": 0.02,
                "mean_target": 0.01,
                "universe_mean_target": 0.01,
                "mean_target_excess": 0.0,
                "beat_rate": 0.50,
                "spearman": 0.0,
            },
            3: {
                "hit_rate": 0.45,
                "universe_hit_rate": 0.43,
                "hit_rate_excess": 0.02,
                "mean_target": 0.01,
                "universe_mean_target": 0.01,
                "mean_target_excess": 0.0,
                "beat_rate": 0.50,
                "spearman": 0.0,
            }
        }

        self.assertTrue(_passes_runtime_promotion_gate(recent_metrics=metrics, horizon_days=60))
        failing = {60: dict(metrics[60]), 3: dict(metrics[3], hit_rate_excess=0.0)}
        self.assertFalse(_passes_runtime_promotion_gate(recent_metrics=failing, horizon_days=60))
        missing = {
            60: dict(metrics[60]),
            3: {"hit_rate": 0.90, "mean_target": 0.20, "beat_rate": 1.0, "spearman": 0.50},
        }
        self.assertFalse(_passes_runtime_promotion_gate(recent_metrics=missing, horizon_days=60))

    def test_report_renders_excess_gate_metrics(self) -> None:
        service = ShortlistModelService(db_manager=object())
        summaries = pd.DataFrame(
            [
                dict(
                    self._gate_row("ridge_model_last_fold", hit_rate=0.45, universe_hit_rate=0.43, mean_target=0.01, universe_mean_target=0.01),
                    dates=20,
                    avg_pick_count=2.0,
                    gross_mean_target=0.011,
                    round_trip_cost=0.001,
                    positive_date_rate=0.50,
                    ge_2pct_rate=0.25,
                    ge_5pct_rate=0.10,
                    top_ticker="AAA",
                    top_ticker_pick_share=0.10,
                    net_sharpe=0.20,
                    newey_west_t=0.30,
                    years_for_t_1_96=10.0,
                )
            ]
        )
        gate = service._load_promotion_gate()

        text = "\n".join(
            service._render_promotion_gate(
                promotion_gate=gate,
                summaries=summaries,
                required_recent_windows=(),
                required_fold_windows=(1,),
            )
        )

        self.assertIn("gate_note: floors are excess-over-universe", text)
        self.assertIn("min_recent_1fold_hit_rate_excess", text)
        self.assertIn("- hit_rate: 0.450000", text)
        self.assertIn("- universe_hit_rate: 0.430000", text)
        self.assertIn("- hit_rate_excess: 0.020000", text)
        self.assertIn("- net_mean_target: 0.010000", text)
        self.assertIn("- universe_mean_target: 0.010000", text)
        self.assertIn("- mean_target_excess: 0.000000", text)

    def test_reversal_rules_rank_pullbacks_on_reversal_dates(self) -> None:
        service = ShortlistModelService(db_manager=object())
        date = pd.Timestamp("2026-02-03")
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": date,
                    "ticker": "LOW",
                    "sector": "Energy",
                    "regime_matching_test_regime": "reversal",
                    "roc_63": -0.20,
                    "rsi_14": 35.0,
                    "close_vs_20d_low": 0.01,
                    "sma_50_dist": -0.15,
                    "relative_strength_index_vs_spy": 30.0,
                    "sma_200_dist": -0.05,
                    "vol_alpha": 0.8,
                },
                {
                    "snapshot_date": date,
                    "ticker": "HIGH",
                    "sector": "Energy",
                    "regime_matching_test_regime": "reversal",
                    "roc_63": 0.30,
                    "rsi_14": 72.0,
                    "close_vs_20d_low": 0.25,
                    "sma_50_dist": 0.12,
                    "relative_strength_index_vs_spy": 90.0,
                    "sma_200_dist": 0.20,
                    "vol_alpha": 1.5,
                },
            ]
        )

        scored = service._score_reversal_rules(frame)

        low_score = float(scored.loc[scored["ticker"] == "LOW", "predicted_alpha"].iloc[0])
        high_score = float(scored.loc[scored["ticker"] == "HIGH", "predicted_alpha"].iloc[0])
        self.assertGreater(low_score, high_score)

    def test_reversal_fold_feature_screen_uses_matched_pool(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=12)

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    [
                        {
                            "snapshot_date": snapshot_date,
                            "classification": "reversal" if index % 2 else "trending",
                        }
                        for index, snapshot_date in enumerate(dates)
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return [snapshot_date.strftime("%Y-%m-%d") for snapshot_date in dates]

        service = ShortlistModelService(db_manager=FakeDB())
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker_index, ticker in enumerate(("AAA", "BBB")):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 50.0 + ticker_index,
                        "roc_63": 0.01 * ticker_index,
                        "sma_200_dist": 0.01,
                        "vol_alpha": 1.0,
                        "rsi_2": float(ticker_index),
                        "alpha_vs_sector_20d": 0.01 * ticker_index,
                    }
                )
        screened_dates: list[pd.Timestamp] = []

        def fake_screen(frame, **_kwargs):
            screened_dates.extend(pd.to_datetime(frame["snapshot_date"]).drop_duplicates().tolist())
            return ["rsi_2"]

        def fake_score_model(**kwargs):
            scored = kwargs["test_frame"].copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_feature_ic_survivors_from_frame", side_effect=fake_screen), \
             patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                pd.DataFrame(rows),
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=4,
                max_train_dates=4,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=4,
                label_horizon_dates=1,
                min_feature_ic=0.01,
                min_regime_train_dates=2,
                regime_matching_mode="train_only",
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(screened_dates)
        expected_screen_dates = set(pd.to_datetime(list(dates[2:7:2])).normalize())
        observed_screen_dates = set(pd.to_datetime(screened_dates).normalize())
        self.assertTrue(observed_screen_dates.issubset(expected_screen_dates))

    def test_reversal_universe_extension_uses_lagged_regime_without_lookahead(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=25)

        class FakeDB:
            def load_regime_meter(self):
                return pd.DataFrame(
                    [
                        {"snapshot_date": dates[0], "classification": "reversal"},
                        {"snapshot_date": dates[4], "classification": "trending"},
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return [snapshot_date.strftime("%Y-%m-%d") for snapshot_date in dates]

        service = ShortlistModelService(db_manager=FakeDB())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": dates[21],
                    "ticker": "PULL",
                    "sector": "Energy",
                    "passed_any_strategy": False,
                    "passed_slots_json": "[]",
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 30.0,
                    "sma_200_dist": -0.05,
                    "roc_63": -0.02,
                    "rsi_14": 45.0,
                    "relative_strength_index_vs_spy": 30.0,
                    "regime_green": False,
                    "alpha_vs_sector_20d": 0.01,
                },
                {
                    "snapshot_date": dates[24],
                    "ticker": "LOOKAHEAD",
                    "sector": "Energy",
                    "passed_any_strategy": False,
                    "passed_slots_json": "[]",
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 30.0,
                    "sma_200_dist": -0.05,
                    "roc_63": -0.02,
                    "rsi_14": 45.0,
                    "relative_strength_index_vs_spy": 30.0,
                    "regime_green": False,
                    "alpha_vs_sector_20d": 0.01,
                },
            ]
        )

        eligible = service._build_matured_eligible_universe(
            frame,
            target_column="alpha_vs_sector_20d",
            eligible_universe_mode="passed_or_trend",
        )

        self.assertIn("PULL", set(eligible["ticker"]))
        self.assertNotIn("LOOKAHEAD", set(eligible["ticker"]))

    def test_report_renders_regime_feature_survivor_summary(self) -> None:
        service = ShortlistModelService(db_manager=object())

        lines = service._render_regime_feature_survivors(
            {
                "reversal": {
                    "folds": 2,
                    "features": {"rsi_2": 2, "ret_1d": 1, "roc_63": 1},
                }
            }
        )
        text = "\n".join(lines)

        self.assertIn("## Surviving Features By Fold Regime", text)
        self.assertIn("### reversal", text)
        self.assertIn("- reversal_core_survivors: ret_1d, rsi_2", text)

    def test_filter_eligible_universe_passed_or_trend_broadens_research_set(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                    "regime_green": 0,
                    "sma_200_dist": -0.02,
                    "roc_63": -0.01,
                    "relative_strength_index_vs_spy": 40.0,
                },
                {
                    "ticker": "BBB",
                    "passed_any_strategy": 0,
                    "md_volume_30d": 35_000_000.0,
                    "adj_close": 90.0,
                    "regime_green": 1,
                    "sma_200_dist": 0.10,
                    "roc_63": 0.08,
                    "relative_strength_index_vs_spy": 68.0,
                },
                {
                    "ticker": "CCC",
                    "passed_any_strategy": 0,
                    "md_volume_30d": 35_000_000.0,
                    "adj_close": 90.0,
                    "regime_green": 1,
                    "sma_200_dist": -0.01,
                    "roc_63": 0.08,
                    "relative_strength_index_vs_spy": 68.0,
                },
            ]
        )

        passed_only = filter_eligible_universe(frame, eligible_universe_mode="passed_only")
        passed_or_trend = filter_eligible_universe(frame, eligible_universe_mode="passed_or_trend")

        self.assertEqual(sorted(passed_only["ticker"].tolist()), ["AAA"])
        self.assertEqual(sorted(passed_or_trend["ticker"].tolist()), ["AAA", "BBB"])

    def test_filter_eligible_universe_prefers_historical_passed_slots_json(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "passed_any_strategy": 0,
                    "passed_slots_json": '["energy"]',
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                },
                {
                    "ticker": "BBB",
                    "passed_any_strategy": 1,
                    "passed_slots_json": "[]",
                    "md_volume_30d": 30_000_000.0,
                    "adj_close": 100.0,
                },
            ]
        )

        passed_only = filter_eligible_universe(frame, eligible_universe_mode="passed_only")

        self.assertEqual(passed_only["ticker"].tolist(), ["AAA"])

    def test_model_reason_summary_uses_relative_language(self) -> None:
        service = ShortlistModelService(db_manager=object())

        reasons = service._top_reason_names(
            {
                "roc_63__rank_all": 0.94,
                "relative_strength_index_vs_spy": 1.8,
                "sma_200_dist": 0.7,
            }
        )

        self.assertIn("top-tier 63d momentum", reasons)
        self.assertIn("strong RS vs SPY", reasons)
        self.assertIn("well above 200d trend", reasons)
        self.assertEqual(
            service._format_reason_summary(reasons),
            "strong RS vs SPY, top-tier 63d momentum, well above 200d trend",
        )

    def test_shortlist_model_writes_walk_forward_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)

            dates = pd.bdate_range("2026-01-02", periods=32)
            tickers = [
                ("AAA", "Energy"),
                ("BBB", "Materials"),
                ("CCC", "Industrials"),
                ("DDD", "Information Technology"),
            ]
            rows: list[dict[str, object]] = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, (ticker, sector) in enumerate(tickers):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": sector,
                        "passed_any_strategy": 1,
                        "md_volume_30d": 60_000_000.0,
                        "adj_close": 100.0 + ticker_index,
                        "alpha_vs_sector_20d": None if date_index == len(dates) - 1 else 0.02 * (ticker_index + 1) + 0.001 * date_index,
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float((feature_index + 1) * 0.05 + ticker_index + date_index * 0.03)
                    rows.append(row)
            snapshot_frame = pd.DataFrame(rows)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame
                    self.run_rows: list[dict[str, object]] = []
                    self.prediction_rows: list[dict[str, object]] = []

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()
                def insert_shortlist_model_run(self, *, row):
                    self.run_rows.append(dict(row))
                    return len(self.run_rows)
                def replace_shortlist_model_predictions(
                    self,
                    *,
                    generated_at,
                    horizon_days,
                    eligible_universe_mode="passed_only",
                    model_scope="global",
                    rows,
                ):
                    self.prediction_rows = [dict(row) for row in rows]
                    return len(self.prediction_rows)

            fake_db = FakeDB(paths, snapshot_frame)
            service = ShortlistModelService(fake_db)
            config = {
                "scan_policy": {
                    "shortlist_model": {
                        "min_feature_ic": 0.0,
                        "promotion_gate": {
                            "enabled": True,
                            "min_recent_20d_hit_rate_excess": -1.0,
                            "min_recent_20d_beat_universe_rate": 0.0,
                            "min_recent_20d_mean_target_excess": -1.0,
                            "min_recent_60d_hit_rate_excess": -1.0,
                            "min_recent_60d_beat_universe_rate": 0.0,
                            "min_recent_60d_mean_target_excess": -1.0,
                            "min_recent_1fold_hit_rate_excess": -1.0,
                            "min_recent_1fold_beat_universe_rate": 0.0,
                            "min_recent_1fold_mean_target_excess": -1.0,
                            "min_recent_3fold_hit_rate_excess": -1.0,
                            "min_recent_3fold_beat_universe_rate": 0.0,
                            "min_recent_3fold_mean_target_excess": -1.0,
                            "min_recent_20d_spearman": -1.0,
                            "min_recent_60d_spearman": -1.0,
                            "min_recent_1fold_spearman": -1.0,
                            "min_recent_3fold_spearman": -1.0,
                            "max_recent_20d_top_ticker_date_rate": 1.0,
                            "max_recent_60d_top_ticker_date_rate": 1.0,
                            "max_recent_1fold_top_ticker_date_rate": 1.0,
                            "max_recent_3fold_top_ticker_date_rate": 1.0,
                        },
                    },
                },
            }
            with patch("src.research.shortlist_model_service.load_feature_config", return_value=config):
                report = service.run(
                    top_n=2,
                    horizon_days=20,
                    min_train_dates=6,
                    test_window_dates=2,
                    recent_dates=4,
                    xgboost_config="balanced_depth4",
                )

            self.assertEqual(report.target_column, "alpha_vs_sector_20d")
            self.assertGreater(report.oos_dates, 0)
            self.assertGreater(report.live_candidates, 0)

            report_text = (paths.reports_dir / "shortlist_model.md").read_text(encoding="utf-8")
            self.assertIn("# Shortlist Model", report_text)
            self.assertIn("- eligible_universe_mode: passed_only", report_text)
            self.assertIn("- live_output_top_n: 2", report_text)
            self.assertIn("- promotion_top_n: 2", report_text)
            self.assertIn("- candidate_models:", report_text)
            self.assertIn("reversal_rules", report_text)
            self.assertIn("- selected_model:", report_text)
            self.assertIn("## Regime Matching", report_text)
            self.assertIn("- regime_matching_folds: attempted=", report_text)
            self.assertIn("- attempted_folds:", report_text)
            self.assertIn("- matched_folds:", report_text)
            self.assertIn("- fallback_folds:", report_text)
            self.assertIn("- unknown_folds:", report_text)
            self.assertIn("## Surviving Features By Fold Regime", report_text)
            self.assertIn("## Promotion Gate", report_text)
            self.assertIn("## Full Walk-Forward Evaluation", report_text)
            self.assertIn("## Live Top Candidates", report_text)
            self.assertIn("### signal_proxy", report_text)
            self.assertIn("### lasso_model", report_text)

            self.assertTrue((paths.reports_dir / "shortlist_model_oos_predictions.csv").exists())
            self.assertTrue((paths.reports_dir / "shortlist_model_live_predictions.csv").exists())
            oos_predictions = pd.read_csv(paths.reports_dir / "shortlist_model_oos_predictions.csv")
            self.assertIn("model_rank", oos_predictions.columns)
            self.assertIn("signal_proxy_rank", oos_predictions.columns)
            self.assertIn("ensemble_model_rank", oos_predictions.columns)
            signal_rows = oos_predictions[oos_predictions["model_name"] == "signal_proxy"]
            ensemble_rows = oos_predictions[oos_predictions["model_name"] == "ensemble_model"]
            self.assertTrue(signal_rows["model_rank"].notna().all())
            self.assertTrue(signal_rows["signal_proxy_rank"].notna().all())
            self.assertTrue(ensemble_rows["model_rank"].notna().all())
            self.assertTrue(ensemble_rows["ensemble_model_rank"].notna().all())
            self.assertEqual(len(fake_db.run_rows), 1)
            self.assertGreater(len(fake_db.prediction_rows), 0)

    def test_shortlist_model_parser_accepts_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "shortlist-model",
                "--top",
                "8",
                "--horizon",
                "20",
                "--min-train-dates",
                "200",
                "--max-train-dates",
                "252",
                "--test-window-dates",
                "15",
                "--oos-stride-dates",
                "60",
                "--recent-dates",
                "30",
                "--eligible-universe-mode",
                "passed_or_trend",
                "--model-scope",
                "sector_specific",
                "--xgboost-config",
                "faster_shallow",
                "--feature-profile",
                "no_gap_risk",
            ]
        )
        self.assertEqual(args.command, "shortlist-model")
        self.assertEqual(args.top, 8)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.min_train_dates, 200)
        self.assertEqual(args.max_train_dates, 252)
        self.assertEqual(args.test_window_dates, 15)
        self.assertEqual(args.oos_stride_dates, 60)
        self.assertEqual(args.recent_dates, 30)
        self.assertEqual(args.eligible_universe_mode, "passed_or_trend")
        self.assertEqual(args.model_scope, "sector_specific")
        self.assertEqual(args.xgboost_config, "faster_shallow")
        self.assertEqual(args.feature_profile, "no_gap_risk")
        self.assertFalse(args.dry_run)

        dry_run_args = parser.parse_args(["shortlist-model", "--dry-run"])
        self.assertTrue(dry_run_args.dry_run)
        path_args = parser.parse_args(["shortlist-model", "--target-type", "path"])
        self.assertEqual(path_args.target_type, "path")

    def test_runtime_loader_returns_lasso_model_context(self) -> None:
        captured: dict[str, object] = {}

        class FakeDB:
            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, limit=1):
                captured["runs_model_scope"] = model_scope
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "target_column": "alpha_vs_sector_20d_pos",
                            "live_snapshot_date": "2026-05-19",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-19"]

            def load_shortlist_model_predictions(self, *, generated_at, horizon_days, eligible_universe_mode=None, model_scope=None, dataset_split, model_name):
                captured.setdefault("prediction_model_scopes", []).append(model_scope)
                if model_name != "lasso_model":
                    return pd.DataFrame()
                if dataset_split == "oos":
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": 0.11,
                                "actual_alpha_vs_sector": 0.04,
                            },
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "BBB",
                                "sector": "Energy",
                                "md_volume_30d": 40_000_000.0,
                                "predicted_alpha": 0.08,
                                "actual_alpha_vs_sector": 0.01,
                            },
                            {
                                "snapshot_date": "2026-05-18",
                                "ticker": "CCC",
                                "sector": "Energy",
                                "md_volume_30d": 30_000_000.0,
                                "predicted_alpha": 0.02,
                                "actual_alpha_vs_sector": -0.04,
                            },
                        ]
                    )
                return pd.DataFrame(
                    [
                        {
                            "snapshot_date": "2026-05-19",
                            "ticker": "AAA",
                            "sector": "Energy",
                            "md_volume_30d": 50_000_000.0,
                            "predicted_alpha": 0.11,
                            "details_json": '{"model_top_reasons": ["strong 63d momentum", "strong RS vs SPY"], "model_reason_summary": "strong 63d momentum, strong RS vs SPY"}',
                        },
                        {
                            "snapshot_date": "2026-05-19",
                            "ticker": "BBB",
                            "sector": "Energy",
                            "md_volume_30d": 40_000_000.0,
                            "predicted_alpha": 0.08,
                            "details_json": '{"model_top_reasons": ["strong RS vs SPY"], "model_reason_summary": "strong RS vs SPY"}',
                        }
                    ]
                )

        context = load_live_shortlist_model_context(
            FakeDB(),
            top_n=1,
            refresh_if_stale=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(captured["runs_model_scope"], "sector_specific")
        self.assertEqual(context.champion_model, "lasso_model")
        self.assertEqual(context.target_column, "alpha_vs_sector_20d_pos")
        self.assertEqual(context.live_predictions.iloc[0]["ticker"], "AAA")
        self.assertEqual(
            context.live_predictions.iloc[0]["model_comparison_summary"],
            "BBB in Energy on strong 63d momentum",
        )

    def test_walk_forward_predictions_use_dense_windows_with_sparse_retrain_stride(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=30)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)

        predictions = service._walk_forward_predictions(
            frame,
            target_column="alpha_vs_sector_20d",
            model_name="signal_proxy",
            min_train_dates=5,
            test_window_dates=2,
            model_scope="global",
            evaluation_stride_dates=10,
        )

        self.assertIsNotNone(predictions)
        assert predictions is not None
        self.assertEqual(len(predictions["snapshot_date"].drop_duplicates()), 6)

    def test_walk_forward_predictions_predict_all_dates_between_twenty_day_retrains(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=30)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)

        predictions = service._walk_forward_predictions(
            frame,
            target_column="alpha_vs_sector_20d",
            model_name="signal_proxy",
            min_train_dates=5,
            test_window_dates=20,
            model_scope="global",
            evaluation_stride_dates=20,
        )

        self.assertIsNotNone(predictions)
        assert predictions is not None
        predicted_dates = sorted(pd.to_datetime(predictions["snapshot_date"]).drop_duplicates().tolist())
        self.assertEqual(len(predicted_dates), 25)
        self.assertEqual(predicted_dates[0], dates[5])
        self.assertEqual(predicted_dates[-1], dates[-1])

    def test_walk_forward_predictions_embargo_overlapping_training_labels(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=18)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        seen_folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            test_frame = kwargs["test_frame"]
            seen_folds.append(
                (
                    pd.to_datetime(train_frame["snapshot_date"]).max(),
                    pd.to_datetime(test_frame["snapshot_date"]).min(),
                )
            )
            scored = test_frame.copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=5,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=5,
                label_horizon_dates=5,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(seen_folds)
        for train_end, test_start in seen_folds:
            self.assertLessEqual(
                dates.get_loc(train_end),
                dates.get_loc(test_start) - 5 - 1,
            )

    def test_walk_forward_predictions_strides_training_labels_by_horizon(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=24)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        observed_train_dates: list[list[pd.Timestamp]] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            observed_train_dates.append(sorted(pd.to_datetime(train_frame["snapshot_date"]).drop_duplicates().tolist()))
            scored = kwargs["test_frame"].copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=5,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=5,
                label_horizon_dates=5,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(observed_train_dates)
        for fold_dates in observed_train_dates:
            indexes = [dates.get_loc(date_value) for date_value in fold_dates]
            self.assertTrue(all((right - left) >= 5 for left, right in zip(indexes, indexes[1:])))

    def test_prepare_model_matrices_accepts_rank_feature_ic_survivors(self) -> None:
        service = ShortlistModelService(db_manager=object())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "AAA",
                    "sector": "Energy",
                    "md_volume_30d": 50_000_000.0,
                    "roc_63": 0.10,
                    "relative_strength_index_vs_spy": 70.0,
                },
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "BBB",
                    "sector": "Energy",
                    "md_volume_30d": 50_000_000.0,
                    "roc_63": 0.20,
                    "relative_strength_index_vs_spy": 80.0,
                },
            ]
        )

        _train, _test, feature_names, _standardized = service._prepare_model_matrices(
            frame,
            frame,
            feature_columns_override=["roc_63__rank_all"],
        )

        self.assertIn("roc_63__rank_all", feature_names)
        self.assertNotIn("relative_strength_index_vs_spy", feature_names)

    def test_walk_forward_feature_ic_screen_uses_train_rows_only(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=8)
        tickers = [f"T{index:02d}" for index in range(15)]
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker_index, ticker in enumerate(tickers):
                future_only_signal = float(ticker_index) if date_index >= 4 else 1.0
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": float(ticker_index),
                        "roc_63": future_only_signal,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": float(ticker_index),
                    }
                )
        frame = pd.DataFrame(rows)
        observed_feature_sets: list[list[str]] = []

        def fake_score_model(**kwargs):
            observed_feature_sets.append(list(kwargs["feature_columns_override"]))
            scored = kwargs["test_frame"].copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=4,
                test_window_dates=1,
                model_scope="global",
                evaluation_stride_dates=4,
                min_feature_ic=0.80,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(observed_feature_sets)
        first_fold_features = observed_feature_sets[0]
        self.assertTrue(any(feature.startswith("relative_strength_index_vs_spy") for feature in first_fold_features))
        self.assertFalse(any(feature.startswith("roc_63") for feature in first_fold_features))

    def test_walk_forward_feature_ic_screen_respects_feature_override(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=8)
        tickers = [f"T{index:02d}" for index in range(15)]
        rows = []
        for snapshot_date in dates:
            for ticker_index, ticker in enumerate(tickers):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": float(ticker_index),
                        "roc_63": float(ticker_index),
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": float(ticker_index),
                    }
                )
        frame = pd.DataFrame(rows)
        observed_feature_sets: list[list[str]] = []

        def fake_score_model(**kwargs):
            observed_feature_sets.append(list(kwargs["feature_columns_override"]))
            scored = kwargs["test_frame"].copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=4,
                test_window_dates=1,
                model_scope="global",
                evaluation_stride_dates=4,
                feature_columns_override=["relative_strength_index_vs_spy__rank_all"],
                min_feature_ic=0.80,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(observed_feature_sets)
        self.assertEqual(observed_feature_sets[0], ["relative_strength_index_vs_spy__rank_all"])

    def test_walk_forward_predictions_stride_training_labels_on_horizon_grid(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=25)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        train_date_counts: list[int] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            test_frame = kwargs["test_frame"]
            train_date_counts.append(int(train_frame["snapshot_date"].nunique()))
            scored = test_frame.copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=5,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=5,
                label_horizon_dates=5,
            )

        self.assertIsNotNone(predictions)
        self.assertEqual(train_date_counts, [1, 2, 3])

    def test_walk_forward_predictions_caps_training_window_before_label_stride(self) -> None:
        service = ShortlistModelService(db_manager=object())
        dates = pd.bdate_range("2026-01-02", periods=30)
        rows = []
        for date_index, snapshot_date in enumerate(dates):
            for ticker in ("AAA", "BBB"):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "md_volume_30d": 50_000_000.0,
                        "relative_strength_index_vs_spy": 70.0 + date_index,
                        "roc_63": 0.1,
                        "sma_200_dist": 0.1,
                        "vol_alpha": 1.0,
                        "alpha_vs_sector_20d": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        observed_train_dates: list[list[pd.Timestamp]] = []

        def fake_score_model(**kwargs):
            train_frame = kwargs["train_frame"]
            test_frame = kwargs["test_frame"]
            observed_train_dates.append(sorted(pd.to_datetime(train_frame["snapshot_date"]).drop_duplicates().tolist()))
            scored = test_frame.copy()
            scored["predicted_alpha"] = 0.0
            scored["model_top_reasons"] = [[] for _ in range(len(scored.index))]
            scored["model_reason_summary"] = None
            return scored

        with patch.object(service, "_score_model", side_effect=fake_score_model):
            predictions = service._walk_forward_predictions(
                frame,
                target_column="alpha_vs_sector_20d",
                model_name="ridge_model",
                min_train_dates=5,
                max_train_dates=8,
                test_window_dates=2,
                model_scope="global",
                evaluation_stride_dates=10,
                label_horizon_dates=1,
            )

        self.assertIsNotNone(predictions)
        self.assertTrue(observed_train_dates)
        self.assertLessEqual(max(len(train_dates) for train_dates in observed_train_dates), 8)

    def test_feature_ic_report_writes_ranked_survivor_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            db_manager = type("FakeDB", (), {"paths": paths})()
            service = ShortlistModelService(db_manager=db_manager)
            dates = pd.bdate_range("2026-01-02", periods=25)
            rows = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
                    signal = float(date_index + ticker_index)
                    rows.append(
                        {
                            "snapshot_date": snapshot_date,
                            "ticker": ticker,
                            "sector": "Energy",
                            "md_volume_30d": 50_000_000.0,
                            "relative_strength_index_vs_spy": signal,
                            "roc_63": 1.0,
                            "sma_200_dist": 1.0,
                            "vol_alpha": 1.0,
                            "alpha_vs_sector_20d": signal,
                        }
                    )
            frame = pd.DataFrame(rows)

            result = service._feature_ic_report(
                frame,
                target_column="alpha_vs_sector_20d",
                min_train_dates=5,
                test_window_dates=2,
                evaluation_stride_dates=5,
                label_horizon_dates=5,
                min_feature_ic=0.50,
            )

            report_text = (paths.reports_dir / "feature_ic_report.md").read_text(encoding="utf-8")
            self.assertIn("# Feature IC Report", report_text)
            self.assertIn("| relative_strength_index_vs_spy |", report_text)
            self.assertIn("true", report_text)
            self.assertIn("relative_strength_index_vs_spy", result["surviving_features"])

    def test_feature_ic_screen_drops_date_constant_features(self) -> None:
        service = ShortlistModelService(db_manager=object())
        rows = []
        for date_index, snapshot_date in enumerate(pd.bdate_range("2026-01-02", periods=12)):
            for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "spy_roc_20": float(date_index),
                        "qqq_roc_20": float(date_index),
                        "rsi_2": float(ticker_index),
                        "alpha_vs_sector_20d": float(date_index + ticker_index),
                    }
                )
        survivors = service._feature_ic_survivors_from_frame(
            pd.DataFrame(rows),
            target_column="alpha_vs_sector_20d",
            min_feature_ic=0.10,
        )

        self.assertIn("rsi_2", survivors)
        self.assertNotIn("spy_roc_20", survivors)
        self.assertNotIn("spy_roc_20__rank_all", survivors)
        self.assertNotIn("qqq_roc_20__rank_sector", survivors)

    def test_feature_ic_screen_drops_age_features_and_duplicate_survivors(self) -> None:
        service = ShortlistModelService(db_manager=object())
        rows = []
        for snapshot_date in pd.bdate_range("2026-01-02", periods=14):
            for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
                signal_value = float(ticker_index)
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "rsi_2": signal_value,
                        "ret_1d": signal_value,
                        "analyst_snapshot_age_days": signal_value,
                        "analyst_revision_snapshot_age_days": signal_value,
                        "alpha_vs_sector_20d": signal_value,
                    }
                )
        survivors = service._feature_ic_survivors_from_frame(
            pd.DataFrame(rows),
            target_column="alpha_vs_sector_20d",
            min_feature_ic=0.50,
        )

        self.assertIn("rsi_2", survivors)
        self.assertNotIn("ret_1d", survivors)
        self.assertNotIn("analyst_snapshot_age_days", survivors)
        self.assertNotIn("analyst_snapshot_age_days__rank_all", survivors)
        self.assertNotIn("analyst_revision_snapshot_age_days", survivors)

    def test_feature_ic_screen_applies_min_observation_floor(self) -> None:
        service = ShortlistModelService(db_manager=object())
        rows = []
        for date_index, snapshot_date in enumerate(pd.bdate_range("2026-01-02", periods=40)):
            for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
                full_signal = float(ticker_index)
                sparse_signal = full_signal if date_index < 4 else None
                rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "ticker": ticker,
                        "sector": "Energy",
                        "rsi_2": full_signal,
                        "analyst_target_upside": sparse_signal,
                        "alpha_vs_sector_20d": full_signal,
                    }
                )
        survivors = service._feature_ic_survivors_from_frame(
            pd.DataFrame(rows),
            target_column="alpha_vs_sector_20d",
            min_feature_ic=0.50,
            min_observation_fraction=0.20,
        )

        self.assertIn("rsi_2", survivors)
        self.assertNotIn("analyst_target_upside", survivors)

    def test_feature_ic_report_marks_duplicate_survivors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            service = ShortlistModelService(db_manager=type("FakeDB", (), {"paths": paths})())
            rows = []
            for snapshot_date in pd.bdate_range("2026-01-02", periods=20):
                for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
                    signal_value = float(ticker_index)
                    rows.append(
                        {
                            "snapshot_date": snapshot_date,
                            "ticker": ticker,
                            "sector": "Energy",
                            "rsi_2": signal_value,
                            "ret_1d": signal_value,
                            "analyst_snapshot_age_days": signal_value,
                            "alpha_vs_sector_20d": signal_value,
                        }
                    )

            result = service._feature_ic_report(
                pd.DataFrame(rows),
                target_column="alpha_vs_sector_20d",
                min_train_dates=5,
                test_window_dates=2,
                evaluation_stride_dates=5,
                label_horizon_dates=5,
                min_feature_ic=0.50,
                feature_columns_override=["rsi_2", "ret_1d", "analyst_snapshot_age_days"],
            )

            report_text = (paths.reports_dir / "feature_ic_report.md").read_text(encoding="utf-8")
            self.assertEqual(result["surviving_features"], ["rsi_2"])
            self.assertIn("| ret_1d |", report_text)
            self.assertIn("| ret_1d | 1.000000 | 1.000000 | 12 | rsi_2 | false |", report_text)
            self.assertNotIn("analyst_snapshot_age_days", report_text)

    def test_shortlist_model_writes_failure_report_when_no_candidate_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            paths.logs_dir.mkdir(parents=True, exist_ok=True)
            dates = pd.bdate_range("2026-01-02", periods=30)
            rows = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, ticker in enumerate(("AAA", "BBB")):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": "Energy",
                        "passed_any_strategy": 1,
                        "passed_slots_json": '["energy"]',
                        "md_volume_30d": 50_000_000.0,
                        "adj_close": 100.0,
                        "alpha_vs_sector_20d": 0.01 * (ticker_index + 1),
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float(feature_index + ticker_index + date_index)
                    rows.append(row)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame
                    self.decommission_calls = []

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()
                def decommission_shortlist_model_runs(self, **kwargs):
                    self.decommission_calls.append(kwargs)
                    return 1

            gate = {
                "scan_policy": {
                    "shortlist_model": {
                        "min_feature_ic": 0.0,
                        "promotion_gate": {
                            "enabled": True,
                            "min_recent_20d_hit_rate_excess": 1.01,
                            "min_recent_20d_beat_universe_rate": 1.01,
                            "min_recent_20d_mean_target_excess": 1.01,
                            "min_recent_60d_hit_rate_excess": 1.01,
                            "min_recent_60d_beat_universe_rate": 1.01,
                            "min_recent_60d_mean_target_excess": 1.01,
                            "min_recent_20d_spearman": 1.01,
                            "min_recent_60d_spearman": 1.01,
                            "min_recent_1fold_spearman": 1.01,
                            "min_recent_3fold_spearman": 1.01,
                        }
                    }
                }
            }
            fake_db = FakeDB(paths, pd.DataFrame(rows))
            service = ShortlistModelService(fake_db)

            with patch("src.research.shortlist_model_service.load_feature_config", return_value=gate), \
                 patch.object(service, "_score_xgboost_model", return_value=None), \
                 self.assertRaisesRegex(ValueError, "No shortlist model candidate passed"):
                service.run(top_n=1, min_train_dates=4, test_window_dates=2)

            report_text = (paths.reports_dir / "shortlist_model.md").read_text(encoding="utf-8")
            self.assertIn("## Promotion Failure", report_text)
            self.assertIn("- selected_model: n/a", report_text)
            self.assertIn("- live_output_top_n: 1", report_text)
            self.assertIn("- promotion_top_n: 2", report_text)
            self.assertIn("- oos_evaluation_stride_dates: 20", report_text)
            self.assertIn("- training_label_policy: horizon-strided non-overlapping dates after label embargo", report_text)
            self.assertIn("## Regime Matching", report_text)
            self.assertIn("- regime_matching_folds: attempted=", report_text)
            self.assertIn("- attempted_folds:", report_text)
            self.assertIn("- matched_folds:", report_text)
            self.assertIn("- fallback_folds:", report_text)
            self.assertIn("- unknown_folds:", report_text)
            self.assertIn("## Surviving Features By Fold Regime", report_text)
            self.assertIn("- days_since_last_champion: n/a", report_text)
            self.assertTrue((paths.reports_dir / "shortlist_model_oos_predictions.csv").exists())
            self.assertTrue((paths.reports_dir / "feature_ic_report.md").exists())
            self.assertEqual(len(fake_db.decommission_calls), 1)
            self.assertEqual(fake_db.decommission_calls[0]["horizon_days"], 20)
            self.assertEqual(fake_db.decommission_calls[0]["eligible_universe_mode"], "passed_only")

    def test_shortlist_model_dry_run_does_not_persist_or_decommission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = AppPaths(
                root_dir=root,
                data_dir=root / "data",
                duckdb_path=root / "data" / "market_data.duckdb",
                sqlite_path=root / "data" / "ledger.sqlite",
                reports_dir=root / "reports",
                logs_dir=root / "logs",
                config_path=root / "config.yaml",
                env_path=root / ".env",
                production_strategy_path=root / "production_strategy.json",
                production_strategies_path=root / "production_strategies.json",
            )
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            dates = pd.bdate_range("2026-01-02", periods=30)
            rows = []
            for date_index, snapshot_date in enumerate(dates):
                for ticker_index, ticker in enumerate(("AAA", "BBB")):
                    row = {
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": "Energy",
                        "passed_any_strategy": 1,
                        "passed_slots_json": '["energy"]',
                        "md_volume_30d": 50_000_000.0,
                        "adj_close": 100.0,
                        "alpha_vs_sector_20d": 0.01 * (ticker_index + 1),
                    }
                    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS):
                        row[column] = float(feature_index + ticker_index + date_index)
                    rows.append(row)

            class FakeDB:
                def __init__(self, paths, snapshot_frame):
                    self.paths = paths
                    self._snapshot_frame = snapshot_frame
                    self.decommission_calls = []
                    self.run_rows = []
                    self.prediction_rows = []

                def initialize(self): return None
                def load_universe_daily_snapshots(self, snapshot_date=None):
                    return self._snapshot_frame.copy()
                def decommission_shortlist_model_runs(self, **kwargs):
                    self.decommission_calls.append(kwargs)
                    return 1
                def insert_shortlist_model_run(self, *, row):
                    self.run_rows.append(row)
                    return 1
                def replace_shortlist_model_predictions(self, **kwargs):
                    self.prediction_rows.extend(kwargs.get("rows", []))

            gate = {
                "scan_policy": {
                    "shortlist_model": {
                        "min_feature_ic": 0.0,
                        "promotion_gate": {
                            "enabled": True,
                            "min_recent_20d_hit_rate_excess": 1.01,
                            "min_recent_20d_beat_universe_rate": 1.01,
                            "min_recent_20d_mean_target_excess": 1.01,
                            "min_recent_60d_hit_rate_excess": 1.01,
                            "min_recent_60d_beat_universe_rate": 1.01,
                            "min_recent_60d_mean_target_excess": 1.01,
                            "min_recent_20d_spearman": 1.01,
                            "min_recent_60d_spearman": 1.01,
                            "min_recent_1fold_spearman": 1.01,
                            "min_recent_3fold_spearman": 1.01,
                        }
                    }
                }
            }
            fake_db = FakeDB(paths, pd.DataFrame(rows))
            service = ShortlistModelService(fake_db)

            with patch("src.research.shortlist_model_service.load_feature_config", return_value=gate), \
                 patch.object(service, "_score_xgboost_model", return_value=None), \
                 self.assertRaisesRegex(ValueError, "No shortlist model candidate passed"):
                service.run(top_n=1, min_train_dates=4, test_window_dates=2, persist=False)

            self.assertEqual(fake_db.decommission_calls, [])
            self.assertEqual(fake_db.run_rows, [])
            self.assertEqual(fake_db.prediction_rows, [])

    def test_runtime_loader_does_not_refresh_stale_model_without_explicit_permission(self) -> None:
        class FakeDB:
            def __init__(self):
                self.refresh_count = 0

            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, xgboost_config="baseline", limit=1):
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "live_snapshot_date": "2026-05-18",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-19"]

            def load_shortlist_model_predictions(self, **kwargs):
                raise AssertionError("stale runtime loader should not read predictions after declining refresh")

            def load_universe_daily_snapshots(self, snapshot_date=None):
                self.refresh_count += 1
                raise AssertionError("runtime loader should not retrain without allow_refresh=True")

        fake_db = FakeDB()
        context = load_live_shortlist_model_context(
            fake_db,
            refresh_if_stale=True,
            allow_refresh=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNone(context)
        self.assertEqual(fake_db.refresh_count, 0)

    def test_runtime_loader_rejects_champion_with_negative_latest_spearman(self) -> None:
        class FakeDB:
            def load_shortlist_model_runs(self, *, horizon_days, eligible_universe_mode=None, model_scope=None, xgboost_config="baseline", limit=1):
                return pd.DataFrame(
                    [
                        {
                            "generated_at": "2026-05-26T17:00:00+00:00",
                            "champion_model": "lasso_model",
                            "live_snapshot_date": "2026-05-22",
                        }
                    ]
                )

            def list_universe_daily_snapshot_dates(self):
                return ["2026-05-22"]

            def load_shortlist_model_predictions(self, *, dataset_split, model_name, **kwargs):
                if model_name != "lasso_model":
                    return pd.DataFrame()
                if dataset_split == "live":
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_date": "2026-05-22",
                                "ticker": "AAA",
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": 0.11,
                            }
                        ]
                    )
                rows = []
                for snapshot_date, targets in [
                    ("2026-05-18", (0.04, 0.03, -0.01)),
                    ("2026-05-19", (0.05, 0.02, -0.01)),
                    ("2026-05-20", (0.04, 0.01, -0.01)),
                    ("2026-05-21", (-0.04, -0.02, 0.03)),
                ]:
                    for ticker, predicted_alpha, target in zip(("AAA", "BBB", "CCC"), (0.11, 0.08, 0.01), targets):
                        rows.append(
                            {
                                "snapshot_date": snapshot_date,
                                "ticker": ticker,
                                "sector": "Energy",
                                "md_volume_30d": 50_000_000.0,
                                "predicted_alpha": predicted_alpha,
                                "actual_alpha_vs_sector": target,
                            }
                        )
                return pd.DataFrame(rows)

        context = load_live_shortlist_model_context(
            FakeDB(),
            top_n=2,
            test_window_dates=1,
            refresh_if_stale=False,
            eligible_universe_mode="passed_only",
            model_scope="sector_specific",
        )

        self.assertIsNone(context)

    def test_champion_selection_refuses_models_that_fail_promotion_gate(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model", "mean_target": 0.08, "beat_universe_rate": 0.60, "positive_date_rate": 0.70},
                {"model": "lasso_model", "mean_target": 0.04, "beat_universe_rate": 0.55, "positive_date_rate": 0.60},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model_last_fold", "hit_rate": 0.45, "beat_universe_rate": 0.40, "mean_target": -0.01, "spearman": 0.20},
                {"model": "xgboost_model_trailing_3folds", "hit_rate": 0.85, "beat_universe_rate": 0.85, "mean_target": 0.04, "spearman": 0.20},
                {"model": "xgboost_model_full_oos", "hit_rate": 0.55, "beat_universe_rate": 0.55, "mean_target": 0.04, "spearman": 0.20},
                {"model": "lasso_model_last_fold", "hit_rate": 0.40, "beat_universe_rate": 0.35, "mean_target": -0.02, "spearman": 0.20},
                {"model": "lasso_model_trailing_3folds", "hit_rate": 0.85, "beat_universe_rate": 0.85, "mean_target": 0.04, "spearman": 0.20},
                {"model": "lasso_model_full_oos", "hit_rate": 0.52, "beat_universe_rate": 0.52, "mean_target": 0.01, "spearman": 0.20},
            ]
        )
        promotion_gate = {
            "enabled": True,
            "min_recent_20d_hit_rate_excess": 0.50,
            "min_recent_20d_beat_universe_rate": 0.50,
            "min_recent_20d_mean_target_excess": 0.0,
            "min_recent_60d_hit_rate_excess": 0.50,
            "min_recent_60d_beat_universe_rate": 0.50,
            "min_recent_60d_mean_target_excess": 0.0,
            "min_recent_1fold_hit_rate_excess": 0.50,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_3fold_hit_rate_excess": 0.50,
            "min_recent_3fold_beat_universe_rate": 0.50,
            "min_recent_3fold_mean_target_excess": 0.0,
            "min_recent_20d_spearman": 0.0,
            "min_recent_60d_spearman": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "min_recent_3fold_spearman": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
            )

    def test_champion_selection_refuses_models_with_negative_latest_spearman(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "lasso_model", "mean_target": 0.08, "beat_universe_rate": 0.80, "positive_date_rate": 0.80},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {"model": "lasso_model_last_fold", "hit_rate": 0.0, "beat_universe_rate": 0.0, "mean_target": -0.03, "spearman": -0.10},
                {"model": "lasso_model_trailing_3folds", "hit_rate": 0.67, "beat_universe_rate": 0.67, "mean_target": 0.01, "spearman": 0.02},
                {"model": "lasso_model_full_oos", "hit_rate": 0.85, "beat_universe_rate": 0.85, "mean_target": 0.04, "spearman": 0.04},
            ]
        )
        promotion_gate = {
            "enabled": True,
            "min_recent_20d_hit_rate_excess": 0.50,
            "min_recent_20d_beat_universe_rate": 0.50,
            "min_recent_20d_mean_target_excess": 0.0,
            "min_recent_60d_hit_rate_excess": 0.50,
            "min_recent_60d_beat_universe_rate": 0.50,
            "min_recent_60d_mean_target_excess": 0.0,
            "min_recent_1fold_hit_rate_excess": 0.50,
            "min_recent_1fold_beat_universe_rate": 0.50,
            "min_recent_1fold_mean_target_excess": 0.0,
            "min_recent_3fold_hit_rate_excess": 0.50,
            "min_recent_3fold_beat_universe_rate": 0.50,
            "min_recent_3fold_mean_target_excess": 0.0,
            "min_recent_20d_spearman": 0.0,
            "min_recent_60d_spearman": 0.0,
            "min_recent_1fold_spearman": 0.0,
            "min_recent_3fold_spearman": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
            )

    def test_sixty_day_promotion_gate_requires_last_fold_recency(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model", "mean_target": 0.08, "beat_universe_rate": 0.80, "positive_date_rate": 0.80},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model_last_fold", "hit_rate": 0.10, "beat_universe_rate": 0.10, "mean_target": -0.10, "spearman": -0.20},
                {"model": "xgboost_model_trailing_3folds", "hit_rate": 0.60, "beat_universe_rate": 0.60, "mean_target": 0.08, "spearman": 0.02},
                {"model": "xgboost_model_full_oos", "hit_rate": 0.60, "beat_universe_rate": 0.60, "mean_target": 0.08, "spearman": 0.02},
            ]
        )
        promotion_gate = service._load_promotion_gate()

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
                required_recent_windows=service._promotion_recent_windows(horizon_days=60),
                required_fold_windows=service._promotion_fold_windows(horizon_days=60),
            )

        self.assertEqual(service._promotion_recent_windows(horizon_days=60), ())
        self.assertEqual(service._promotion_fold_windows(horizon_days=60), (1, 3))

    def test_sixty_day_promotion_gate_accepts_positive_fold_recency(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "xgboost_model", "mean_target": 0.08, "beat_universe_rate": 0.80, "positive_date_rate": 0.80},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                self._gate_row("xgboost_model_last_fold", hit_rate=0.60, universe_hit_rate=0.57, beat_universe_rate=0.60, mean_target=0.01, universe_mean_target=0.0, spearman=0.01),
                self._gate_row("xgboost_model_trailing_3folds", hit_rate=0.60, universe_hit_rate=0.57, beat_universe_rate=0.60, mean_target=0.08, universe_mean_target=0.0, spearman=0.02),
                self._gate_row("xgboost_model_full_oos", hit_rate=0.60, universe_hit_rate=0.57, beat_universe_rate=0.60, mean_target=0.08, universe_mean_target=0.0, spearman=0.02),
            ]
        )
        promotion_gate = service._load_promotion_gate()

        champion, passed = service._choose_champion_model(
            full_summaries=full_summaries,
            acceptance_summaries=acceptance_summaries,
            promotion_gate=promotion_gate,
            required_recent_windows=service._promotion_recent_windows(horizon_days=60),
            required_fold_windows=service._promotion_fold_windows(horizon_days=60),
        )

        self.assertEqual(champion, "xgboost_model")
        self.assertTrue(passed)

    def test_champion_selection_refuses_ticker_concentrated_acceptance_window(self) -> None:
        service = ShortlistModelService(db_manager=object())
        full_summaries = pd.DataFrame(
            [
                {"model": "ridge_model", "mean_target": 0.08, "beat_universe_rate": 0.80, "positive_date_rate": 0.80},
            ]
        )
        acceptance_summaries = pd.DataFrame(
            [
                {
                    "model": "ridge_model_full_oos",
                    "hit_rate": 0.85,
                    "beat_universe_rate": 0.85,
                    "mean_target": 0.04,
                    "spearman": 0.04,
                    "top_ticker_date_rate": 0.20,
                },
                {
                    "model": "ridge_model_last_fold",
                    "hit_rate": 0.85,
                    "beat_universe_rate": 0.85,
                    "mean_target": 0.04,
                    "spearman": 0.04,
                    "top_ticker_date_rate": 0.45,
                },
                {
                    "model": "ridge_model_trailing_3folds",
                    "hit_rate": 0.85,
                    "beat_universe_rate": 0.85,
                    "mean_target": 0.04,
                    "spearman": 0.04,
                    "top_ticker_date_rate": 0.20,
                },
            ]
        )
        promotion_gate = service._load_promotion_gate()

        with self.assertRaisesRegex(ValueError, "No shortlist model candidate passed the promotion gate"):
            service._choose_champion_model(
                full_summaries=full_summaries,
                acceptance_summaries=acceptance_summaries,
                promotion_gate=promotion_gate,
            )

    def test_classification_target_preserves_unmatured_alpha_as_missing(self) -> None:
        service = ShortlistModelService(db_manager=object())
        frame = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "ticker": "AAA",
                    "sector": "Energy",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 50.0,
                    "alpha_vs_sector_20d_pos": 1.0,
                },
                {
                    "snapshot_date": "2026-01-03",
                    "ticker": "BBB",
                    "sector": "Energy",
                    "passed_any_strategy": 1,
                    "md_volume_30d": 50_000_000.0,
                    "adj_close": 50.0,
                    "alpha_vs_sector_20d_pos": float("nan"),
                },
            ]
        )

        prepared = service._prepare_snapshot_frame(frame)
        matured = service._build_matured_eligible_universe(
            prepared,
            target_column="alpha_vs_sector_20d_pos",
            eligible_universe_mode="passed_only",
        )

        self.assertEqual(matured["ticker"].tolist(), ["AAA"])
