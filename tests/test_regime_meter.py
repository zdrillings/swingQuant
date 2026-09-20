from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.research.regime_meter_service import RegimeMeterService
from src.scan.service import RegimeGatePolicy, ScanService
from src.settings import AppPaths, RuntimeSettings
from src.utils.db_manager import DatabaseManager
from src.utils.strategy import ExitRules, ProductionStrategy


def _paths(root: Path) -> AppPaths:
    return AppPaths(
        root_dir=root,
        data_dir=root / "data",
        duckdb_path=root / "data" / "market_data.duckdb",
        sqlite_path=root / "data" / "ledger.sqlite",
        reports_dir=root / "reports",
        logs_dir=root / "logs",
        config_path=root / "config.yaml",
        env_path=root / ".env",
        production_strategy_path=root / "production_strategy.json",
    )


def _settings(paths: AppPaths) -> RuntimeSettings:
    return RuntimeSettings(
        paths=paths,
        env={
            "GMAIL_USER": "sender@example.com",
            "GMAIL_APP_PASSWORD": "secret",
            "RECIPIENT_EMAIL": "recipient@example.com",
        },
        total_capital=50_000.0,
        risk_per_trade=0.02,
    )


def _insert_snapshot_day(manager: DatabaseManager, snapshot_date: date, sign: float = 1.0) -> None:
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for index, ticker in enumerate(tickers, start=1):
        roc = float(index)
        alpha = sign * float(index) / 100.0
        rows.append((snapshot_date.isoformat(), ticker, roc, alpha, float(index) / 1000.0))
    with manager.duckdb_connection() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO universe_daily_snapshots
                (snapshot_date, ticker, roc_126, alpha_vs_sector_20d, ret_1d)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


class RegimeMeterServiceTests(unittest.TestCase):
    def test_classification_threshold_boundaries_are_neutral(self) -> None:
        classify = RegimeMeterService.classify

        self.assertEqual(classify(-0.051, reversal_ic_threshold=-0.05, trending_ic_threshold=0.05), "reversal")
        self.assertEqual(classify(-0.050, reversal_ic_threshold=-0.05, trending_ic_threshold=0.05), "neutral")
        self.assertEqual(classify(0.050, reversal_ic_threshold=-0.05, trending_ic_threshold=0.05), "neutral")
        self.assertEqual(classify(0.051, reversal_ic_threshold=-0.05, trending_ic_threshold=0.05), "trending")

    def test_transition_matrix_counts_session_horizon_pairs(self) -> None:
        service = RegimeMeterService(db_manager=object())
        frame = pd.DataFrame(
            {
                "snapshot_date": pd.bdate_range("2026-01-02", periods=6),
                "classification": ["neutral", "neutral", "trending", "reversal", "reversal", "neutral"],
            }
        )

        matrix = service._transition_matrix(frame, horizon_sessions=2)

        self.assertEqual(matrix["neutral"]["trending"], 1)
        self.assertEqual(matrix["neutral"]["reversal"], 1)
        self.assertEqual(matrix["trending"]["reversal"], 1)
        self.assertEqual(matrix["reversal"]["neutral"], 1)

    def test_backfill_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            manager = DatabaseManager(paths)
            manager.initialize()
            start = date(2026, 1, 1)
            for offset in range(25):
                _insert_snapshot_day(manager, start + timedelta(days=offset), sign=1.0)

            service = RegimeMeterService(manager)
            service.run(backfill=True)
            first = manager.load_regime_meter()
            service.run(backfill=True)
            second = manager.load_regime_meter()

        pd.testing.assert_frame_equal(first, second)

    def test_latest_recomputes_trailing_window_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            manager = DatabaseManager(paths)
            manager.initialize()
            start = date(2026, 1, 1)
            for offset in range(25):
                _insert_snapshot_day(manager, start + timedelta(days=offset), sign=1.0)
            service = RegimeMeterService(manager)
            service.run(backfill=True)

            with manager.duckdb_connection() as connection:
                connection.execute(
                    "UPDATE regime_meter SET classification = 'bogus' WHERE snapshot_date = ?",
                    ((start + timedelta(days=24)).isoformat(),),
                )
            _insert_snapshot_day(manager, start + timedelta(days=25), sign=-1.0)
            service.run(latest=True)
            frame = manager.load_regime_meter()

        self.assertNotIn("bogus", set(frame["classification"].astype(str)))

    def test_scan_uses_latest_matured_regime_row_without_lookahead(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            manager = DatabaseManager(paths)
            manager.initialize()
            start = date(2026, 1, 1)
            for offset in range(25):
                _insert_snapshot_day(manager, start + timedelta(days=offset), sign=1.0)
            manager.replace_regime_meter_rows(
                [
                    {
                        "snapshot_date": "2026-01-05",
                        "mom_ic_daily": -0.2,
                        "wml_20d_alpha": -0.1,
                        "wml_1d": 0.0,
                        "mom_ic_20d_avg": -0.2,
                        "mom_ic_60d_avg": None,
                        "classification": "reversal",
                    },
                    {
                        "snapshot_date": "2026-01-25",
                        "mom_ic_daily": 0.2,
                        "wml_20d_alpha": 0.1,
                        "wml_1d": 0.0,
                        "mom_ic_20d_avg": 0.2,
                        "mom_ic_60d_avg": None,
                        "classification": "trending",
                    },
                ]
            )
            service = ScanService(manager)
            state = service._load_regime_gate_state(
                policy=RegimeGatePolicy(-0.05, 0.05, False, ()),
                scan_date=date(2026, 1, 25),
            )

        self.assertIsNotNone(state)
        self.assertEqual(state.snapshot_date, date(2026, 1, 5))
        self.assertEqual(state.classification, "reversal")

    def test_scan_stand_down_emails_zero_pick_report(self) -> None:
        email_calls = []

        class FakeDB:
            def initialize(self): pass
            def list_universe_rows(self, active_only=True):
                return [{"ticker": "AAA", "sector": "Technology"}]
            def list_open_trades(self):
                return []
            def replace_scan_candidates(self, *, scan_date, rows):
                self.rows = list(rows)
            def load_price_history(self, tickers):
                return pd.DataFrame()
            def load_latest_regime_meter(self, *, as_of_date=None, horizon_sessions=20):
                return {
                    "snapshot_date": date(2026, 8, 10),
                    "classification": "reversal",
                    "mom_ic_20d_avg": -0.12,
                    "mom_ic_60d_avg": -0.08,
                    "wml_20d_alpha": -0.03,
                }

        strategy = ProductionStrategy(
            strategy_id=1,
            promoted_at="2026-01-01T00:00:00",
            indicators={},
            exit_rules=ExitRules(None, None, 20),
            slot="momentum",
            sector="ALL",
        )
        snapshot = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-09-08"),
                    "ticker": "AAA",
                    "sector": "Technology",
                    "regime_green": True,
                }
            ]
        )
        paths = _paths(Path("."))
        with patch("src.scan.service.get_settings", return_value=_settings(paths)), \
             patch("src.scan.service.load_feature_config", return_value={"regime_gating": {"enforce": True}}), \
             patch("src.scan.service.load_active_strategies", return_value={"momentum": strategy}), \
             patch("src.scan.service.overlay_price_history", return_value=pd.DataFrame()), \
             patch("src.scan.service.build_analysis_frame", return_value=(snapshot, {})), \
             patch("src.scan.service.latest_snapshot", return_value=snapshot):
            report = ScanService(
                FakeDB(),
                market_data_client=object(),
                email_sender=lambda subject, html_body, settings: email_calls.append((subject, html_body)),
            ).run()

        self.assertEqual(report.candidate_count, 0)
        self.assertTrue(report.emailed)
        self.assertEqual(len(email_calls), 1)
        self.assertIn("REGIME STAND-DOWN", email_calls[0][0])
        self.assertIn("regime: reversal (mom_ic_20d -0.120)", email_calls[0][1])

    def test_enforce_false_does_not_stand_down(self) -> None:
        service = ScanService(db_manager=None)
        state = service._load_regime_gate_state
        self.assertEqual(
            service._stood_down_slots(
                regime_state=None,
                scan_strategies={"momentum": object()},
            ),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
