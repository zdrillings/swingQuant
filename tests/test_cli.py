from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import build_parser, main
from src.settings import AppPaths, RuntimeSettings


class CliTests(unittest.TestCase):
    def test_regime_meter_parser_accepts_required_modes(self) -> None:
        args = build_parser().parse_args(["regime-meter", "--backfill", "--latest", "--report", "--email"])

        self.assertEqual(args.command, "regime-meter")
        self.assertTrue(args.backfill)
        self.assertTrue(args.latest)
        self.assertTrue(args.report)
        self.assertTrue(args.email)

    def test_monitor_parser_disables_email_by_default(self) -> None:
        default_args = build_parser().parse_args(["monitor"])
        email_args = build_parser().parse_args(["monitor", "--email"])

        self.assertEqual(default_args.command, "monitor")
        self.assertFalse(default_args.email)
        self.assertTrue(email_args.email)

    def test_shortlist_model_defaults_to_config_when_flags_omitted(self) -> None:
        settings = RuntimeSettings(
            paths=AppPaths(
                root_dir=Path("."),
                data_dir=Path("data"),
                duckdb_path=Path("data/market_data.duckdb"),
                sqlite_path=Path("data/ledger.sqlite"),
                reports_dir=Path("reports"),
                logs_dir=Path("logs"),
                config_path=Path("config.yaml"),
                env_path=Path(".env"),
                production_strategy_path=Path("production_strategy.json"),
            ),
            env={},
            total_capital=50_000.0,
            risk_per_trade=0.02,
        )

        class FakeReport:
            output_path = Path("reports/shortlist_model.md")
            target_column = "alpha_vs_sector_60d"
            champion_model = "xgboost_model"
            oos_dates = 3
            live_candidates = 2

        captured: dict[str, object] = {}

        def fake_run(self, **kwargs):
            captured.update(kwargs)
            return FakeReport()

        with patch("src.settings.get_settings", return_value=settings), \
             patch("src.cli.configure_logging", return_value=None), \
             patch("src.cli.load_feature_config", return_value={"scan_policy": {"shortlist_model": {"horizon_days": 60, "top_n": 2}}}), \
             patch("src.cli.ShortlistModelService.run", new=fake_run):
            exit_code = main(["shortlist-model", "--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["horizon_days"], 60)
        self.assertEqual(captured["top_n"], 2)

    def test_scan_failure_sends_failure_email(self) -> None:
        settings = RuntimeSettings(
            paths=AppPaths(
                root_dir=Path("."),
                data_dir=Path("data"),
                duckdb_path=Path("data/market_data.duckdb"),
                sqlite_path=Path("data/ledger.sqlite"),
                reports_dir=Path("reports"),
                logs_dir=Path("logs"),
                config_path=Path("config.yaml"),
                env_path=Path(".env"),
                production_strategy_path=Path("production_strategy.json"),
            ),
            env={
                "GMAIL_USER": "sender@example.com",
                "GMAIL_APP_PASSWORD": "secret",
                "RECIPIENT_EMAIL": "recipient@example.com",
            },
            total_capital=50_000.0,
            risk_per_trade=0.02,
        )
        email_calls: list[dict[str, object]] = []

        with patch("src.settings.get_settings", return_value=settings), \
             patch("src.cli.configure_logging", return_value=None), \
             patch("src.cli.ScanService.run", side_effect=ValueError("Scan snapshot is stale")), \
             patch("src.cli.send_html_email", side_effect=lambda **kwargs: email_calls.append(kwargs)):
            exit_code = main(["scan"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(email_calls), 1)
        self.assertEqual(email_calls[0]["subject"], "SwingQuant Evening Brief Failed")
        self.assertIn("Scan snapshot is stale", str(email_calls[0]["html_body"]))


if __name__ == "__main__":
    unittest.main()
