from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main
from src.settings import AppPaths, RuntimeSettings


class CliTests(unittest.TestCase):
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
