from __future__ import annotations

import unittest

from src.settings import AppPaths, RuntimeSettings
from src.utils.emailer import _validate_email_settings


class EmailerTests(unittest.TestCase):
    def _settings(self, **env_overrides: str) -> RuntimeSettings:
        env = {
            "GMAIL_USER": "sender@example.com",
            "GMAIL_APP_PASSWORD": "app-password",
            "RECIPIENT_EMAIL": "recipient@example.com",
        }
        env.update(env_overrides)
        paths = AppPaths(
            root_dir=None,
            data_dir=None,
            duckdb_path=None,
            sqlite_path=None,
            reports_dir=None,
            logs_dir=None,
            config_path=None,
            env_path=None,
            production_strategy_path=None,
        )
        return RuntimeSettings(
            paths=paths,
            env=env,
            total_capital=None,
            risk_per_trade=None,
        )

    def test_validate_email_settings_accepts_valid_addresses(self) -> None:
        settings = self._settings()
        _validate_email_settings(settings)

    def test_validate_email_settings_rejects_invalid_recipient(self) -> None:
        settings = self._settings(RECIPIENT_EMAIL="zdrillingsgmail.com")

        with self.assertRaisesRegex(ValueError, "Invalid email address in .env"):
            _validate_email_settings(settings)

