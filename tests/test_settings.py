from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import settings


class SettingsTests(unittest.TestCase):
    def test_settings_load_capital_parameters_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                'TOTAL_CAPITAL=75000\nRISK_PER_TRADE=0.03\nGMAIL_USER="user@example.com"\n',
                encoding="utf-8",
            )
            settings.get_settings.cache_clear()
            with patch("src.settings._project_root", return_value=root):
                loaded = settings.get_settings()
            settings.get_settings.cache_clear()

            self.assertEqual(loaded.total_capital, 75000.0)
            self.assertEqual(loaded.risk_per_trade, 0.03)
            self.assertEqual(loaded.env["GMAIL_USER"], "user@example.com")
