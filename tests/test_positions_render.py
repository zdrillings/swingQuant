from __future__ import annotations

import unittest

from src.positions.service import PositionSnapshot, PositionsReport


class PositionsRenderTests(unittest.TestCase):
    def test_render_console_handles_empty_report(self) -> None:
        report = PositionsReport(
            generated_at="2026-05-11",
            position_count=0,
            sell_count=0,
            hold_count=0,
            snapshots=(),
        )
        rendered = report.render_console()
        self.assertIn("Open positions: 0", rendered)
        self.assertIn("No open positions.", rendered)
