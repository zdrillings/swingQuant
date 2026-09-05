from __future__ import annotations

from pathlib import Path
import unittest


class NightlyPipelineScriptTests(unittest.TestCase):
    def test_shortlist_promotion_gate_failure_continues_to_downstream_steps(self) -> None:
        script = Path("ops/nightly_pipeline.sh").read_text(encoding="utf-8")

        self.assertIn("shortlist_status=\"${PIPESTATUS[0]}\"", script)
        self.assertIn("trap - ERR", script)
        self.assertIn("trap notify_failure ERR", script)
        self.assertIn("No shortlist model candidate passed the promotion gate", script)
        self.assertIn("scan will be skipped", script)
        self.assertIn("shortlist_promotion_failed=1", script)
        self.assertIn("promotion_failures_file=\"data/promotion_failures.txt\"", script)
        self.assertIn("record_promotion_failure", script)
        self.assertIn("consecutive_promotion_failures", script)
        self.assertIn("SwingQuant scan skipped - no promotable shortlist champion", script)
        self.assertIn("clear_promotion_failures", script)
        self.assertIn("scan skipped because shortlist-model produced no promotable champion", script)
        self.assertIn("--oos-stride-dates 60", script)
        self.assertIn("exit \"${shortlist_status}\"", script)
        self.assertLess(
            script.index("echo \"[$(date --iso-8601=seconds)] analyst-snapshot\""),
            script.index("echo \"[$(date --iso-8601=seconds)] universe-backfill"),
        )
        self.assertLess(
            script.index("No shortlist model candidate passed the promotion gate"),
            script.index("echo \"[$(date --iso-8601=seconds)] extended-hours-snapshot\""),
        )
        self.assertLess(
            script.index("shortlist_promotion_failed=1"),
            script.index("echo \"[$(date --iso-8601=seconds)] scan skipped"),
        )


if __name__ == "__main__":
    unittest.main()
