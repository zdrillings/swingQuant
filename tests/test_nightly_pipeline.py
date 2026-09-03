from __future__ import annotations

from pathlib import Path
import unittest


class NightlyPipelineScriptTests(unittest.TestCase):
    def test_shortlist_promotion_gate_failure_continues_to_downstream_steps(self) -> None:
        script = Path("ops/nightly_pipeline.sh").read_text(encoding="utf-8")

        self.assertIn("shortlist_status=\"${PIPESTATUS[0]}\"", script)
        self.assertIn("No shortlist model candidate passed the promotion gate", script)
        self.assertIn("continuing with previously persisted model context", script)
        self.assertIn("exit \"${shortlist_status}\"", script)
        self.assertLess(
            script.index("No shortlist model candidate passed the promotion gate"),
            script.index("echo \"[$(date --iso-8601=seconds)] analyst-snapshot\""),
        )


if __name__ == "__main__":
    unittest.main()
