from __future__ import annotations

import unittest
from unittest.mock import patch

from src.sync import universe


HTML_TEMPLATE = """
<html>
  <body>
    <table>
      <thead>
        <tr><th>Symbol</th><th>GICS Sector</th></tr>
      </thead>
      <tbody>
        <tr><td>AAA</td><td>Industrials</td></tr>
        <tr><td>BRK.B</td><td>Financials</td></tr>
      </tbody>
    </table>
  </body>
</html>
"""


class SyncUniverseTests(unittest.TestCase):
    def test_fetch_html_uses_explicit_user_agent(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self) -> bytes:
                return b"<html></html>"

        def fake_urlopen(request):
            captured["user_agent"] = request.headers.get("User-agent")
            captured["accept_language"] = request.headers.get("Accept-language")
            return FakeResponse()

        with patch("src.sync.universe.urlopen", side_effect=fake_urlopen):
            html = universe._fetch_html("https://example.com")

        self.assertEqual(html, "<html></html>")
        self.assertIn("Mozilla/5.0", captured["user_agent"])
        self.assertEqual(captured["accept_language"], "en-US,en;q=0.9")

    def test_scrape_sp_universe_parses_html_with_stringio_and_normalizes_tickers(self) -> None:
        with patch("src.sync.universe._fetch_html", return_value=HTML_TEMPLATE):
            members = universe.scrape_sp_universe()

        tickers = [member.ticker for member in members]
        sectors = {member.ticker: member.sector for member in members}
        self.assertEqual(tickers, ["AAA", "BRK-B"])
        self.assertEqual(sectors["BRK-B"], "Financials")
