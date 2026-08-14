from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from src.schwab.client import DEFAULT_REDIRECT_URI, SchwabClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class SchwabClientTests(unittest.TestCase):
    def test_build_authorization_url_uses_registered_redirect_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "schwab.yml"
            credentials_path.write_text(
                "client_id: abc\nclient_secret: def\n",
                encoding="utf-8",
            )

            url = SchwabClient(credentials_path=credentials_path).build_authorization_url()

        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "api.schwabapi.com")
        self.assertEqual(query["client_id"], ["abc"])
        self.assertEqual(query["redirect_uri"], [DEFAULT_REDIRECT_URI])

    def test_exchange_authorization_code_accepts_full_callback_url_and_saves_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "schwab.yml"
            token_path = Path(tmpdir) / "tokens.json"
            credentials_path.write_text(
                "app_key: abc\napp_secret: def\nredirect_uri: https://127.0.0.1\n",
                encoding="utf-8",
            )
            captured_body = {}

            def fake_urlopen(request, timeout):
                captured_body["body"] = request.data.decode("utf-8")
                return FakeResponse(
                    {
                        "access_token": "access",
                        "refresh_token": "refresh",
                        "expires_in": 1800,
                    }
                )

            with patch("src.schwab.client.urlopen", side_effect=fake_urlopen):
                token_state = SchwabClient(
                    credentials_path=credentials_path,
                    token_path=token_path,
                ).exchange_authorization_code("https://127.0.0.1/?code=AUTHCODE&session=ignored")

            self.assertEqual(token_state.access_token, "access")
            self.assertEqual(token_state.refresh_token, "refresh")
            self.assertTrue(token_path.exists())
            self.assertIn("code=AUTHCODE", captured_body["body"])
            self.assertIn("redirect_uri=https%3A%2F%2F127.0.0.1", captured_body["body"])

    def test_load_credentials_accepts_simple_key_value_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "schwab.json"
            credentials_path.write_text(
                "}\nclient_id: abc,\nclient_secret: def\n}\n",
                encoding="utf-8",
            )

            credentials = SchwabClient(credentials_path=credentials_path).load_credentials()

        self.assertEqual(credentials.client_id, "abc")
        self.assertEqual(credentials.client_secret, "def")
        self.assertEqual(credentials.redirect_uri, DEFAULT_REDIRECT_URI)

    def test_placeholder_callback_code_fails_before_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "schwab.yml"
            credentials_path.write_text("client_id: abc\nclient_secret: def\n", encoding="utf-8")

            with self.assertRaisesRegex(Exception, "placeholder"):
                SchwabClient(credentials_path=credentials_path).exchange_authorization_code("https://127.0.0.1/?code=...")


if __name__ == "__main__":
    unittest.main()
