from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import gzip
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen


AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TRADER_BASE_URL = "https://api.schwabapi.com/trader/v1"
DEFAULT_REDIRECT_URI = "https://127.0.0.1"
DEFAULT_CREDENTIALS_PATH = "~/schwab.yml"
DEFAULT_TOKEN_PATH = "~/.schwab_tokens.json"


class SchwabError(RuntimeError):
    """Raised when Schwab configuration or API calls fail."""


@dataclass(frozen=True)
class SchwabCredentials:
    client_id: str
    client_secret: str
    redirect_uri: str = DEFAULT_REDIRECT_URI


@dataclass(frozen=True)
class SchwabTokenState:
    access_token: str
    refresh_token: str | None
    expires_at: str | None
    raw: dict[str, Any]


class SchwabClient:
    def __init__(
        self,
        *,
        credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
        token_path: str | Path = DEFAULT_TOKEN_PATH,
    ) -> None:
        self.credentials_path = Path(credentials_path).expanduser()
        self.token_path = Path(token_path).expanduser()

    def build_authorization_url(self) -> str:
        credentials = self.load_credentials()
        return f"{AUTH_URL}?{urlencode({'client_id': credentials.client_id, 'redirect_uri': credentials.redirect_uri})}"

    def exchange_authorization_code(self, code_or_url: str) -> SchwabTokenState:
        credentials = self.load_credentials()
        code = self._extract_authorization_code(code_or_url)
        response = self._post_token(
            credentials=credentials,
            form={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": credentials.redirect_uri,
            },
        )
        return self._save_token_response(response)

    def refresh_access_token(self) -> SchwabTokenState:
        credentials = self.load_credentials()
        token_state = self.load_token_state()
        if not token_state.refresh_token:
            raise SchwabError("Stored Schwab token state does not include a refresh_token. Re-authorize first.")
        response = self._post_token(
            credentials=credentials,
            form={
                "grant_type": "refresh_token",
                "refresh_token": token_state.refresh_token,
            },
        )
        if "refresh_token" not in response and token_state.refresh_token:
            response["refresh_token"] = token_state.refresh_token
        return self._save_token_response(response)

    def list_account_numbers(self) -> list[dict[str, Any]]:
        return self._request_json("GET", f"{TRADER_BASE_URL}/accounts/accountNumbers")

    def list_positions(self, *, account_hash: str | None = None) -> list[dict[str, Any]]:
        accounts = self.list_account_numbers()
        if not accounts:
            return []
        selected_accounts = accounts
        if account_hash:
            selected_accounts = [
                account
                for account in accounts
                if str(account.get("hashValue") or account.get("accountNumber") or "") == str(account_hash)
            ]
            if not selected_accounts:
                raise SchwabError(f"No Schwab account matched hash/account value {account_hash!r}.")

        positions: list[dict[str, Any]] = []
        for account in selected_accounts:
            hash_value = account.get("hashValue")
            if not hash_value:
                continue
            account_data = self._request_json("GET", f"{TRADER_BASE_URL}/accounts/{hash_value}?fields=positions")
            securities_account = account_data.get("securitiesAccount", {})
            for position in securities_account.get("positions", []) or []:
                instrument = position.get("instrument", {}) or {}
                positions.append(
                    {
                        "account_hash": hash_value,
                        "account_number": account.get("accountNumber"),
                        "ticker": instrument.get("symbol"),
                        "asset_type": instrument.get("assetType"),
                        "instrument_type": instrument.get("type"),
                        "description": instrument.get("description"),
                        "quantity": position.get("longQuantity") or position.get("shortQuantity") or 0,
                        "market_value": position.get("marketValue"),
                        "average_price": position.get("averagePrice"),
                        "raw": position,
                    }
                )
        return positions

    def load_credentials(self) -> SchwabCredentials:
        if not self.credentials_path.exists():
            raise SchwabError(f"Missing Schwab credentials file: {self.credentials_path}")
        data = self._load_credentials_data()
        client_id = data.get("client_id") or data.get("app_key") or data.get("consumer_key")
        client_secret = data.get("client_secret") or data.get("app_secret") or data.get("consumer_secret")
        redirect_uri = data.get("redirect_uri") or DEFAULT_REDIRECT_URI
        if not client_id or not client_secret:
            raise SchwabError(
                f"{self.credentials_path} must contain client_id/client_secret "
                "or app_key/app_secret."
            )
        return SchwabCredentials(
            client_id=str(client_id),
            client_secret=str(client_secret),
            redirect_uri=str(redirect_uri),
        )

    def _load_credentials_data(self) -> dict[str, Any]:
        raw_text = self.credentials_path.read_text(encoding="utf-8")
        if self.credentials_path.suffix.lower() in {".yml", ".yaml"}:
            import yaml

            data = yaml.safe_load(raw_text) or {}
            if not isinstance(data, dict):
                raise SchwabError(f"Schwab credentials file must contain a YAML mapping: {self.credentials_path}")
            return data
        try:
            return json.loads(raw_text)
        except JSONDecodeError as exc:
            data = self._parse_loose_credentials(raw_text)
            if data:
                return data
            raise SchwabError(
                f"Schwab credentials file is not valid JSON: {self.credentials_path}. "
                'Expected format: {"client_id": "...", "client_secret": "...", "redirect_uri": "https://127.0.0.1"}'
            ) from exc

    @staticmethod
    def _parse_loose_credentials(raw_text: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw_line in raw_text.splitlines():
            line = raw_line.strip().strip(",")
            if not line or line in {"{", "}"} or line.startswith("#"):
                continue
            if ":" not in line:
                return {}
            key, value = line.split(":", 1)
            clean_key = key.strip().strip('"').strip("'")
            clean_value = value.strip().strip(",").strip().strip('"').strip("'")
            if clean_key and clean_value:
                values[clean_key] = clean_value
        return values

    def load_token_state(self) -> SchwabTokenState:
        if not self.token_path.exists():
            raise SchwabError(f"Missing Schwab token file: {self.token_path}. Run `sq schwab token CODE` first.")
        data = json.loads(self.token_path.read_text(encoding="utf-8"))
        access_token = data.get("access_token")
        if not access_token:
            raise SchwabError(f"Stored Schwab token file is missing access_token: {self.token_path}")
        return SchwabTokenState(
            access_token=str(access_token),
            refresh_token=str(data.get("refresh_token")) if data.get("refresh_token") else None,
            expires_at=str(data.get("expires_at")) if data.get("expires_at") else None,
            raw=data,
        )

    def _request_json(self, method: str, url: str) -> Any:
        token_state = self._fresh_token_state()
        request = Request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {token_state.access_token}",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        return self._open_json(request)

    def _fresh_token_state(self) -> SchwabTokenState:
        token_state = self.load_token_state()
        if token_state.expires_at:
            try:
                expires_at = datetime.fromisoformat(token_state.expires_at)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at <= datetime.now(timezone.utc) + timedelta(minutes=2):
                    return self.refresh_access_token()
            except ValueError:
                return self.refresh_access_token()
        return token_state

    def _post_token(self, *, credentials: SchwabCredentials, form: dict[str, str]) -> dict[str, Any]:
        auth = base64.b64encode(f"{credentials.client_id}:{credentials.client_secret}".encode("utf-8")).decode("ascii")
        body = urlencode(form).encode("utf-8")
        request = Request(
            TOKEN_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        return self._open_json(request)

    def _open_json(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=30) as response:
                body = self._decode_response_body(response.read(), response.headers.get("Content-Encoding"))
        except HTTPError as exc:
            details = self._decode_response_body(exc.read(), exc.headers.get("Content-Encoding"))
            raise SchwabError(f"Schwab API request failed with HTTP {exc.code}: {details}") from exc
        if not body:
            return {}
        return json.loads(body)

    @staticmethod
    def _decode_response_body(body: bytes, content_encoding: str | None) -> str:
        if not body:
            return ""
        if content_encoding and "gzip" in content_encoding.lower():
            try:
                body = gzip.decompress(body)
            except OSError:
                pass
        return body.decode("utf-8", errors="replace")

    def _save_token_response(self, response: dict[str, Any]) -> SchwabTokenState:
        if "access_token" not in response:
            raise SchwabError(f"Schwab token response did not include access_token: {sorted(response)}")
        saved = dict(response)
        expires_in = saved.get("expires_in")
        if expires_in is not None:
            saved["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=int(float(expires_in)))
            ).isoformat()
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
        return SchwabTokenState(
            access_token=str(saved["access_token"]),
            refresh_token=str(saved.get("refresh_token")) if saved.get("refresh_token") else None,
            expires_at=str(saved.get("expires_at")) if saved.get("expires_at") else None,
            raw=saved,
        )

    @staticmethod
    def _extract_authorization_code(code_or_url: str) -> str:
        value = code_or_url.strip()
        if not value:
            raise SchwabError("Authorization code is empty.")
        if "://" not in value:
            if value == "...":
                raise SchwabError("Authorization code is still the placeholder '...'. Paste the real code from Schwab.")
            return value
        parsed = urlparse(value)
        code = parse_qs(parsed.query).get("code", [None])[0]
        if not code:
            raise SchwabError("Callback URL did not include a code query parameter.")
        if code == "...":
            raise SchwabError("Callback URL still contains the placeholder code=.... Paste the real Schwab callback URL.")
        return code
