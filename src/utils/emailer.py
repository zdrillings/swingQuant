from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from src.settings import RuntimeSettings


def _is_valid_email_address(value: str) -> bool:
    if not value or " " in value:
        return False
    local_part, separator, domain = value.partition("@")
    if separator != "@":
        return False
    if not local_part or not domain:
        return False
    if "." not in domain:
        return False
    return True


def _validate_email_settings(settings: RuntimeSettings) -> None:
    required_keys = ("GMAIL_USER", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL")
    missing = [key for key in required_keys if not settings.env.get(key)]
    if missing:
        raise ValueError(f"Missing required email settings in .env: {', '.join(missing)}")

    invalid_addresses = [
        key
        for key in ("GMAIL_USER", "RECIPIENT_EMAIL")
        if not _is_valid_email_address(settings.env[key])
    ]
    if invalid_addresses:
        details = ", ".join(f"{key}={settings.env[key]!r}" for key in invalid_addresses)
        raise ValueError(f"Invalid email address in .env: {details}")


def send_html_email(
    *,
    subject: str,
    html_body: str,
    settings: RuntimeSettings,
) -> None:
    _validate_email_settings(settings)

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.env["GMAIL_USER"]
    message["To"] = settings.env["RECIPIENT_EMAIL"]
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(settings.env["GMAIL_USER"], settings.env["GMAIL_APP_PASSWORD"])
        server.sendmail(
            settings.env["GMAIL_USER"],
            [settings.env["RECIPIENT_EMAIL"]],
            message.as_string(),
        )
