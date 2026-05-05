from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from src.settings import RuntimeSettings


def send_html_email(
    *,
    subject: str,
    html_body: str,
    settings: RuntimeSettings,
) -> None:
    required_keys = ("GMAIL_USER", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL")
    missing = [key for key in required_keys if not settings.env.get(key)]
    if missing:
        raise ValueError(f"Missing required email settings in .env: {', '.join(missing)}")

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
