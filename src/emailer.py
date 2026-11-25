"""Send the claim e-mail over SMTP (TLS), with the claim PDF and uploaded files attached.

Settings come from Streamlit secrets (.streamlit/secrets.toml) or environment variables:

    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587                 # 587 = STARTTLS, 465 = SSL
    SMTP_USER = "your.address@gmail.com"
    SMTP_PASSWORD = "16-char app password"     # Gmail: Google Account > Security > App passwords
    CLAIMS_TO = "claims-team@example.com"      # where claims go (comma-separated allowed)
    SEND_COPY_TO_CLAIMANT = true

Nothing is logged; the password is read only when sending.
"""
import mimetypes
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid


@dataclass
class SMTPSettings:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    claims_to: str = ""
    sender_name: str = "Travel Shield Claim Assistant"
    copy_to_claimant: bool = True

    @property
    def configured(self) -> bool:
        return all([self.host, self.port, self.user, self.password, self.claims_to])

    @classmethod
    def load(cls, secrets=None) -> "SMTPSettings":
        def get(key, default=""):
            try:
                if secrets is not None and key in secrets:
                    return secrets[key]
            except Exception:
                pass
            return os.getenv(key, default)
        copy = str(get("SEND_COPY_TO_CLAIMANT", "true")).lower() in ("1", "true", "yes")
        return cls(host=get("SMTP_HOST"), port=int(get("SMTP_PORT", 587) or 587), user=get("SMTP_USER"),
                   password=get("SMTP_PASSWORD"), claims_to=get("CLAIMS_TO"), copy_to_claimant=copy)


def build_message(settings: SMTPSettings, subject: str, text: str, html: str, attachments: list[tuple[str, bytes]],
                  claimant_email: str | None = None) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{settings.sender_name} <{settings.user}>"
    msg["To"] = settings.claims_to
    if claimant_email and settings.copy_to_claimant:
        msg["Cc"] = claimant_email
        msg["Reply-To"] = claimant_email
    msg["Message-ID"] = make_msgid(domain=(settings.user.split("@")[-1] or "localhost"))
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    for name, data in attachments:
        ctype, _ = mimetypes.guess_type(name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return msg


def send(settings: SMTPSettings, msg: EmailMessage, timeout: int = 30) -> str:
    """Send and return the Message-ID. Raises on failure (the app shows the error)."""
    context = ssl.create_default_context()
    if int(settings.port) == 465:
        with smtplib.SMTP_SSL(settings.host, settings.port, context=context, timeout=timeout) as s:
            s.login(settings.user, settings.password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(settings.host, settings.port, timeout=timeout) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(settings.user, settings.password)
            s.send_message(msg)
    return msg["Message-ID"]
