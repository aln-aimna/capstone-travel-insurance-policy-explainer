"""Smoke tests for the non-UI parts. Run: python -m pytest tests  (or: python tests/test_core.py)"""
import smtplib
import sys
from datetime import date, timedelta
from email import message_from_bytes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import claims, emailer  # noqa: E402
from src.chatbot import PolicyChatbot  # noqa: E402
from src.glossary import expand_query  # noqa: E402
from src.language import RuleLanguageDetector  # noqa: E402


def sample_claim(**over):
    c = {
        "insured": {"name": "Juan Dela Cruz", "age": 34, "sex": "Male", "policy_number": "TS-2026-00123",
                    "address": "Antipolo City, Rizal", "mobile": "0917 000 0000", "email": "juan@example.com"},
        "claimant": {"same_as_insured": True},
        "loss_types": ["Flight Delay"], "loss_other": "",
        "incident": {"description": "Na-delay ng 10 oras ang flight ko sa Incheon dahil sa bagyo.",
                     "date_of_loss": str(date.today() - timedelta(days=5)), "place": "Incheon, South Korea"},
        "physician": {"applicable": False},
        "receipts": [{"or_number": "OR-1001", "description": "Airport hotel", "currency": "KRW", "amount": 85000},
                     {"or_number": "OR-1002", "description": "Meals", "currency": "KRW", "amount": 23000}],
        "payment": {"payee_name": "Juan Dela Cruz", "method": "Direct credit to bank account",
                    "bank_account_name": "Juan Dela Cruz", "bank_account_number": "001234567890",
                    "bank_name_address": "BPI Antipolo", "account_type": "Savings"},
        "documents_ready": ["Certificate from the Airline"], "attachment_names": ["airline_cert.pdf"],
        "declaration_agreed": True, "signature_name": "Juan Dela Cruz", "signature_date": str(date.today()),
    }
    c.update(over)
    return c


def test_valid_claim_passes():
    errors, warnings = claims.validate_claim(sample_claim())
    assert errors == [], errors
    assert claims.total_claimed(sample_claim()) == 108000


def test_missing_fields_and_deadline_warning():
    c = sample_claim(declaration_agreed=False, receipts=[])
    c["incident"]["date_of_loss"] = str(date.today() - timedelta(days=120))
    errors, warnings = claims.validate_claim(c)
    assert any("Declaration" in e for e in errors)
    assert any("receipt" in e.lower() for e in errors)
    assert any("90 days" in w for w in warnings)


def test_documents_and_suggestions():
    docs = claims.required_documents(["Luggage Delay"])
    assert "Property Irregularity Report (PIR) or Baggage Damage Report" in docs
    assert docs[0] == "Duly-accomplished Notification of Claim (NOC)"
    assert claims.suggest_loss_types("Hindi dumating yung maleta ko") == ["Luggage Delay"]
    assert "Flight Delay" in claims.suggest_loss_types("na-delay ng 9 oras ang flight")
    assert claims.suggest_loss_types("something unclear", topic="medical") == ["Medical Expenses"]


def test_pdf_and_email_message():
    c = sample_claim()
    ref = claims.reference_number(c)
    pdf = claims.render_claim_pdf(c, ref)
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000
    text, html = claims.claim_email_bodies(c, ref)
    assert "001234567890" not in text and "********7890" in text      # account masked in body
    s = emailer.SMTPSettings(host="smtp.test", port=587, user="bot@test.ph", password="x", claims_to="claims@test.ph")
    msg = emailer.build_message(s, f"[Travel Shield Claim] {ref}", text, html,
                                [(f"{ref}.pdf", pdf), ("receipt.jpg", b"\xff\xd8fake")], claimant_email="juan@example.com")
    parsed = message_from_bytes(msg.as_bytes())
    names = [p.get_filename() for p in parsed.walk() if p.get_filename()]
    assert names == [f"{ref}.pdf", "receipt.jpg"]
    assert parsed["Cc"] == "juan@example.com"


def test_send_uses_starttls_and_login(monkeypatch=None):
    calls = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None): calls.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self, context=None): calls.append(("starttls",))
        def login(self, u, p): calls.append(("login", u))
        def send_message(self, m): calls.append(("send", m["To"]))

    orig = smtplib.SMTP
    smtplib.SMTP = FakeSMTP
    try:
        s = emailer.SMTPSettings(host="smtp.gmail.com", port=587, user="bot@test.ph", password="x", claims_to="claims@test.ph")
        msg = emailer.build_message(s, "subj", "t", "<p>h</p>", [])
        emailer.send(s, msg)
    finally:
        smtplib.SMTP = orig
    assert [c[0] for c in calls] == ["connect", "starttls", "login", "send"]


def test_language_and_glossary():
    r = RuleLanguageDetector()
    assert r.predict_one("What does the policy cover if my luggage is lost?") == "english"
    assert r.predict_one("Ano ang dapat kong gawin kung nawala ang aking pasaporte?") in ("tagalog", "taglish")
    assert "baggage" in expand_query("Nawala ang bagahe ko")
    assert "delay" in expand_query("na-delay yung flight")


def test_chatbot_offline_answers_in_users_language():
    bot = PolicyChatbot(api_key="")
    r = bot.answer("Nawala ng airline yung maleta ko, ano pwede ko i-claim?")
    assert r.mode == "offline" and r.topic == "baggage"
    assert any(c["page"] in (26, 27) for c in r.chunks)
    oos = bot.answer("What is the best restaurant in Seoul?")
    assert oos.mode == "guardrail"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
