"""Claim Assistant back end: the Notification of Claim form as data, validation, document
checklist, loss-type suggestion and a filled-in PDF.

The fields mirror Claim-Form.pdf ("Notification of Claim - Travel Insurance", 4 pages):
  Insured's information, Claimant's information, Type of loss, Details of injury or sickness,
  Attending physician statement, Official receipts, Payee & bank details,
  Authority / release / declaration, Claims reimbursement checklist.
"FOR EVALUATION PURPOSES (DO NOT FILL-UP)" is left for the insurer and is not collected.
"""
import io
import re
from datetime import date, datetime

from . import config

LOSS_TYPES = [
    "Medical Expenses", "Repatriation Expenses", "Trip Cancellation", "Trip Curtailment",
    "Flight Delay", "Luggage and Personal Effects", "Luggage Delay", "Personal Accident", "Others",
]

COMPULSORY_DOCS = [
    "Duly-accomplished Notification of Claim (NOC)",
    "Copy of Insurance Policy",
    "Request Letter for Reimbursement",
    "Original Official Receipt/s (O.R.) of all payments made",
    "Copy of Passport with Exit/Entry Dates",
]
MEDICAL_DOCS = [
    "Medical Report with Admitting Medical History",
    "Clinical / Laboratory Results",
    "Detailed Statement of Account (itemized)",
    "Copy of Operative Report or Histopathology Report",
    "Copy of Registered Death Certificate (if applicable)",
    "Dental Report (for Emergency Dental Care)",
]
LUGGAGE_DOCS = [
    "Property Irregularity Report (PIR) or Baggage Damage Report",
    "Incident Report from Client",
    "Original Receipts of Basic Necessity Purchased",
]
CANCELLATION_DOCS = [
    "Airline Itinerary / Booking",
    "Certification of Trip Cancellation",
    "Copy of Airline Ticket",
    "Incident Report from Client",
]
FLIGHT_DELAY_DOCS = ["Certificate from the Airline", "Incident Report from Client"]

DOCS_BY_LOSS = {
    "Medical Expenses": MEDICAL_DOCS,
    "Repatriation Expenses": MEDICAL_DOCS,
    "Personal Accident": MEDICAL_DOCS,
    "Trip Cancellation": CANCELLATION_DOCS,
    "Trip Curtailment": CANCELLATION_DOCS,
    "Flight Delay": FLIGHT_DELAY_DOCS,
    "Luggage and Personal Effects": LUGGAGE_DOCS,
    "Luggage Delay": LUGGAGE_DOCS,
}

# Policy reminders shown next to each loss type (with the clause they come from).
POLICY_TIPS = {
    "Medical Expenses": "Call the insurer's hotline before or as soon as possible after admission (Notice of Claim, p. 42). "
                        "Follow-up treatment at home: within 30 days, max 10% of the medical limit (Section 1, p. 13).",
    "Repatriation Expenses": "Evacuation/repatriation must be arranged by the Company's medical team (Section 3, pp. 13-14).",
    "Trip Cancellation": "Insurance must have been bought at least 3 days before departure (Section 9, p. 23).",
    "Trip Curtailment": "Call the hotline for approval before cutting the trip short (Section 10, p. 23).",
    "Flight Delay": "Paid per full 8-hour delay; get the airline's written confirmation of the reason and length (Section 11, pp. 25-26).",
    "Luggage and Personal Effects": "Theft must be reported to the police within 24 hours; items over USD 200 need receipts (pp. 27, 30).",
    "Luggage Delay": "Delay must exceed 6 hours on the outbound trip; buy essentials within 4 days of arrival (Section 17, pp. 28-29).",
    "Personal Accident": "Paid per the Injury Table in Section 7 (pp. 17-19).",
}

# Topic predicted by the chatbot's classifier -> loss type to pre-tick in the form.
TOPIC_TO_LOSS = {
    "medical": "Medical Expenses", "covid": "Medical Expenses", "dental": "Medical Expenses",
    "hospital_family": "Medical Expenses", "evacuation": "Repatriation Expenses",
    "trip_cancellation": "Trip Cancellation", "trip_curtailment": "Trip Curtailment",
    "flight_delay": "Flight Delay", "baggage": "Luggage and Personal Effects",
    "personal_accident": "Personal Accident", "money_documents": "Others",
    "liability_car_rental": "Others",
}
KEYWORDS_TO_LOSS = [
    (r"\b(delay(ed)? (bag|luggage|maleta|bagahe)|hindi dumating|didn'?t arrive|late (bag|luggage))", "Luggage Delay"),
    (r"(lost|nawala|stolen|nanakaw|ninakaw|damaged|nasira).{0,25}(bag|luggage|maleta|bagahe|gamit)|"
     r"(bag|luggage|maleta|bagahe).{0,25}(lost|nawala|stolen|nanakaw|ninakaw|damaged|nasira)", "Luggage and Personal Effects"),
    (r"\b(flight|eroplano|lipad).{0,30}(delay|na-delay|naantala|late)|(delay|na-delay|naantala).{0,30}(flight|eroplano)", "Flight Delay"),
    (r"\b(cancel|kinansela|kanselado|hindi natuloy|di natuloy)", "Trip Cancellation"),
    (r"(umuwi ng maaga|cut short|curtail|return(ed)? early)", "Trip Curtailment"),
    (r"(hospital|ospital|doctor|doktor|clinic|nagkasakit|sick|fever|lagnat|covid|ngipin|dental|confine)", "Medical Expenses"),
    (r"(repatriat|evacuat|mortal remains|labi|bangkay)", "Repatriation Expenses"),
    (r"(accident|aksidente|naaksidente|fracture|bali|injur|disab)", "Personal Accident"),
]


def suggest_loss_types(description: str, topic: str | None = None) -> list[str]:
    """Keyword rules first (precise), then the classifier's topic as a fallback."""
    text = description.lower()
    found = [loss for rx, loss in KEYWORDS_TO_LOSS if re.search(rx, text)]
    if not found and topic in TOPIC_TO_LOSS:
        found = [TOPIC_TO_LOSS[topic]]
    return list(dict.fromkeys(found))


def required_documents(loss_types: list[str]) -> list[str]:
    docs = list(COMPULSORY_DOCS)
    for lt in loss_types:
        for d in DOCS_BY_LOSS.get(lt, []):
            if d not in docs:
                docs.append(d)
    return docs


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _as_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception:
        return None


def validate_claim(c: dict, today: date | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block submission; warnings are policy reminders."""
    today = today or date.today()
    errors, warnings = [], []
    ins = c.get("insured", {})
    for key, label in [("name", "Insured's name"), ("policy_number", "Policy number"),
                       ("address", "Insured's complete address"), ("email", "Insured's e-mail address"),
                       ("mobile", "Insured's mobile number")]:
        if not str(ins.get(key, "")).strip():
            errors.append(f"{label} is required.")
    if ins.get("email") and not EMAIL_RE.match(ins["email"]):
        errors.append("Insured's e-mail address is not valid.")
    if not c.get("loss_types"):
        errors.append("Please check at least one Type of Loss.")
    if "Others" in c.get("loss_types", []) and not str(c.get("loss_other", "")).strip():
        errors.append("Please specify the 'Others' type of loss.")
    if not str(c.get("incident", {}).get("description", "")).strip():
        errors.append("Please describe what happened.")
    receipts = [r for r in c.get("receipts", []) if str(r.get("or_number", "")).strip() or float(r.get("amount") or 0) > 0]
    if not receipts:
        errors.append("Add at least one official receipt (O.R. number, description and amount).")
    for i, r in enumerate(receipts, 1):
        if float(r.get("amount") or 0) <= 0:
            errors.append(f"Receipt {i}: amount must be greater than zero.")
    pay = c.get("payment", {})
    if not str(pay.get("payee_name", "")).strip():
        errors.append("Name of payee is required.")
    if pay.get("method") == "Direct credit to bank account":
        for key, label in [("bank_account_name", "Bank account name"), ("bank_account_number", "Bank account number"),
                           ("bank_name_address", "Bank complete address")]:
            if not str(pay.get(key, "")).strip():
                errors.append(f"{label} is required for direct credit.")
    if not c.get("declaration_agreed"):
        errors.append("You must agree to the Authority, Release and Declaration statement.")
    if not str(c.get("signature_name", "")).strip():
        errors.append("Type your full name as signature.")

    # policy reminders
    d = _as_date(c.get("incident", {}).get("date_of_loss"))
    if d:
        days = (today - d).days
        if days < 0:
            errors.append("Date of loss/assistance cannot be in the future.")
        elif days > 90:
            warnings.append(f"The loss was {days} days ago. The form requires submission within 90 days of the "
                            "assistance; late documents are accepted only if it was not reasonably possible to submit "
                            "on time, and never after 1 year (Proof of Loss, p. 42).")
        elif days > 30:
            warnings.append(f"The loss was {days} days ago. Written notice is due within 30 days of the loss "
                            "(Notice of Claim, p. 42). Submit as soon as possible.")
    if any(t in c.get("loss_types", []) for t in ("Luggage and Personal Effects",)) and not c.get("police_report"):
        warnings.append("For theft or robbery, a police report made within 24 hours is required (pp. 27-28, 42).")
    return errors, warnings


def total_claimed(c: dict) -> float:
    return round(sum(float(r.get("amount") or 0) for r in c.get("receipts", [])), 2)


def reference_number(c: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pol = re.sub(r"[^A-Za-z0-9]", "", c.get("insured", {}).get("policy_number", ""))[-6:] or "NOPOL"
    return f"TS-NOC-{stamp}-{pol.upper()}"


def mask_account(n: str) -> str:
    digits = re.sub(r"\s", "", str(n or ""))
    return ("*" * max(len(digits) - 4, 0) + digits[-4:]) if digits else ""


# ---------------------------------------------------------------------- PDF
def render_claim_pdf(c: dict, ref: str) -> bytes:
    """Draw a filled Notification of Claim as a PDF (ReportLab)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Notification of Claim - Travel Insurance")
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11, textColor=colors.white, spaceBefore=0, spaceAfter=0)
    small = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8.5, leading=11)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    navy = colors.HexColor("#14365c")

    def band(title):
        t = Table([[Paragraph(title, h)]], colWidths=[178 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), navy), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    def grid(rows, widths=(45, 44, 45, 44)):
        data = [[Paragraph(f"<b>{a}</b>", small), Paragraph(str(b or "-"), body),
                 Paragraph(f"<b>{x}</b>", small) if x else "", Paragraph(str(y or "-"), body) if x else ""]
                for a, b, x, y in rows]
        t = Table(data, colWidths=[w * mm for w in widths])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                               ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f7"))]))
        return t

    ins, cl, inc, phy, pay = (c.get(k, {}) for k in ("insured", "claimant", "incident", "physician", "payment"))
    story = [
        Paragraph("<b>Notification of Claim - Travel Insurance</b>", ParagraphStyle("t", parent=ss["Title"], fontSize=15)),
        Paragraph(f"Travel Shield Insurance Policy &middot; {config.INSURER_NAME}", ParagraphStyle("c", parent=small, alignment=1)),
        Paragraph(f"Reference: <b>{ref}</b> &middot; Submitted: {datetime.now():%Y-%m-%d %H:%M} via Travel Shield Claim Assistant",
                  ParagraphStyle("c2", parent=small, alignment=1)),
        Spacer(1, 6),
        band("INSURED'S INFORMATION"),
        grid([("Insured's Name", ins.get("name"), "Age / Sex", f"{ins.get('age', '')} / {ins.get('sex', '')}"),
              ("Policy Number", ins.get("policy_number"), "Complete Address", ins.get("address")),
              ("Home", ins.get("home"), "Office", ins.get("office")),
              ("Mobile", ins.get("mobile"), "Fax", ins.get("fax")),
              ("E-mail Address", ins.get("email"), "", "")]),
        Spacer(1, 5), band("CLAIMANT'S INFORMATION"),
    ]
    if cl.get("same_as_insured"):
        story.append(Paragraph("Claimant is the insured.", body))
    else:
        story.append(grid([("Claimant's Name", cl.get("name"), "Age / Sex", f"{cl.get('age', '')} / {cl.get('sex', '')}"),
                           ("Complete Address", cl.get("address"), "Birthday", cl.get("birthday")),
                           ("Relationship to Insured", cl.get("relationship"), "Mobile", cl.get("mobile")),
                           ("Home", cl.get("home"), "Office", cl.get("office"))]))
    losses = ", ".join(c.get("loss_types", [])) + (f" (Others: {c.get('loss_other')})" if c.get("loss_other") else "")
    story += [Spacer(1, 5), band("TYPE OF LOSS"), Paragraph(losses or "-", body), Spacer(1, 5),
              band("DETAILS OF INJURY OR SICKNESS / INCIDENT"),
              grid([("Date of loss / assistance", inc.get("date_of_loss"), "Country / place", inc.get("place")),
                    ("Nature and condition", inc.get("nature"), "Hospitalization / consultation dates", inc.get("hospital_dates")),
                    ("Name of hospital", inc.get("hospital_name"), "Attending physician", inc.get("physician_name")),
                    ("Hospital address", inc.get("hospital_address"), "Hospital tel. / fax", inc.get("hospital_phone")),
                    ("Prior treatment of same illness", inc.get("prior_treatment"), "", "")]),
              Paragraph(f"<b>What happened:</b> {inc.get('description', '')}", body)]
    if phy.get("applicable"):
        story += [Spacer(1, 5), band("ATTENDING PHYSICIAN STATEMENT (as provided by claimant)"),
                  grid([("Patient type", phy.get("patient_type"), "Complete diagnosis", phy.get("diagnosis")),
                        ("Admission date", phy.get("admission_date"), "Discharge date", phy.get("discharge_date")),
                        ("Date of consultation", phy.get("consultation_date"), "Continuous treatment for chronic disease?", phy.get("chronic")),
                        ("Other diseases affecting condition?", phy.get("other_disease"), "If yes, describe", phy.get("other_disease_details"))]),
                  Paragraph("<i>The physician's signature is collected on the original form submitted with the documents.</i>", small)]
    rows = [[Paragraph("<b>O.R. Number</b>", small), Paragraph("<b>Description</b>", small), Paragraph("<b>Currency</b>", small),
             Paragraph("<b>Amount</b>", small)]]
    for r in c.get("receipts", []):
        rows.append([str(r.get("or_number", "")), Paragraph(str(r.get("description", "")), body), str(r.get("currency", "")),
                     f"{float(r.get('amount') or 0):,.2f}"])
    rows.append(["", Paragraph("<b>TOTAL AMOUNT CLAIMED</b>", body), "", f"{total_claimed(c):,.2f}"])
    rt = Table(rows, colWidths=[35 * mm, 93 * mm, 20 * mm, 30 * mm])
    rt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")), ("ALIGN", (3, 0), (3, -1), "RIGHT")]))
    story += [Spacer(1, 5), band("OFFICIAL RECEIPTS SUBMITTED"), rt,
              Paragraph(f"Total amount claimed in words: {c.get('total_in_words', '')}", small), Spacer(1, 5),
              band("PAYMENT DETAILS"),
              grid([("Name of payee (as on check)", pay.get("payee_name"), "Relationship (if not insured)", pay.get("payee_relationship")),
                    ("Mode of payment", pay.get("method"), "Account type", pay.get("account_type")),
                    ("Bank account name", pay.get("bank_account_name"), "Bank account number", pay.get("bank_account_number")),
                    ("Bank complete address", pay.get("bank_name_address"), "Relationship to patient (if other account)", pay.get("bank_relationship"))]),
              Spacer(1, 5), band("AUTHORITY, RELEASE AND DECLARATION"),
              Paragraph("The claimant agreed to the Authority, Release and Subrogation, and Declaration statements of the "
                        "Notification of Claim form, authorizing the insurer to obtain medical records related to this claim and "
                        "declaring that all statements are complete and true.", small),
              grid([("Signature over printed name", f"/s/ {c.get('signature_name', '')} (typed e-signature)", "Date", c.get("signature_date"))]),
              Spacer(1, 5), band("CLAIMS REIMBURSEMENT CHECKLIST"),
              Paragraph("<br/>".join(("[x] " if d in c.get("documents_ready", []) else "[ ] ") + d
                                     for d in required_documents(c.get("loss_types", []))), body),
              Paragraph(f"<b>Attached files:</b> {', '.join(c.get('attachment_names', [])) or 'none'}", small),
              Spacer(1, 6),
              Paragraph("Submission of documents does not guarantee approval; claims are evaluated subject to the limits, terms "
                        "and conditions of the Travel Policy. FOR EVALUATION PURPOSES section to be completed by the insurer.", small)]
    doc.build(story)
    return buf.getvalue()


def claim_email_bodies(c: dict, ref: str) -> tuple[str, str]:
    """Plain-text and HTML e-mail bodies. The bank account number is masked in the body; the full
    value is only in the attached PDF."""
    ins, inc, pay = c.get("insured", {}), c.get("incident", {}), c.get("payment", {})
    lines = [
        f"New travel insurance claim: {ref}", "",
        f"Insured: {ins.get('name')}  |  Policy No.: {ins.get('policy_number')}",
        f"Contact: {ins.get('mobile')}  |  {ins.get('email')}",
        f"Type of loss: {', '.join(c.get('loss_types', []))}",
        f"Date of loss: {inc.get('date_of_loss')}  |  Place: {inc.get('place')}",
        f"Total amount claimed: {total_claimed(c):,.2f}",
        f"Payment: {pay.get('method')}  |  Payee: {pay.get('payee_name')}"
        + (f"  |  Account: {mask_account(pay.get('bank_account_number'))}" if pay.get("bank_account_number") else ""),
        "", "What happened:", str(inc.get("description", "")), "",
        f"Attachments: {', '.join(['Notification of Claim (PDF)'] + c.get('attachment_names', []))}",
        "", "Sent by the Travel Shield Claim Assistant. Original receipts must still be submitted as required by the policy.",
    ]
    text = "\n".join(lines)
    rows = "".join(f"<tr><td style='padding:4px 10px;color:#555'>{k}</td><td style='padding:4px 10px'><b>{v}</b></td></tr>"
                   for k, v in [("Reference", ref), ("Insured", ins.get("name")), ("Policy No.", ins.get("policy_number")),
                                ("Contact", f"{ins.get('mobile')} / {ins.get('email')}"),
                                ("Type of loss", ", ".join(c.get("loss_types", []))),
                                ("Date / place of loss", f"{inc.get('date_of_loss')} / {inc.get('place')}"),
                                ("Total claimed", f"{total_claimed(c):,.2f}"),
                                ("Payment", f"{pay.get('method')} - {pay.get('payee_name')} {mask_account(pay.get('bank_account_number'))}")])
    html = (f"<div style='font-family:Arial,sans-serif;font-size:14px'><h2 style='color:#14365c'>Travel Shield - Notification of Claim</h2>"
            f"<table style='border-collapse:collapse'>{rows}</table><p><b>What happened:</b><br>{inc.get('description', '')}</p>"
            f"<p style='color:#777;font-size:12px'>The completed Notification of Claim is attached as a PDF, with the claimant's "
            f"supporting files. Original receipts must still be submitted as required by the policy.</p></div>")
    return text, html
