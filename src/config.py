"""Paths and constants shared by the app, scripts and notebooks."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_QA = ROOT / "data" / "qa"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

POLICY_PDF = DATA_RAW / "Travel-Shield-ePolicy.pdf"
CLAIM_FORM_PDF = DATA_RAW / "Claim-Form.pdf"
CHUNKS_FILE = DATA_PROCESSED / "policy_chunks.jsonl"
QA_ITEMS_FILE = DATA_QA / "policy_qa_items.json"
QA_FILE = DATA_QA / "policy_qa_dataset.csv"
CLASSIFIER_FILE = DATA_QA / "classifier_dataset.csv"
TOPIC_MODEL_FILE = MODELS / "topic_classifier.joblib"
LANG_MODEL_FILE = MODELS / "language_classifier.joblib"
LLM_CACHE = RESULTS / "llm_cache.jsonl"

SEED = 42

# Claude model. Haiku keeps the demo fast and cheap; switch to Sonnet for harder questions.
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
# USD per million tokens (platform.claude.com pricing page, checked Nov 2026 - same table as the
# earlier sentiment project). Update if you change model or prices change.
PRICE_PER_MTOK = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

LANGUAGES = ["english", "tagalog", "taglish"]

TOPICS = [
    "medical", "covid", "dental", "evacuation", "hospital_family", "personal_accident",
    "trip_cancellation", "trip_curtailment", "flight_delay", "baggage", "money_documents",
    "liability_car_rental", "exclusions", "eligibility", "claims", "policy_admin",
    "optional_benefits", "out_of_scope",
]

TOPIC_LABELS = {
    "medical": "Medical expenses", "covid": "COVID-19", "dental": "Dental",
    "evacuation": "Evacuation & repatriation", "hospital_family": "Hospital & family support",
    "personal_accident": "Personal accident", "trip_cancellation": "Trip cancellation",
    "trip_curtailment": "Trip curtailment", "flight_delay": "Flight delay & misconnection",
    "baggage": "Baggage", "money_documents": "Money & travel documents",
    "liability_car_rental": "Liability & car rental", "exclusions": "Exclusions",
    "eligibility": "Eligibility & coverage period", "claims": "Claims process",
    "policy_admin": "Policy administration", "optional_benefits": "Optional benefits",
    "out_of_scope": "Out of scope",
}

# Public contact details printed in the policy footer (pp. 1-47).
INSURER_NAME = "Prudential Guarantee and Assurance, Inc."
INSURER_PHONES = "(632) 1234-5678"#(632) 8810-4916 / 8878-3000 / 8651-9500"
INSURER_EMAIL = "testmail@gmail.com"#"customercare@prudentialguarantee.com"

for _p in (DATA_PROCESSED, DATA_QA, MODELS, RESULTS, CHARTS):
    _p.mkdir(parents=True, exist_ok=True)
