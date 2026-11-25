"""Question-topic classifier (18 classes) for routing and analytics.

What the topic is used for in the app:
  * out_of_scope -> polite refusal without spending an API call
  * claims       -> show the "open Claim Assistant" button
  * every topic  -> pre-ticks the Type of Loss in the claim form (TOPIC_TO_LOSS) and feeds the
                    analytics tab (what customers ask about most)

Features: word 1-2-gram TF-IDF on the glossary-expanded text (so Tagalog words also produce their
English equivalents) + char_wb 2-5-gram TF-IDF (robust to Taglish spelling: "covered ba",
"kovered", "na-delay"). Model: logistic regression. Baselines in scripts/evaluate.py:
majority class and multinomial Naive Bayes on words only.
"""
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .glossary import expand_query


def _expand_all(texts):
    return [expand_query(t).lower() for t in texts]


# Weak supervision: the policy's own structure labels its text. Every chunk sits under a heading
# ("SECTION 14: LOSS OR DAMAGE OF BAGGAGE...") that maps to one topic, so the 248 chunks become
# ~230 extra (English) training examples for free. First matching rule wins.
SECTION_TO_TOPIC = [
    (r"^Section 2\b", "dental"),
    (r"^Section 1\b|Prescribed Medicines", "medical"),
    (r"Covid|COVID", "covid"),
    (r"^Section 3\b", "evacuation"),
    (r"^Section (4|5|6|19)\b", "hospital_family"),
    (r"^Section (7|8)\b|Personal Accident Benefit \(Section 7", "personal_accident"),
    (r"^Section 9\b|Trip Cancellation Benefit \(Section 9", "trip_cancellation"),
    (r"^Section 10\b", "trip_curtailment"),
    (r"^Section (11|12|13)\b", "flight_delay"),
    (r"^Section (14|17)\b|Baggage", "baggage"),
    (r"^Section (15|16)\b", "money_documents"),
    (r"^Section (18|20)\b|Liability Benefit|Car Rental", "liability_car_rental"),
    (r"Exclusions Applicable To All|Medical and Emergency Expenses \(Section 1", "exclusions"),
    (r"Definitions|Effective|Limits Of Coverage|Geographical|Age Limitation", "eligibility"),
    (r"Notice of Claim|Proof of Loss|Claim Forms|To Whom|Time of Payment|Cumulative|Subrogation|Obligations", "claims"),
    (r"Premium|Grace|Cancellation$|Free Look|Fraud|Arbitration|Legal Action|Renewal|Currency|Termination", "policy_admin"),
    (r"^Optional", "optional_benefits"),
]


def weak_labels_from_policy(chunks: list[dict], max_chars: int = 500) -> pd.DataFrame:
    rows = []
    for c in chunks:
        topic = next((t for rx, t in SECTION_TO_TOPIC if re.search(rx, c["section"])), None)
        if topic:
            rows.append({"text": c["text"][:max_chars], "topic": topic})
    return pd.DataFrame(rows)


def build_topic_model(C: float = 8.0) -> Pipeline:
    features = FeatureUnion([
        ("words", Pipeline([
            ("expand", FunctionTransformer(_expand_all)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ])),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                  min_df=1, lowercase=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=5000, C=C, class_weight="balanced")),
    ])


def predict_with_confidence(model, text: str) -> tuple[str, float]:
    proba = model.predict_proba([text])[0]
    i = proba.argmax()
    return model.classes_[i], float(proba[i])
