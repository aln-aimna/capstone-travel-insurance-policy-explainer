"""The customer's Schedule of Coverages (benefit limits).

The policy wording (Travel-Shield-ePolicy.pdf) says most benefits are paid "up to the amount shown in
the Policy Schedule". data/plans/schedule_of_coverages.csv holds one such schedule, taken from the
SCHEDULE OF COVERAGES table of a real Certificate of Insurance (benefit amounts only, no personal data).
All amounts are in Philippine Pesos.

Used by:
  * the chatbot prompt  -> Claude can quote the actual limit next to the policy conditions
  * offline mode        -> "magkano / how much" questions get the limit from this table
  * the Claim Assistant -> shows the limit for each type of loss and warns if the claim is above it
"""
import re

import pandas as pd

from . import config

SCHEDULE_FILE = config.ROOT / "data" / "plans" / "schedule_of_coverages.csv"
ASSISTANCE_HOTLINE = "63 2 1234 1234" #63 2 8459 4734 (24-hour worldwide travel assistance, call collect)"

AMOUNT_RE = re.compile(r"\b(how much|magkano|limit|maximum|max|hanggang magkano|ilan ang|sum insured|"
                       r"amount|halaga|benefit amount|payout|ma-claim ko|makukuha ko)\b", re.I)


def load_schedule() -> pd.DataFrame:
    if not SCHEDULE_FILE.exists():
        return pd.DataFrame(columns=["benefit", "policy_section", "topic", "claim_loss_type", "max_php", "limit_text", "pages"])
    df = pd.read_csv(SCHEDULE_FILE)
    df["claim_loss_type"] = df["claim_loss_type"].fillna("")
    return df


def schedule_text(df: pd.DataFrame | None = None) -> str:
    """Compact table for the system prompt."""
    df = load_schedule() if df is None else df
    if df.empty:
        return ""
    return "\n".join(f"- {r.benefit} ({r.policy_section}): {r.limit_text}" for r in df.itertuples())


def is_amount_question(text: str) -> bool:
    return bool(AMOUNT_RE.search(text))


def rows_for_topic(topic: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_schedule() if df is None else df
    if topic == "covid":
        topic = "medical"
    return df[df.topic == topic]


def limit_for_loss_types(loss_types: list[str], df: pd.DataFrame | None = None) -> list[dict]:
    """Main benefit limit for each Type of Loss ticked in the claim form."""
    df = load_schedule() if df is None else df
    out = []
    for lt in loss_types:
        rows = df[df.claim_loss_type == lt]
        if rows.empty:
            continue
        main = rows.iloc[0]            # first row = the benefit that type of loss is mainly paid under
        out.append({"loss_type": lt, "benefit": main.benefit, "max_php": float(main.max_php),
                    "limit_text": main.limit_text, "section": main.policy_section})
    return out
