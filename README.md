# 🛡️ Travel Shield Policy Explainer & Claim Assistant

**Uplift Code Camp capstone · Banking, Finance & Insurance sector**

A trilingual (English · Tagalog · Taglish) AI assistant for Prudential Guarantee's **Travel Shield** travel
insurance. It explains the 47-page policy in plain language with page citations, quotes the customer's own
benefit limits, and turns the paper *Notification of Claim* form into a guided online form that e-mails the
completed claim to the claims team.

> **Problem.** Filipino travellers buy travel insurance without understanding it. The policy is 47 pages of legal
> English, the claim form is a 4-page PDF, and the questions come in at the worst moment (lost luggage at
> Incheon, a hospital in Tokyo), often in Tagalog or Taglish. Customers miss deadlines (30-day notice,
> 90-day proof of loss), forget documents, and flood customer care with questions the policy already answers.

## What it does

The app has four tabs:

| Tab | What it does |
|---|---|
| 💬 **Ask about your policy** | Answers in the customer's own register (English / Tagalog / Taglish), grounded only in the policy, citing *[Section, p.]*. Quotes the plan's limits from the Schedule of Coverages. Off-topic questions are declined without an API call. Example questions are in the sidebar. |
| 📝 **Claim Assistant** | Every section of the official Notification of Claim (insured, claimant, type of loss, injury/sickness details, physician statement, official receipts, payee & bank, declaration). Suggests the type of loss from a free-text description, shows the limit and the exact document checklist for each type of loss, warns about the 30/90-day deadlines and about receipts above the limit, renders a filled PDF and e-mails it with the attachments. |
| 📋 **Coverages** | The Schedule of Coverages: maximum amount per benefit (PHP) and the policy section it comes from, with a search box. |
| 📄 **Policy** | The full policy wording: search clauses in any of the three languages, or browse all 81 sections by page; download the original PDF. Lets anyone check a chat citation against the source. |

The sidebar shows whether Claude is connected, the session's API cost, and the 24-hour travel assistance hotline
(**+63 2 8459 4734**, call collect).

## How it works

```
question ──► language ID (char n-gram LR) ──► topic classifier (18 topics) ──► guardrail / claim hand-off
                 english|tagalog|taglish          words+chars+glossary, weak labels
                         │
                         ▼
   hybrid cross-lingual retrieval:  BM25 over 248 section-tagged policy chunks
                                  + Tagalog→English insurance glossary expansion
                                  + QA bank: nearest checked Q&A (any language) boosts its cited pages
                         │
                         ▼
   Claude (claude-haiku-5-5): grounded answer, [Section, p.] citations, plan limits from the Schedule
   of Coverages, same language as the customer
```

**Offline fallback** (no API key, or Claude unavailable). The bot answers from the checked Q&A dataset in three tiers:

1. Clear match: the reference answer.
2. Unsure: the 2-3 closest answers, each under its question.
3. Nothing related: the two most relevant policy clauses with page numbers, plus the insurer's contacts.

For "how much / magkano" questions it also shows the matching limits from the Schedule of Coverages.

On 236 held-out everyday phrasings, offline mode answers confidently 55% of the time, shows related answers 29% and
falls back to clauses 17%. Claude answers questions outside the dataset as well.

**Thinking.** Requests ask Claude to skip extended thinking, so the output budget goes to the visible answer. If a
model rejects that setting, the bot retries without it. An empty reply falls back to the reference answer.

## Results (held-out test set: 19 unseen items × 3 languages = 57 questions)

| Component | Baseline | This project |
|---|---|---|
| Language ID accuracy | rules 82.5% | **96.5%** (char n-grams + rule features); English vs Filipino 100% |
| Topic accuracy, 18 classes, unseen items | majority 10.5% · Naive Bayes 29.8% | **45.6%** (LogReg + weak labels from policy sections) |
| Retrieval Hit@3, unseen questions | TF-IDF 63% | **76%** (BM25 + glossary + QA bank) |
| … Tagalog questions only | BM25 56% | **89%** |
| Retrieval Hit@3, everyday FAQ paraphrases (n = 232) | TF-IDF 49% | **71%** |
| Estimated Claude cost (Haiku, ~2.2k input tokens) | | **≈ $0.35 per 1,000 answers** |

Charts: `results/charts/`. End-to-end answer quality (Claude-as-judge, citation validity, language match):
`python scripts/eval_llm.py` with your API key → `results/llm_metrics.json`.
Test sets are small (18-19 items per language), so treat differences of a few points as noise.

## Data

* `data/raw/Travel-Shield-ePolicy.pdf`, `data/raw/Claim-Form.pdf` — the source documents.
* `data/qa/` — the **input/output dataset** written from the policy: 95 items × 3 languages = **285 Q&A pairs**,
  each citing section and page, plus 236 extra everyday phrasings for the classifier. See
  [`data/qa/DATASET_CARD.md`](data/qa/DATASET_CARD.md). Files: `policy_qa_dataset.csv` (input → output),
  `classifier_dataset.csv`, `train.jsonl` / `test.jsonl` (chat format for few-shot or fine-tuning).
  To add or fix an item, edit `data/qa/seed_part*.py` and run `python scripts/build_dataset.py`.
* `data/plans/schedule_of_coverages.csv` — the **benefit limits** (PHP) from the *Schedule of Coverages* of a
  Travel Shield Certificate of Insurance (amounts only, no personal data). The policy wording says most benefits are
  paid "up to the amount shown in the Policy Schedule"; this table supplies those amounts. Limits paid per period are
  stored as totals (e.g. flight delay PHP 2,000 per 8 hours up to 64 hours = PHP 16,000).
  For a different plan, replace the numbers in this file.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt               # "python -m pip" installs into the Python that runs Streamlit

python scripts/build_dataset.py     # data/qa/*  (already built)
python -m src.ingest                # data/processed/policy_chunks.jsonl (already built)
python scripts/train_models.py      # models/*.joblib
python scripts/evaluate.py          # results/metrics.json + charts
python -m pytest tests              # smoke tests (claims, e-mail, PDF, chatbot)

copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then fill in the keys
streamlit run app.py
```

Stop the app with **Ctrl + C** in its terminal. After changing code or secrets, restart it and refresh the browser.

### Secrets (`.streamlit/secrets.toml`)

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude answers (optional: offline mode works without it). Also read from a `.env` file. |
| `SMTP_HOST`, `SMTP_PORT` | the mail server; port 587 (STARTTLS) or 465 (SSL) |
| `SMTP_USER`, `SMTP_PASSWORD` | the account that **sends** the claim e-mail (its login) |
| `CLAIMS_TO` | the address that **receives** claims (comma-separated allowed) |
| `SEND_COPY_TO_CLAIMANT` | `true` to CC the customer's e-mail from the form |

Without SMTP settings the app still validates the claim and lets the customer download the filled PDF.

**Gmail.** Use `smtp.gmail.com`, port 587 and an **App Password**, not your normal password. To get one, turn on
2-Step Verification, then go to Google Account → Security → App passwords. Paste the 16 letters without spaces.
School/work accounts and very new accounts may not offer App Passwords.

**Mailtrap (for demos).** Free; it catches every e-mail in a web inbox, attachments included, and delivers nothing
to real addresses:

```toml
SMTP_HOST = "sandbox.smtp.mailtrap.io"
SMTP_PORT = 587
SMTP_USER = "<username from Mailtrap>"
SMTP_PASSWORD = "<password from Mailtrap>"
CLAIMS_TO = "claims@travelshield-demo.com"
```

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Sidebar says *Offline mode* | No `ANTHROPIC_API_KEY` found. Save `secrets.toml`, restart Streamlit. |
| *(Claude unavailable: AuthenticationError)* | The key was rejected. Test it outside the app; check the file is saved and no old `ANTHROPIC_API_KEY` Windows variable overrides it. |
| Claude answers but no text appears | Fixed: thinking is disabled and empty replies fall back to the reference answer. Update `src/chatbot.py`. |
| `ModuleNotFoundError: reportlab` | `python -m pip install -r requirements.txt` |
| `SMTPServerDisconnected` | Wrong host/port pair, or the network blocks outgoing mail. Use `smtp.gmail.com` + 587 (or 465); try a phone hotspot. |
| `SMTPAuthenticationError` | Gmail needs an App Password, not the account password. |
| Code changes don't show up | An editor saved its older copy over the file. Close the tab without saving, reopen, restart Streamlit. |

## Repository

```
app.py                     Streamlit UI (chat · claim assistant · coverages · policy)
src/ingest.py              PDF → clean pages → section-aware chunks
src/language.py            rule + ML language ID (english / tagalog / taglish)
src/glossary.py            Tagalog/Taglish → English insurance glossary
src/retriever.py           BM25/TF-IDF + glossary + QA-bank hybrid retriever
src/topic_classifier.py    18-topic classifier + weak labels from policy sections
src/prompts.py             versioned system prompt (v3, with the Schedule of Coverages) and user-turn builder
src/chatbot.py             pipeline, guardrails, Claude call (streaming, thinking off), tiered offline fallback, cost tracking
src/plans.py               Schedule of Coverages: limits per benefit / topic / type of loss, assistance hotline
src/claims.py              claim form schema, validation, document checklist, PDF, e-mail bodies
src/emailer.py             SMTP (STARTTLS/SSL) sender with attachments
scripts/                   build_dataset · train_models · evaluate · eval_llm
notebooks/                 01 data & dataset · 02 language & topic models · 03 retrieval & RAG
data/ models/ results/ tests/ docs/ presentation/
```

Reused from earlier bootcamp work: the chunking approach and streaming chat loop from `full_app.py`, and the
cached, cost-logged Claude calls (including turning thinking off) from the sentiment project's `src/llm.py`.

## Limitations & next steps

* One Schedule of Coverages is built in. A next step is letting each customer upload their own Certificate of
  Insurance so the limits match their plan.
* The dataset was written by one author; a native-speaker and an underwriter review of the answers is recommended.
* The claims mailbox receives personal and bank data: use an official, TLS-only mailbox, and in production replace
  e-mail with the insurer's claims API and add authentication.
* The bot explains the policy; it never approves or denies claims. Limits are maximums, not promised payouts.

*Not affiliated with Prudential Guarantee and Assurance, Inc. Built for an educational capstone using the
publicly provided policy wording.*
