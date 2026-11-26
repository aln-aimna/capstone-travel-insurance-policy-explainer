# 🛡️ Travel Shield Policy Explainer & Claim Assistant

**Uplift Code Camp capstone · Banking, Finance & Insurance sector**

A trilingual (English · Tagalog · Taglish) AI assistant for Prudential Guarantee's **Travel Shield** travel
insurance. It explains the 47-page policy in plain language with page citations, and turns the paper
*Notification of Claim* form into a guided online form that e-mails the completed claim to the claims team.

> **Problem.** Filipino travellers buy travel insurance without understanding it. The policy is 47 pages of legal
> English, the claim form is a 4-page PDF, and the questions come in at the worst moment (lost luggage at
> Incheon, a hospital in Tokyo), often in Tagalog or Taglish. Customers miss deadlines (30-day notice,
> 90-day proof of loss), forget documents, and flood customer care with questions the policy already answers.

## What it does

| | |
|---|---|
| 💬 **Policy chat** | Answers in the customer's own register (English / Tagalog / Taglish), grounded only in the policy, citing *[Section, p.]*. Refuses to invent amounts that live in the customer's Policy Schedule. Out-of-scope questions are declined without an API call. |
| 📝 **Claim Assistant** | Every section of the official Notification of Claim (insured, claimant, type of loss, injury/sickness details, physician statement, official receipts, payee & bank, declaration). Suggests the type of loss from a free-text description, shows the exact document checklist and the policy deadline rules, validates, renders a filled PDF and e-mails it with the attachments (SMTP/TLS). |
| 📊 **Insights** | Held-out evaluation metrics and the topics/languages customers ask about. |

## How it works

```
question ──► language ID (char n-gram LR) ──► topic classifier (18 topics) ──► guardrail / claim hand-off
                 english|tagalog|taglish          words+chars+glossary, weak labels
                         │
                         ▼
   hybrid cross-lingual retrieval:  BM25 over 248 section-tagged policy chunks
                                  + Tagalog→English insurance glossary expansion
                                  + QA-bank: nearest checked Q&A (any language) boosts its cited pages
                         │
                         ▼
   Claude (claude-haiku-5-5): grounded answer, [Section, p.] citations, same language as the customer
   (no API key → offline mode: the checked reference answer in the customer's language)
```

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
* To add or fix an item, edit `data/qa/seed_part*.py` and run `python scripts/build_dataset.py`.
* `data/plans/schedule_of_coverages.csv` — the **benefit limits** (PHP) from a Travel Shield Certificate of
  Insurance's *Schedule of Coverages* (amounts only, no personal data). The policy wording says most benefits are paid
  "up to the amount shown in the Policy Schedule"; this table supplies those amounts. The chatbot quotes them, the
  sidebar shows them, and the Claim Assistant shows the limit per type of loss and warns when receipts exceed it.
  For a different plan, replace the numbers in this file.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/build_dataset.py     # data/qa/*  (already built)
python -m src.ingest                # data/processed/policy_chunks.jsonl (already built)
python scripts/train_models.py      # models/*.joblib
python scripts/evaluate.py          # results/metrics.json + charts
python -m pytest tests              # smoke tests (claims, e-mail, PDF, chatbot)

copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then fill in the keys
streamlit run app.py
```

### Secrets (`.streamlit/secrets.toml`)

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude answers (optional: offline mode works without it) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | sending the claim e-mail (Gmail: `smtp.gmail.com`, 587, an **App Password**) |
| `CLAIMS_TO` | address that receives claims (comma-separated allowed) |
| `SEND_COPY_TO_CLAIMANT` | `true` to CC the customer |

Without SMTP settings the app still validates the claim and lets the customer download the filled PDF.

## Repository

```
app.py                     Streamlit UI (chat · claim assistant · insights)
src/ingest.py              PDF → clean pages → section-aware chunks
src/language.py            rule + ML language ID (english / tagalog / taglish)
src/glossary.py            Tagalog/Taglish → English insurance glossary
src/retriever.py           BM25/TF-IDF + glossary + QA-bank hybrid retriever
src/topic_classifier.py    18-topic classifier + weak labels from policy sections
src/prompts.py             versioned system prompt and grounded user-turn builder
src/chatbot.py             pipeline, guardrails, Claude call (streaming), offline fallback, cost tracking
src/claims.py              claim form schema, validation, document checklist, PDF, e-mail bodies
src/emailer.py             SMTP (STARTTLS/SSL) sender with attachments
scripts/                   build_dataset · train_models · evaluate · eval_llm
notebooks/                 01 data & dataset · 02 language & topic models · 03 retrieval & RAG
data/ models/ results/ tests/ docs/ presentation/
```

Reused from earlier bootcamp work: the chunking approach and streaming chat loop from `full_app.py`, and the
cached, cost-logged Claude calls from the sentiment project's `src/llm.py`.

## Limitations & next steps

* One Schedule of Coverages is built in. A next step is letting each customer upload their own Certificate of
  Insurance so the limits match their plan.
* The dataset was written by one author; a native-speaker and an underwriter review of the answers is recommended.
* The claims mailbox receives personal and bank data: use an official, TLS-only mailbox, and in production replace
  e-mail with the insurer's claims API and add authentication.
* The bot explains the policy; it never approves or denies claims.

*Not affiliated with Prudential Guarantee and Assurance, Inc. Built for an educational capstone using the
publicly provided policy wording.*
