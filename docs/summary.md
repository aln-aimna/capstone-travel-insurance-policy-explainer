# Written summary: Travel Shield Policy Explainer & Claim Assistant

## Problem
Travel insurance is bought in seconds and understood by few. Prudential Guarantee's Travel Shield policy
is 47 pages of legal English (20 benefit sections, ~120 exclusions, a COVID-19 endorsement), and its
claim form is a 4-page PDF with a different document checklist per type of loss. Customers ask their
questions in English, Tagalog or Taglish, usually during a stressful event abroad. The consequences are
missed deadlines (30-day written notice, 90-day proof of loss), incomplete claims, avoidable denials and
repetitive customer-care load. For the insurer, every unclear clause costs call-centre minutes and goodwill.

## Approach
1. **Data.** Parsed the policy PDF into 248 chunks, each tagged with its section heading and page. Wrote
   an input/output dataset from the policy: 95 items, each with the same question and answer in English,
   Tagalog and Taglish (285 Q&A pairs), each citing section and page, plus 236 everyday phrasings.
   Answers never invent plan amounts that the policy leaves to the Policy Schedule. Items were split by
   group (no translation of a test question appears in training).
2. **Language ID.** A rule detector (English vocabulary from the policy vs. Tagalog function words and
   affixes) and an ML detector (character 1-4-grams + rule features, logistic regression).
3. **Topic routing.** 18 topics. TF-IDF on glossary-expanded words + character n-grams, logistic
   regression, plus **weak supervision**: policy chunks labelled by their section heading.
4. **Cross-lingual retrieval.** BM25 over the chunks, a Tagalog→English insurance glossary for query
   expansion, and a QA bank whose nearest checked questions (any language) boost the pages they cite and are
   passed to the LLM as reference answers.
5. **Generation.** Claude (Haiku) with a versioned, grounded prompt: answer only from the excerpts, cite
   [Section, p.], reply in the customer's register, never invent amounts, never approve claims, emergency
   first. Offline fallback returns the checked reference answer in the customer's language.
6. **Claim Assistant.** The Notification of Claim as a guided form: loss-type suggestion from free text,
   per-loss document checklist, deadline warnings computed from the date of loss, validation, a filled PDF,
   and SMTP e-mail with attachments (bank number masked in the e-mail body).

## Results (57 held-out questions from 19 unseen items)
* Language ID: **96.5%** (rules 82.5%); English vs Filipino separation 100%.
* Topic: **45.6%** over 18 classes on never-seen items (majority 10.5%, Naive Bayes 29.8%, LogReg without
  weak labels 38.6%). The topic is advisory; answers do not depend on it.
* Retrieval Hit@3: **76%** (TF-IDF baseline 63%). On Tagalog questions the glossary + QA bank raise Hit@3
  from 56% to **89%**. On 232 everyday FAQ paraphrases: 49% → **71%**.
* Cost: about 2.2k input tokens per answer, **≈ $0.35 per 1,000 answers** on Haiku.
* End-to-end Claude answer quality is measured by `scripts/eval_llm.py` (Claude-as-judge vs the reference
  answers, citation validity, language match) once an API key is configured.

## Value
For customers: correct answers in their own language at 2 a.m. abroad, and a claim that is complete and on
time the first time. For the insurer: deflected FAQ calls, fewer incomplete claims to chase, structured claim
data on arrival, and analytics on which clauses confuse customers. The same pipeline extends to any policy
PDF (motor, health, home) by re-running ingestion and writing a new Q&A set.

## Limitations
Small test sets (±10-point uncertainty); one-author dataset needing native-speaker and underwriter review;
policy wording only (no per-customer schedule yet); e-mail delivery is a prototype for a claims API.
