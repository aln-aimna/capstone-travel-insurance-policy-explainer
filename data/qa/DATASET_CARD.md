# Travel Shield Q&A Dataset - Dataset Card

**Source:** `data/raw/Travel-Shield-ePolicy.pdf` (Prudential Guarantee Travel Shield Insurance Policy, 47 pages incl. the COVID-19 Endorsement) and `data/raw/Claim-Form.pdf` (Notification of Claim checklist).

**How it was made:** every item was written by reading the policy clause and citing it (section + PDF page). Each item has the same question and answer in English, Tagalog and Taglish. Amounts that the policy leaves to the customer's Policy Schedule are not invented; the answer says to check the Policy Schedule.

| File | Rows | Purpose |
|---|---|---|
| `policy_qa_items.json` | 95 items | master file, 3 languages side by side |
| `policy_qa_dataset.csv` | 285 | input (question) -> output (answer), one row per language |
| `classifier_dataset.csv` | 521 | question/paraphrase -> topic label |
| `train.jsonl` / `test.jsonl` | 228 / 57 | chat format for few-shot / fine-tuning |

**Columns of `policy_qa_dataset.csv`:** `qa_id, item_id, topic, section, pages, language, input, output, split`.

**Split:** grouped by item, ~20% of the items of each topic in test, seed 42. No translation of a test question is in train.

## Items per topic

| Topic | Items |
|---|---|
| medical | 6 |
| covid | 7 |
| dental | 2 |
| evacuation | 3 |
| hospital_family | 4 |
| personal_accident | 6 |
| trip_cancellation | 6 |
| trip_curtailment | 3 |
| flight_delay | 5 |
| baggage | 7 |
| money_documents | 3 |
| liability_car_rental | 4 |
| exclusions | 8 |
| eligibility | 7 |
| claims | 11 |
| policy_admin | 6 |
| optional_benefits | 4 |
| out_of_scope | 3 |

## Example

| Language | Input | Output |
|---|---|---|
| english | My luggage didn't arrive with my flight. What can I claim? | Baggage Delay pays for essential clothing and toiletries if your checked-in baggage is delayed more than 6 hours after arrival at the overseas destination, with additional amounts after 24 and 48 hours, up to the Policy Schedule limit. Get written confirmation of the delay from the airline and keep all receipts (Section 17, pp. 28-29). |
| tagalog | Hindi dumating ang bagahe ko kasabay ng flight. Ano ang maaari kong i-claim? | Ang Baggage Delay ay nagbabayad ng mahahalagang damit at toiletries kung ang checked-in na bagahe ay naantala nang higit 6 oras mula pagdating sa destinasyon sa ibang bansa, at may dagdag pagkalipas ng 24 at 48 oras, hanggang sa limitang nasa Policy Schedule. Kumuha ng nakasulat na kumpirmasyon mula sa airline at itago ang lahat ng resibo (Section 17, pp. 28-29). |
| taglish | Hindi dumating yung luggage ko pagdating ko. Ano pwede i-claim? | Ang Baggage Delay ay nagbabayad ng essential clothes at toiletries kung delayed ang checked-in baggage mo ng more than 6 hours pagdating sa overseas destination, may dagdag pa after 24 at 48 hours, up to Policy Schedule limit. Kumuha ng written confirmation ng delay sa airline at itago lahat ng receipts (Section 17, pp. 28-29). |

## Limitations

- Written by one author; a native-speaker review of the Tagalog lines is recommended.
- Policy wording only: plan-specific amounts live in each customer's Policy Schedule.
- Not legal advice; the insurer's claims team makes all coverage decisions.
