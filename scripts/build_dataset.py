"""Build the input/output datasets from the hand-written seed items.

Outputs (all in data/qa/):
  policy_qa_items.json      one record per item (all three languages side by side)
  policy_qa_dataset.csv     long format: one row per (item, language) -> input, output
  classifier_dataset.csv    questions + extra phrasings with topic labels, for the topic model
  train.jsonl / test.jsonl  chat-format records (system/user/assistant) for few-shot or fine-tuning
  DATASET_CARD.md           what is in the data and how it was made

The split is by ITEM (group split), so the three translations of a test question never appear
in training. That keeps the evaluation honest for the QA-bank retriever and the classifier.

Run:  python scripts/build_dataset.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "qa"))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402

import seed_part1, seed_part2, seed_part3  # noqa: E402

LANGS = {"en": "english", "tl": "tagalog", "tg": "taglish"}
REQUIRED = ["id", "topic", "section", "pages", "q_en", "q_tl", "q_tg", "a_en", "a_tl", "a_tg"]
TEST_FRACTION = 0.2


def load_items():
    items = seed_part1.ITEMS + seed_part2.ITEMS + seed_part3.ITEMS
    ids = Counter(it["id"] for it in items)
    dupes = [k for k, v in ids.items() if v > 1]
    assert not dupes, f"duplicate ids: {dupes}"
    for it in items:
        missing = [k for k in REQUIRED if k not in it or it[k] in ("", None)]
        assert not missing, f"{it.get('id')}: missing {missing}"
        assert it["topic"] in config.TOPICS, f"{it['id']}: unknown topic {it['topic']}"
        assert all(1 <= p <= 47 for p in it["pages"]), f"{it['id']}: page out of range"
        it.setdefault("alt", [])
    return items


def split_items(items, seed=config.SEED):
    """Group split stratified by topic: ~20% of the items of each topic go to test
    (at least one when the topic has 3+ items; topics with fewer stay in train)."""
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for it in items:
        by_topic[it["topic"]].append(it["id"])
    test_ids = set()
    for topic, ids in sorted(by_topic.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = round(len(ids) * TEST_FRACTION)
        if len(ids) >= 3:
            n_test = max(1, n_test)
        test_ids.update(ids[:n_test])
    return test_ids


def main():
    items = load_items()
    test_ids = split_items(items)
    for it in items:
        it["split"] = "test" if it["id"] in test_ids else "train"

    out = config.DATA_QA
    (out / "policy_qa_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # long input/output format
    rows = []
    for it in items:
        for code, lang in LANGS.items():
            rows.append(dict(
                qa_id=f"{it['id']}-{code.upper()}", item_id=it["id"], topic=it["topic"],
                section=it["section"], pages=";".join(map(str, it["pages"])), language=lang,
                input=it[f"q_{code}"], output=it[f"a_{code}"], split=it["split"],
            ))
    qa = pd.DataFrame(rows)
    qa.to_csv(out / "policy_qa_dataset.csv", index=False, encoding="utf-8-sig")

    # classifier data: the 3 questions + extra phrasings
    crow = []
    for it in items:
        for code, lang in LANGS.items():
            crow.append(dict(text=it[f"q_{code}"], topic=it["topic"], language=lang,
                             item_id=it["id"], source="question", split=it["split"]))
        for a in it["alt"]:
            crow.append(dict(text=a, topic=it["topic"], language="", item_id=it["id"],
                             source="paraphrase", split=it["split"]))
    clf = pd.DataFrame(crow)
    clf.to_csv(out / "classifier_dataset.csv", index=False, encoding="utf-8-sig")

    # chat-format JSONL (usable as few-shot examples or for supervised fine-tuning)
    from src.prompts import SYSTEM_PROMPT_SHORT
    for split in ("train", "test"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in qa[qa.split == split].itertuples():
                rec = {"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_SHORT},
                    {"role": "user", "content": r.input},
                    {"role": "assistant", "content": r.output},
                ], "metadata": {"qa_id": r.qa_id, "topic": r.topic, "language": r.language,
                                "pages": r.pages, "section": r.section}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_card(items, qa, clf)
    print(f"items: {len(items)}  (train {sum(i['split']=='train' for i in items)}, test {len(test_ids)})")
    print(f"QA pairs: {len(qa)}   classifier rows: {len(clf)}")
    print(qa.groupby(["split", "language"]).size().unstack())


def write_card(items, qa, clf):
    topic_counts = Counter(it["topic"] for it in items)
    lines = [
        "# Travel Shield Q&A Dataset - Dataset Card", "",
        "**Source:** `data/raw/Travel-Shield-ePolicy.pdf` (Prudential Guarantee Travel Shield Insurance "
        "Policy, 47 pages incl. the COVID-19 Endorsement) and `data/raw/Claim-Form.pdf` (Notification of "
        "Claim checklist).", "",
        "**How it was made:** every item was written by reading the policy clause and citing it "
        "(section + PDF page). Each item has the same question and answer in English, Tagalog and "
        "Taglish. Amounts that the policy leaves to the customer's Policy Schedule are not invented; "
        "the answer says to check the Policy Schedule.", "",
        "| File | Rows | Purpose |", "|---|---|---|",
        f"| `policy_qa_items.json` | {len(items)} items | master file, 3 languages side by side |",
        f"| `policy_qa_dataset.csv` | {len(qa)} | input (question) -> output (answer), one row per language |",
        f"| `classifier_dataset.csv` | {len(clf)} | question/paraphrase -> topic label |",
        f"| `train.jsonl` / `test.jsonl` | {sum(qa.split=='train')} / {sum(qa.split=='test')} | chat format for few-shot / fine-tuning |",
        "", "**Columns of `policy_qa_dataset.csv`:** `qa_id, item_id, topic, section, pages, language, "
        "input, output, split`.", "",
        "**Split:** grouped by item, ~20% of the items of each topic in test, seed 42. "
        "No translation of a test question is in train.", "",
        "## Items per topic", "", "| Topic | Items |", "|---|---|",
    ]
    lines += [f"| {t} | {topic_counts[t]} |" for t in config.TOPICS]
    lines += ["", "## Example", "", "| Language | Input | Output |", "|---|---|---|"]
    ex = qa[qa.item_id == "BAG-04"]
    for r in ex.itertuples():
        lines.append(f"| {r.language} | {r.input} | {r.output} |")
    lines += ["", "## Limitations", "",
              "- Written by one author; a native-speaker review of the Tagalog lines is recommended.",
              "- Policy wording only: plan-specific amounts live in each customer's Policy Schedule.",
              "- Not legal advice; the insurer's claims team makes all coverage decisions."]
    (config.DATA_QA / "DATASET_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
