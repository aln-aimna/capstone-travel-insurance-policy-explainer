"""End-to-end evaluation of the Claude answers on the 57 held-out test questions.

Needs ANTHROPIC_API_KEY (in .env or the environment). Results are cached in results/llm_cache.jsonl,
so re-running costs nothing; delete the file to call the API again.

Metrics
  citation_valid      every page the answer cites was in the retrieved excerpts (no invented citations)
  gold_page_cited     the answer cites at least one page of the reference answer
  language_match      the answer is in the same register as the question (detected by our model)
  judge_score (1-5)   Claude-as-judge: is the answer consistent with the checked reference answer?
                      5 = same facts, 3 = partly, 1 = wrong or contradicts the policy
  cost / latency      from the API usage fields

Run:  python scripts/eval_llm.py [--limit 10] [--model claude-haiku-5-5]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import config  # noqa: E402
from src.chatbot import PolicyChatbot  # noqa: E402
from src.prompts import PROMPT_VERSION  # noqa: E402

JUDGE_PROMPT = """You grade a travel-insurance chatbot answer against a reference answer written by an expert from the policy.
Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}
5 = same key facts and conditions, no contradictions; 4 = correct but misses a minor condition;
3 = partly correct or misses an important condition; 2 = mostly wrong; 1 = contradicts the reference or invents coverage.
Ignore language/register differences (English/Tagalog/Taglish) and wording.

<question>{q}</question>
<reference>{ref}</reference>
<answer>{ans}</answer>"""

PAGE_RE = re.compile(r"p{1,2}\.\s*(\d+)(?:\s*[-–]\s*(\d+))?")


def cited_pages(text):
    pages = set()
    for a, b in PAGE_RE.findall(text):
        a = int(a)
        pages.update(range(a, int(b) + 1) if b else [a])
    return pages


def key(*parts):
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def load_cache(path):
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=config.MODEL)
    args = ap.parse_args()

    bot = PolicyChatbot(model=args.model)
    if not bot.online:
        sys.exit("Set ANTHROPIC_API_KEY first (e.g. in .env).")
    qa = pd.read_csv(config.QA_FILE)
    test = qa[qa.split == "test"]
    if args.limit:
        test = test.head(args.limit)
    cache = load_cache(config.LLM_CACHE)
    rows = []
    for r in test.itertuples():
        k = key(args.model, PROMPT_VERSION, r.input)
        if k not in cache:
            res = bot.answer(r.input)
            judge = bot.client.messages.create(
                model=args.model, max_tokens=150,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=r.input, ref=r.output, ans=res.answer)}])
            jtext = "".join(b.text for b in judge.content if b.type == "text")
            m = re.search(r"\{.*\}", jtext, re.S)
            jj = json.loads(m.group(0)) if m else {"score": None, "reason": jtext}
            rec = {"key": k, "qa_id": r.qa_id, "question": r.input, "language": r.language, "answer": res.answer,
                   "mode": res.mode, "retrieved_pages": sorted({c["page"] for c in res.chunks}),
                   "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "cost_usd": res.cost_usd,
                   "latency_s": res.latency_s, "judge_score": jj.get("score"), "judge_reason": jj.get("reason")}
            cache[k] = rec
            with open(config.LLM_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(0.2)
        rec = cache[k]
        cited = cited_pages(rec["answer"])
        gold = {int(p) for p in str(r.pages).split(";") if p.strip() not in ("", "nan")}
        rows.append({**rec, "citation_valid": bool(cited) and cited <= set(rec["retrieved_pages"]) | gold,
                     "gold_page_cited": bool(cited & gold) if gold else None,
                     "language_match": bot.lang_model.predict_one(rec["answer"]) == r.language})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "llm_eval.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": len(df), "model": args.model, "prompt_version": PROMPT_VERSION,
        "judge_mean": float(df.judge_score.mean()), "judge_ge4": float((df.judge_score >= 4).mean()),
        "citation_valid": float(df.citation_valid.mean()), "gold_page_cited": float(df.gold_page_cited.dropna().mean()),
        "language_match": float(df.language_match.mean()),
        "by_language": df.groupby("language").judge_score.mean().round(2).to_dict(),
        "cost_per_1k_answers_usd": float(df.cost_usd.mean() * 1000), "mean_latency_s": float(df.latency_s.mean()),
    }
    (config.RESULTS / "llm_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
