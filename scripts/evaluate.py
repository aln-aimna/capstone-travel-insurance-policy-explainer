"""Evaluate every ML component on held-out data and write results/metrics.json + charts.

  1. Language ID     rule-based vs char-n-gram ML, on the 57 test questions
  2. Topic routing   majority / Naive Bayes / logistic regression, on the 57 test questions
  3. Retrieval       Hit@k of the gold policy page, ablation of lexical / glossary / QA bank
                     a) "unseen":     test questions, bank = training items only (group split)
                     b) "paraphrase": everyday paraphrases of known FAQs, bank = all dataset questions
  4. FAQ matching    is the top QA-bank hit the right item? (what offline mode answers with)

Run:  python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from src import config  # noqa: E402
from src.language import MLLanguageDetector, RuleLanguageDetector  # noqa: E402
from src.retriever import HybridRetriever, bank_frame  # noqa: E402
from src.ingest import load_chunks  # noqa: E402
from src.topic_classifier import build_topic_model, weak_labels_from_policy  # noqa: E402

LANGS = ["english", "tagalog", "taglish"]
# reference palette (dataviz skill), light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"english": "#2a78d6", "tagalog": "#eb6834", "taglish": "#1baf7a"}


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def gold_pages(p):
    return {int(x) for x in str(p).split(";") if x.strip() not in ("", "nan")}


# ---------------------------------------------------------------------------- 1. language
def eval_language(qa):
    tr, te = qa[qa.split == "train"], qa[qa.split == "test"]
    rule = RuleLanguageDetector()
    ml = MLLanguageDetector().fit(tr.input, tr.language)
    out, preds = {}, {}
    for name, model in [("rule", rule), ("ml", ml)]:
        p = model.predict(te.input)
        preds[name] = p
        binary = accuracy_score(te.language == "english", np.array(p) == "english")
        out[name] = {"accuracy": accuracy_score(te.language, p), "macro_f1": f1_score(te.language, p, average="macro"),
                     "english_vs_filipino_accuracy": binary}
    cm = confusion_matrix(te.language, preds["ml"], labels=LANGS)
    out["ml"]["confusion"] = cm.tolist()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LANGS)
    ax.set_yticks(range(3), LANGS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xlabel("predicted", color=INK2)
    ax.set_ylabel("actual", color=INK2)
    ax.set_title(f"Language ID (ML) - {out['ml']['accuracy']:.0%} on test", loc="left", fontweight="bold", color=INK)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(config.CHARTS / "language_confusion.png", dpi=160)
    plt.close(fig)
    pd.DataFrame({"text": te.input, "gold": te.language, "rule": preds["rule"], "ml": preds["ml"]}).to_csv(
        config.RESULTS / "predictions_language.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------------------- 2. topic
def eval_topic(clf):
    tr = clf[clf.split == "train"]
    te = clf[(clf.split == "test") & (clf.source == "question")]
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "naive_bayes": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.3)),
        "logreg": build_topic_model(),
    }
    weak = weak_labels_from_policy(load_chunks())
    models["logreg_weak"] = build_topic_model()
    out, preds = {}, {}
    for name, m in models.items():
        X = pd.concat([tr[["text", "topic"]], weak]) if name == "logreg_weak" else tr
        m.fit(X.text, X.topic)
        p = m.predict(te.text)
        preds[name] = p
        labels = sorted(te.topic.unique())
        out[name] = {"accuracy": accuracy_score(te.topic, p),
                     "macro_f1": f1_score(te.topic, p, labels=labels, average="macro", zero_division=0),
                     "by_language": {lang: accuracy_score(te.topic[te.language == lang], np.array(p)[te.language.values == lang])
                                     for lang in LANGS}}
    out["n_test"], out["n_train"], out["n_classes_test"] = len(te), len(tr), int(te.topic.nunique())
    pd.DataFrame({"text": te.text, "language": te.language, "gold": te.topic, **preds}).to_csv(
        config.RESULTS / "predictions_topic.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    names = ["majority", "naive_bayes", "logreg", "logreg_weak"]
    nice = ["Majority\nclass", "Naive Bayes\n(words)", "LogReg\n(words+chars+glossary)", "LogReg + weak labels\nfrom policy sections"]
    x = np.arange(len(names))
    w = 0.24
    for k, lang in enumerate(LANGS):
        vals = [out[n]["by_language"][lang] for n in names]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=SERIES[lang], label=lang)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, nice)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style(ax, "Topic classifier accuracy on unseen test questions", "accuracy")
    ax.legend(frameon=False, ncol=3, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(config.CHARTS / "topic_by_language.png", dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- 3-4. retrieval
CONFIGS = {
    "tfidf": dict(lexical="tfidf", use_glossary=False, use_bank=False),
    "bm25": dict(lexical="bm25", use_glossary=False, use_bank=False),
    "bm25+glossary": dict(lexical="bm25", use_glossary=True, use_bank=False),
    "bm25+bank": dict(lexical="bm25", use_glossary=False, use_bank=True),
    "full": dict(lexical="bm25", use_glossary=True, use_bank=True),
}


def hit_at(ret, queries, ks=(1, 3, 5)):
    rows = []
    for q in queries.itertuples():
        gold = gold_pages(q.pages)
        if not gold:
            continue
        pages = ret.top_pages(q.input, max(ks))
        rows.append({"language": q.language, **{f"hit@{k}": bool(gold & set(pages[:k])) for k in ks}})
    df = pd.DataFrame(rows)
    res = {f"hit@{k}": float(df[f"hit@{k}"].mean()) for k in ks}
    res["by_language"] = {lang: {f"hit@{k}": float(g[f"hit@{k}"].mean()) for k in ks}
                          for lang, g in df.groupby("language")}
    res["n"] = len(df)
    return res


def eval_retrieval(qa, clf):
    # a) unseen questions
    test_q = qa[qa.split == "test"]
    bank_train = bank_frame(splits=("train",))
    unseen = {name: hit_at(HybridRetriever(qa=bank_train, **kw), test_q) for name, kw in CONFIGS.items()}

    # b) paraphrases of known FAQs (bank = all 3-language questions, paraphrases held out)
    bank_all = bank_frame(include_paraphrases=False)
    items = json.loads(config.QA_ITEMS_FILE.read_text(encoding="utf-8"))
    pages = {it["id"]: ";".join(map(str, it["pages"])) for it in items}
    para = clf[clf.source == "paraphrase"].copy()
    para = para.assign(input=para.text, pages=para.item_id.map(pages), language="paraphrase")
    paraphrase = {name: hit_at(HybridRetriever(qa=bank_all, **kw), para) for name, kw in CONFIGS.items()}

    # FAQ matching accuracy (offline-mode answer = top bank item)
    ret = HybridRetriever(qa=bank_all)
    top1 = top3 = 0
    for r in para.itertuples():
        hits = [h["item_id"] for h in ret.bank.search(r.input, k=3)]
        top1 += hits[:1] == [r.item_id]
        top3 += r.item_id in hits
    faq = {"top1": top1 / len(para), "top3": top3 / len(para), "n": len(para)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    order = list(CONFIGS)
    for ax, (title, res) in zip(axes, [("Unseen questions (by language)", unseen), ("FAQ paraphrases", paraphrase)]):
        x = np.arange(len(order))
        if "Unseen" in title:
            w = 0.25
            for k, lang in enumerate(LANGS):
                vals = [res[c]["by_language"][lang]["hit@3"] for c in order]
                ax.bar(x + (k - 1) * w, vals, w - 0.02, color=SERIES[lang], label=lang)
            ax.legend(frameon=False, ncol=3, loc="upper left", labelcolor=INK2, fontsize=8)
        else:
            vals = [res[c]["hit@3"] for c in order]
            bars = ax.bar(x, vals, 0.55, color=SERIES["english"])
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, color=INK2, padding=2)
        ax.set_xticks(x, [c.replace("+", "\n+ ") for c in order], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        style(ax, title, "Hit@3 (gold page in top 3)" if "Unseen" in title else None)
    fig.tight_layout()
    fig.savefig(config.CHARTS / "retrieval_ablation.png", dpi=160)
    plt.close(fig)
    return {"unseen": unseen, "paraphrase": paraphrase, "faq_match": faq}


def main():
    qa = pd.read_csv(config.QA_FILE)
    clf = pd.read_csv(config.CLASSIFIER_FILE)
    metrics = {"dataset": {"items": int(qa.item_id.nunique()), "qa_pairs": len(qa), "classifier_rows": len(clf),
                           "test_items": int(qa[qa.split == "test"].item_id.nunique())},
               "language": eval_language(qa), "topic": eval_topic(clf), "retrieval": eval_retrieval(qa, clf)}
    (config.RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    L, T, R = metrics["language"], metrics["topic"], metrics["retrieval"]
    print(f"Language ID  rule {L['rule']['accuracy']:.3f}  ML {L['ml']['accuracy']:.3f}  "
          f"(English vs Filipino: {L['ml']['english_vs_filipino_accuracy']:.3f})")
    print("Topic        " + "  ".join(f"{n} acc {T[n]['accuracy']:.3f} F1 {T[n]['macro_f1']:.3f}"
                                     for n in ("majority", "naive_bayes", "logreg", "logreg_weak")))
    for scen in ("unseen", "paraphrase"):
        print(f"Retrieval {scen:10s}" + "  ".join(f"{c}: {R[scen][c]['hit@1']:.2f}/{R[scen][c]['hit@3']:.2f}"
                                                   for c in CONFIGS) + "   (hit@1/hit@3)")
    for lang in LANGS:
        print(f"   unseen {lang:8s} bm25 {R['unseen']['bm25']['by_language'][lang]['hit@3']:.2f} -> "
              f"full {R['unseen']['full']['by_language'][lang]['hit@3']:.2f}")
    print(f"FAQ match    top1 {R['faq_match']['top1']:.3f}  top3 {R['faq_match']['top3']:.3f}  (n={R['faq_match']['n']})")


if __name__ == "__main__":
    main()
