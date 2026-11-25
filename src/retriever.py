"""Cross-lingual hybrid retriever.

A question can be in English, Tagalog or Taglish, but the policy is in English. Three signals:

  A. Policy index   - lexical search over the 248 policy chunks: word 1-2-gram TF-IDF fused with
                      Okapi BM25 (mean of the two, each scaled to max 1), with each chunk's section
                      heading prepended (so "baggage delay" matches Section 17 text that only says
                      "delayed, misdirected or temporarily misplaced").
  B. Glossary       - Tagalog/Taglish words are expanded to English before querying A
                      (src/glossary.py).
  C. QA bank        - char 2-5-gram TF-IDF over the *training* questions of the Q&A dataset in all
                      three languages. The nearest reference questions are language-agnostic
                      bridges: their cited pages boost the policy chunks on those pages, and their
                      checked answers are passed to the LLM as reference answers.

    score(chunk) = lexical_A(chunk)  +  w_bank * max over top bank hits (sim * [chunk.page in hit.pages])

scripts/evaluate.py runs the ablation (A, A+B, A+C, A+B+C) and reports Hit@k of the gold page.
"""
from dataclasses import dataclass, field

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config
from .glossary import expand_query
from .ingest import load_chunks


@dataclass
class Retrieved:
    chunks: list = field(default_factory=list)       # [{chunk_id, page, section, text, score}]
    references: list = field(default_factory=list)   # [{item_id, question, answer, pages, section, sim, language}]
    query_used: str = ""


class PolicyIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        docs = [f"{c['section']}. {c['text']}" for c in chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.M = self.vec.fit_transform(docs)
        self.pages = np.array([c["page"] for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return linear_kernel(self.vec.transform([query]), self.M).ravel()


class BM25Index:
    """Okapi BM25 over the same chunk texts (light plural stemming)."""

    def __init__(self, chunks: list[dict], k1: float = 1.4, b: float = 0.6):
        self.k1, self.b = k1, b
        self.docs = [self.tokens(f"{c['section']} {c['text']}") for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.len = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    @staticmethod
    def tokens(text: str) -> list[str]:
        out = []
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in ENGLISH_STOP_WORDS or len(w) < 2:
                continue
            out.append(w[:-1] if w.endswith("s") and len(w) > 4 else w)
        return out

    def scores(self, query: str) -> np.ndarray:
        q = set(self.tokens(query))
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for w in q:
                f = tf.get(w)
                if f:
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return out


class QABank:
    """Nearest-neighbour search over reference questions (any language)."""

    def __init__(self, qa: pd.DataFrame, alpha: float = 0.5):
        """Similarity = alpha * char 2-5-gram cosine + (1 - alpha) * word cosine on glossary-expanded
        text. Chars handle Taglish spelling; expanded words let "bagahe" meet "baggage"."""
        self.qa = qa.reset_index(drop=True)
        self.alpha = alpha
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)
        self.M = self.vec.fit_transform(self.qa["input"].str.lower())
        expanded = [expand_query(t).lower() for t in self.qa["input"]]
        self.wvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.W = self.wvec.fit_transform(expanded)

    def search(self, query: str, k: int = 5, exclude: set | None = None) -> list[dict]:
        sims = self.alpha * linear_kernel(self.vec.transform([query.lower()]), self.M).ravel() + \
            (1 - self.alpha) * linear_kernel(self.wvec.transform([expand_query(query).lower()]), self.W).ravel()
        order = sims.argsort()[::-1]
        out, seen = [], set()
        for i in order:
            r = self.qa.iloc[i]
            if r.item_id in seen or (exclude and r.item_id in exclude):
                continue                   # one hit per item (its 3 translations are near-duplicates)
            seen.add(r.item_id)
            pages = [int(p) for p in str(r.pages).split(";") if str(p).strip() not in ("", "nan")]
            out.append(dict(item_id=r.item_id, question=r.input, answer=r.output, pages=pages,
                            section=r.section, topic=r.topic, language=r.language, sim=float(sims[i])))
            if len(out) >= k:
                break
        return out

    def answer_in(self, item_id: str, language: str) -> dict | None:
        rows = self.qa[(self.qa.item_id == item_id) & (self.qa.language == language)]
        return None if rows.empty else rows.iloc[0].to_dict()


def bank_frame(splits=("train", "test"), include_paraphrases=True) -> pd.DataFrame:
    """Reference questions for the QA bank: the 3-language questions, plus (optionally) the extra
    paraphrases from classifier_dataset.csv, which make everyday FAQ phrasings match better
    ("delayed flight 10 hours" -> the flight-delay item). Paraphrase rows inherit the item's
    English answer, pages and section; answer_in() still returns the answer in the user's language."""
    qa = pd.read_csv(config.QA_FILE)
    qa = qa[qa.split.isin(splits)]
    if not include_paraphrases:
        return qa.reset_index(drop=True)
    clf = pd.read_csv(config.CLASSIFIER_FILE)
    alt = clf[(clf.source == "paraphrase") & clf.split.isin(splits)]
    en = qa[qa.language == "english"].set_index("item_id")
    alt = alt[alt.item_id.isin(en.index)]
    extra = pd.DataFrame({
        "qa_id": alt.item_id + "-ALT", "item_id": alt.item_id, "topic": alt.topic,
        "section": en.loc[alt.item_id, "section"].values, "pages": en.loc[alt.item_id, "pages"].values,
        "language": "paraphrase", "input": alt.text, "output": en.loc[alt.item_id, "output"].values,
        "split": alt.split})
    return pd.concat([qa, extra], ignore_index=True)


class HybridRetriever:
    def __init__(self, chunks=None, qa: pd.DataFrame | None = None, use_glossary=True, use_bank=True,
                 w_bank: float = 0.6, bank_k: int = 3, lexical: str = "bm25"):
        """lexical: "tfidf", "bm25" or "hybrid" (mean of both, each scaled to max 1)."""
        self.chunks = chunks if chunks is not None else load_chunks()
        self.index = PolicyIndex(self.chunks)
        self.bm25 = BM25Index(self.chunks) if lexical in ("bm25", "hybrid") else None
        self.lexical = lexical
        if qa is None and config.QA_FILE.exists():
            qa = bank_frame()
        self.bank = QABank(qa) if (use_bank and qa is not None and len(qa)) else None
        self.use_glossary = use_glossary
        self.w_bank, self.bank_k = w_bank, bank_k

    def retrieve(self, query: str, k: int = 4, exclude_items: set | None = None) -> Retrieved:
        """exclude_items: leave-one-item-out evaluation (the bank must not contain the question's
        own item, otherwise retrieval is trivially solved by its translations)."""
        q = expand_query(query) if self.use_glossary else query
        s = self._lexical_scores(q)
        refs = []
        if self.bank is not None:
            refs = self.bank.search(query, k=self.bank_k, exclude=exclude_items)
            boost = np.zeros_like(s)
            for r in refs:
                if r["pages"]:
                    mask = np.isin(self.index.pages, r["pages"])
                    boost = np.maximum(boost, mask * r["sim"])
            s = s + self.w_bank * boost
        order = s.argsort()[::-1][:k]
        chunks = [dict(self.chunks[i], score=float(s[i])) for i in order if s[i] > 0]
        return Retrieved(chunks=chunks, references=refs, query_used=q)

    def _lexical_scores(self, q: str) -> np.ndarray:
        def norm(x):
            return x / x.max() if x.max() > 0 else x
        if self.lexical == "tfidf":
            return norm(self.index.scores(q))
        if self.lexical == "bm25":
            return norm(self.bm25.scores(q))
        return 0.5 * norm(self.index.scores(q)) + 0.5 * norm(self.bm25.scores(q))

    def top_pages(self, query: str, k: int = 5, exclude_items: set | None = None) -> list[int]:
        """Distinct pages in rank order (used for Hit@k evaluation)."""
        res = self.retrieve(query, k=40, exclude_items=exclude_items)
        pages = []
        for c in res.chunks:
            if c["page"] not in pages:
                pages.append(c["page"])
            if len(pages) >= k:
                break
        return pages
