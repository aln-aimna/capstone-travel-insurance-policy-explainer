"""Travel Shield Policy Explainer: language ID -> topic routing -> hybrid retrieval -> Claude.

Pipeline for every customer message
  1. detect_language()       english / tagalog / taglish            (src/language.py)
  2. topic model             18 topics + confidence                  (src/topic_classifier.py)
  3. guardrail               out-of-scope questions answered from the dataset, no API call
  4. HybridRetriever         top policy chunks + reference Q&A       (src/retriever.py)
  5. Claude                  grounded answer with [Section, p.] citations in the user's language
     (offline fallback)      no API key -> best matching reference answer in the user's language

The Anthropic SDK is imported lazily, so the retrieval and ML parts run and test without it.
"""
import os
import time
from dataclasses import asdict, dataclass, field

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config
from .glossary import expand_query
from .language import TAGALOG_WORDS, RuleLanguageDetector
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_content
from .retriever import HybridRetriever, bank_frame
from .topic_classifier import build_topic_model, predict_with_confidence

LANG_TO_CODE = {"english": "english", "tagalog": "tagalog", "taglish": "taglish"}

OFFLINE_INTRO = {
    "english": "Here is what the policy says (offline mode, closest reference answer):",
    "tagalog": "Narito ang sinasabi ng policy (offline mode, pinakamalapit na sagot):",
    "taglish": "Ito yung sinasabi ng policy (offline mode, pinakamalapit na sagot):",
}
CLOSEST_INTRO = {
    "english": "I'm not 100% sure which of these you mean, so here are the closest answers from the policy (offline mode):",
    "tagalog": "Hindi ako sigurado kung alin dito ang ibig mong itanong, kaya narito ang pinakamalapit na mga sagot mula sa policy (offline mode):",
    "taglish": "Di ako sure kung alin dito yung tanong mo, kaya ito yung mga pinaka-malapit na sagot sa policy (offline mode):",
}
# Offline-mode thresholds on the QA-bank similarity (0.5 x char + 0.5 x word cosine), tuned on the
# 236 held-out everyday phrasings: see scripts/evaluate.py -> faq_match / offline_coverage.
CONFIDENT_SIM, CONFIDENT_SIM_IF_TOPIC_AGREES, RELATED_SIM = 0.40, 0.25, 0.12

GENERIC = {"covered", "cover", "coverage", "claim", "claims", "insurance", "insured", "policy", "travel", "trip",
           "pay", "paid", "benefit", "abroad", "company", "sagot", "pwede", "puwede", "maaari", "happens", "mangyayari"}


def content_words(text: str) -> set:
    """Meaningful words of a question after glossary expansion (used to reject look-alike matches)."""
    words = set(re.findall(r"[a-z]{3,}", expand_query(text).lower())) - ENGLISH_STOP_WORDS - TAGALOG_WORDS - GENERIC
    return {w[:5] for w in words}          # 5-letter stems: cancel ~ cancellation, damaged ~ damage


NO_MATCH = {
    "english": "I don't have a ready-made answer for this exact question in offline mode, but these are the parts of the "
               "policy that deal with it. For a definite answer, contact {email} or {phones}.",
    "tagalog": "Wala akong nakahandang sagot sa eksaktong tanong na ito sa offline mode, pero ito ang mga bahagi ng policy "
               "na tumatalakay dito. Para sa tiyak na sagot, makipag-ugnayan sa {email} o {phones}.",
    "taglish": "Wala akong ready na sagot sa exact na tanong na 'to sa offline mode, pero ito yung parts ng policy na "
               "related dito. Para sure, contact mo ang {email} o {phones}.",
}


@dataclass
class ChatResult:
    answer: str
    language: str
    topic: str
    topic_confidence: float
    mode: str                                   # claude | offline | guardrail | error
    chunks: list = field(default_factory=list)
    references: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    suggest_claim: bool = False

    def to_dict(self):
        return asdict(self)


def _load_or_train_models(qa: pd.DataFrame):
    """Load the trained models from models/ (scripts/train_models.py); train quickly if missing."""
    import joblib
    try:
        topic_model = joblib.load(config.TOPIC_MODEL_FILE)
    except Exception:
        clf = pd.read_csv(config.CLASSIFIER_FILE)
        topic_model = build_topic_model().fit(clf.text, clf.topic)
    try:
        lang_model = joblib.load(config.LANG_MODEL_FILE)
    except Exception:
        lang_model = RuleLanguageDetector()
    return topic_model, lang_model


class PolicyChatbot:
    def __init__(self, api_key: str | None = None, model: str = config.MODEL, k_chunks: int = 5,
                 history_turns: int = 6, max_tokens: int = 700):
        self.qa = pd.read_csv(config.QA_FILE)
        self.retriever = HybridRetriever(qa=bank_frame())
        self.topic_model, self.lang_model = _load_or_train_models(self.qa)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.k_chunks, self.history_turns, self.max_tokens = k_chunks, history_turns, max_tokens
        self._client = None
        self.total_cost = 0.0
        self.last_result: ChatResult | None = None

    # ------------------------------------------------------------------ analysis
    @property
    def online(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze(self, text: str) -> dict:
        language = str(self.lang_model.predict_one(text))
        topic, conf = predict_with_confidence(self.topic_model, text)
        return {"language": language, "topic": topic, "topic_confidence": conf}

    def _guardrail(self, text, info, refs) -> str | None:
        """Answer clear out-of-scope questions straight from the dataset (no API cost)."""
        top = refs[0] if refs else None
        margin = top["sim"] - refs[1]["sim"] if top and len(refs) > 1 else 0.0
        if top and top["topic"] == "out_of_scope" and (top["sim"] >= 0.45 or
                                                       (info["topic"] == "out_of_scope" and margin >= 0.12)):
            row = self.retriever.bank.answer_in(top["item_id"], info["language"])
            return row["output"] if row else top["answer"]
        if info["topic"] == "out_of_scope" and info["topic_confidence"] >= 0.55 and (not top or top["sim"] < 0.3):
            row = self.retriever.bank.answer_in("OOS-03", info["language"])
            return row["output"] if row else None
        return None

    def _references_in_language(self, refs, language, max_refs=3):
        out = []
        for r in refs[:max_refs]:
            row = self.retriever.bank.answer_in(r["item_id"], language)
            if row:
                out.append({**r, "question": row["input"], "answer": row["output"]})
            else:
                out.append(r)
        return out

    # ------------------------------------------------------------------ answering
    def prepare(self, text: str, history: list | None = None) -> dict:
        info = self.analyze(text)
        ret = self.retriever.retrieve(text, k=self.k_chunks)
        refs = self._references_in_language(ret.references, info["language"])
        system = SYSTEM_PROMPT.format(language=info["language"], topic=config.TOPIC_LABELS.get(info["topic"], info["topic"]),
                                      insurer_email=config.INSURER_EMAIL, insurer_phones=config.INSURER_PHONES)
        msgs = []
        for m in (history or [])[-self.history_turns:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
        while msgs and msgs[0]["role"] != "user":           # API needs to start with a user turn
            msgs.pop(0)
        msgs.append({"role": "user", "content": build_user_content(text, ret.chunks, refs)})
        return {"question": text, "info": info, "chunks": ret.chunks, "references": refs, "system": system, "messages": msgs,
                "guardrail": self._guardrail(text, info, ret.references)}

    def _result(self, prep, answer, mode, **kw) -> ChatResult:
        info = prep["info"]
        return ChatResult(answer=answer, language=info["language"], topic=info["topic"],
                          topic_confidence=info["topic_confidence"], mode=mode, chunks=prep["chunks"],
                          references=prep["references"], suggest_claim=info["topic"] == "claims", **kw)

    def offline_answer(self, prep) -> str:
        """Three tiers, so the customer is not left at a dead end:
        1. confident match      -> the checked reference answer
        2. related matches      -> the 2-3 closest reference answers, each under its question
        3. nothing related      -> say so, show the most relevant policy clause and the insurer's contacts"""
        lang = prep["info"]["language"]
        info = prep["info"]
        refs = prep["references"]
        top = refs[0] if refs else None
        if top and (top["sim"] >= CONFIDENT_SIM or
                    (top["sim"] >= CONFIDENT_SIM_IF_TOPIC_AGREES and top["topic"] == info["topic"])):
            return f"{OFFLINE_INTRO[lang]}\n\n{top['answer']}"
        q_words = content_words(prep["question"])
        related = [r for r in refs if r["sim"] >= RELATED_SIM and r["topic"] != "out_of_scope"
                   and q_words & content_words(r["question"])][:3]
        if related:
            parts = [CLOSEST_INTRO[lang]]
            parts += [f"**{r['question']}**\n\n{r['answer']}" for r in related]
            return "\n\n".join(parts)
        msg = NO_MATCH[lang].format(email=config.INSURER_EMAIL, phones=config.INSURER_PHONES)
        for c in prep["chunks"][:2]:
            snippet = " ".join(c["text"][:420].split()).rsplit(" ", 1)[0] + "..."
            msg += f"\n\n**{c['section']}, p. {c['page']}**\n\n> {snippet}"
        return msg

    def _cost(self, usage) -> tuple[int, int, float]:
        price = config.PRICE_PER_MTOK.get(self.model, {"input": 0.0, "output": 0.0})
        tin, tout = usage.input_tokens, usage.output_tokens
        return tin, tout, tin / 1e6 * price["input"] + tout / 1e6 * price["output"]

    def answer(self, text: str, history: list | None = None) -> ChatResult:
        """Non-streaming answer (used by scripts and tests)."""
        prep = self.prepare(text, history)
        if prep["guardrail"]:
            return self._result(prep, prep["guardrail"], "guardrail")
        if not self.online:
            return self._result(prep, self.offline_answer(prep), "offline")
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                               system=prep["system"], messages=prep["messages"])
        except Exception as e:                                  # network / key / rate limit
            return self._result(prep, self.offline_answer(prep) + f"\n\n_(Claude unavailable: {type(e).__name__})_", "error")
        text_out = "".join(b.text for b in resp.content if b.type == "text").strip()
        tin, tout, cost = self._cost(resp.usage)
        self.total_cost += cost
        return self._result(prep, text_out, "claude", input_tokens=tin, output_tokens=tout, cost_usd=cost,
                            latency_s=time.perf_counter() - t0)

    def answer_stream(self, text: str, history: list | None = None):
        """Generator of text pieces for st.write_stream(). After it is exhausted, self.last_result
        holds the ChatResult (mode, sources, tokens, cost)."""
        prep = self.prepare(text, history)
        if prep["guardrail"] or not self.online:
            mode = "guardrail" if prep["guardrail"] else "offline"
            out = prep["guardrail"] or self.offline_answer(prep)
            self.last_result = self._result(prep, out, mode)
            yield out
            return
        t0 = time.perf_counter()
        pieces = []
        try:
            with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens,
                                             system=prep["system"], messages=prep["messages"]) as stream:
                for piece in stream.text_stream:
                    pieces.append(piece)
                    yield piece
                final = stream.get_final_message()
            tin, tout, cost = self._cost(final.usage)
            self.total_cost += cost
            self.last_result = self._result(prep, "".join(pieces), "claude", input_tokens=tin, output_tokens=tout,
                                            cost_usd=cost, latency_s=time.perf_counter() - t0)
        except Exception as e:
            fallback = self.offline_answer(prep) + f"\n\n_(Claude unavailable: {type(e).__name__})_"
            self.last_result = self._result(prep, fallback, "error")
            yield ("\n\n" if pieces else "") + fallback


__all__ = ["PolicyChatbot", "ChatResult", "PROMPT_VERSION"]
