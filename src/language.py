"""Language identification for English / Tagalog / Taglish questions.

Two detectors, compared in scripts/evaluate.py:

1. RuleLanguageDetector - no training data. Every token is labelled as
     * English  if it is in the English vocabulary (all words of the English policy + sklearn's
                English stop words) - insurance words like "covered", "claim", "baggage" count here;
     * Tagalog  if it is a Tagalog function word / common word (TAGALOG_WORDS) or carries a Tagalog
                affix (nag-, mag-, ma-, na-, i-, pinaka-, -ng ...);
     * unknown  otherwise.
   The Tagalog share of the labelled tokens decides: < 0.25 english, > 0.80 tagalog, else taglish.
   Taglish is real code-switching, so the middle band is wide on purpose.

2. MLLanguageDetector - character 1-4-gram TF-IDF + logistic regression trained on the training
   split of the Q&A dataset, plus the three rule features. Character n-grams pick up Tagalog
   morphology ("nag", "ng ", "ang", "-in-") without a word list.
"""
import re

import numpy as np

from . import config

TAGALOG_WORDS = set("""
ang ng mga sa si ni kay kina sina ay at o pero kasi dahil kung kapag pag para upang nang na pa
ba po ho naman lang lamang din rin daw raw nga yata pala sana muna talaga kaya ngayon dito
diyan doon dyan
ako ko akin aking ikaw ka mo iyo iyong ikaw siya niya kanya kaniya kami namin amin tayo natin atin
kayo ninyo inyo sila nila kanila ito iyan iyon yan yun yung ito'y eto
ano anong ano-ano anu paano bakit saan sino sinong kailan magkano gaano ilan ilang alin aling
hindi di wala walang huwag wag ayaw meron mayroon may oo opo
bago pagkatapos habang hanggang mula galing tungkol bawat lahat iba ibang isa isang dalawa tatlo
ibang bansa bansa biyahe byahe sakit gamot ospital doktor ngipin pera bagahe maleta pasaporte
labi libing asawa anak magulang kapatid kapamilya bata resibo sasakyan kotse bahay trabaho araw
oras buwan taon gastos bayad babayaran sagot sinasagot sasagutin kompanya patunay dokumento
nawala nawalan ninakaw nanakaw nasira naantala nagkasakit nasaktan namatay naaksidente naospital
kailangan dapat maaari pwede puwede pwedeng gusto alam tulong tulungan magpagamot nagbago isip
paliwanag ipaliwanag sabihin tanong sagutan kumuha nakauwi umuwi pauwi pagbalik nakaalis
tumuloy natuloy maaga lalo pati kahit basta tapos sige salamat nasa mas pinaka sobra iyong ating
lang naman eh kasama sama mag ma pagdating pag-uwi positibo kalamidad lindol baha
bagyo giyera digmaan lasing buntis panganganak manganak nanganak pagbubuntis
""".split())

# Colloquial markers typical of Taglish chat (vs. formal Tagalog "ang/ng/maaari/sagot").
COLLOQUIAL = {"yung", "pag", "pwede", "pwedeng", "ayoko", "lang", "kasi", "naman", "dun", "nung", "ganun",
              "sya", "nila", "talaga", "kaya", "na", "tapos", "basta", "or", "and", "like", "so"}

TAGALOG_AFFIX = re.compile(
    r"^(nag|mag|pag|pinag|ipag|ipa|nakaka|naka|maka|pinaka|napaka|ika|ma|na|ni|in|i)-?"
    r"[a-z]{3,}(an|in|han|hin)?$")
ENGLISH_AFFIX_EXCEPTIONS = {"in", "inside", "into", "income", "insurance", "insured", "injury", "maximum", "may"}

TOKEN_RE = re.compile(r"[a-zA-ZñÑ'\-]+")


def tokenize(text: str) -> list[str]:
    return [t.strip("-'").lower() for t in TOKEN_RE.findall(text) if t.strip("-'")]


def _english_vocab() -> set:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    vocab = set(ENGLISH_STOP_WORDS)
    try:
        from .ingest import load_chunks
        for c in load_chunks():
            vocab.update(tokenize(c["text"]))
    except Exception:            # policy not available: stop words still give a usable detector
        pass
    vocab.update("covered cover covid claim claims trip flight luggage hotel abroad check ok okay "
                 "insurance refund delayed cancel cancelled passport visa phone laptop".split())
    # a few English-looking strings that are Tagalog words in practice
    vocab -= {"ang", "sa", "na", "pa", "ba", "ko", "mo", "may", "ay", "at", "o", "pag", "di", "nag"}
    return vocab


class RuleLanguageDetector:
    def __init__(self, low: float = 0.25, high: float = 0.80):
        self.low, self.high = low, high
        self.en_vocab = _english_vocab()

    def token_labels(self, text: str) -> list[tuple[str, str]]:
        out = []
        for t in tokenize(text):
            base = t.split("-")[-1] if "-" in t else t
            if t in TAGALOG_WORDS:
                out.append((t, "tl"))
            elif t in self.en_vocab or (base in self.en_vocab and "-" in t and len(base) > 3 and
                                         t.split("-")[0] in {"na", "i", "mag", "nag", "ma", "pa"}):
                # "na-delay", "i-claim": English root with Tagalog affix -> counts as both
                out.append((t, "mix") if "-" in t else (t, "en"))
            elif TAGALOG_AFFIX.match(t) and t not in ENGLISH_AFFIX_EXCEPTIONS:
                out.append((t, "tl"))
            else:
                out.append((t, "unk"))
        return out

    def features(self, text: str) -> np.ndarray:
        labs = [lab for _, lab in self.token_labels(text)]
        n = max(len(labs), 1)
        en = sum(lab == "en" for lab in labs) + 0.5 * sum(lab == "mix" for lab in labs)
        tl = sum(lab == "tl" for lab in labs) + 0.5 * sum(lab == "mix" for lab in labs)
        unk = sum(lab == "unk" for lab in labs)
        toks = tokenize(text)
        coll = sum(t in COLLOQUIAL for t in toks) / n
        starts_en = float(bool(labs) and labs[0] in ("en", "mix"))
        return np.array([en / n, tl / n, unk / n, tl / max(en + tl, 1e-9), coll, starts_en])

    def tagalog_share(self, text: str) -> float:
        return float(self.features(text)[3])

    def predict_one(self, text: str) -> str:
        share = self.tagalog_share(text)
        if share < self.low:
            return "english"
        if share > self.high:
            return "tagalog"
        return "taglish"

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]


class MLLanguageDetector:
    """Char n-gram TF-IDF + rule features -> logistic regression."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.rule = RuleLanguageDetector()
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")

    def _X(self, texts, fit=False):
        from scipy.sparse import csr_matrix, hstack
        T = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        R = csr_matrix(np.vstack([self.rule.features(t) for t in texts]) * 3.0)
        return hstack([T, R]).tocsr()

    def fit(self, texts, labels):
        self.clf.fit(self._X(list(texts), fit=True), list(labels))
        return self

    def predict(self, texts):
        return list(self.clf.predict(self._X(list(texts))))

    def predict_one(self, text):
        return self.predict([text])[0]


_DEFAULT = None


def detect_language(text: str) -> str:
    """Use the trained ML detector if models/language_classifier.joblib exists, else the rules."""
    global _DEFAULT
    if _DEFAULT is None:
        try:
            import joblib
            _DEFAULT = joblib.load(config.LANG_MODEL_FILE)
        except Exception:
            _DEFAULT = RuleLanguageDetector()
    return _DEFAULT.predict_one(text)
