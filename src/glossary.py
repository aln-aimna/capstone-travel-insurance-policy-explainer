"""Tagalog / Taglish -> English insurance glossary for cross-lingual query expansion.

The policy is written in English, so a Tagalog question such as "Nawala ng airline ang bagahe ko"
shares almost no words with the clause that answers it. Before retrieval we append English
equivalents of the Tagalog words we recognise ("lost baggage luggage airline"). Hyphenated
Taglish verbs are split so the English root is kept ("na-delay" -> "delay", "i-claim" -> "claim").
"""
import re

GLOSSARY = {
    # people
    "asawa": "spouse", "anak": "child children dependent", "bata": "child minor", "magulang": "parents",
    "nanay": "parent mother", "tatay": "parent father", "mama": "parent", "papa": "parent",
    "kapatid": "brother sister sibling", "kapamilya": "immediate family member relative",
    "pamilya": "family", "lola": "grandparent relative", "lolo": "grandparent relative",
    "kasama": "travelling companion", "amo": "employer", "boss": "employment",
    # health
    "sakit": "illness sickness", "nagkasakit": "illness sickness", "magkasakit": "illness sickness",
    "may sakit": "illness", "ospital": "hospital", "naospital": "hospital confinement in-patient",
    "ma-ospital": "hospital confinement", "na-hospital": "hospital confinement", "na-confine": "confinement hospital in-patient",
    "na-admit": "admitted hospital in-patient", "doktor": "doctor physician", "gamot": "medicine prescribed medicines",
    "gamutan": "treatment medical", "pagpapagamot": "medical treatment", "magpagamot": "medical treatment",
    "ngipin": "dental teeth tooth", "pustiso": "dentures", "nasaktan": "injury injured", "na-injure": "injury",
    "pinsala": "injury damage", "aksidente": "accident", "naaksidente": "accident", "nadisgrasya": "accident",
    "namatay": "death died accidental death", "mamatay": "death", "patay": "death", "bangkay": "mortal remains",
    "labi": "mortal remains", "libing": "burial funeral", "lamay": "funeral", "buntis": "pregnancy",
    "pagbubuntis": "pregnancy", "panganganak": "childbirth", "nanganak": "childbirth", "check-up": "consultation",
    "altapresyon": "hypertension", "high blood": "hypertension", "hika": "asthma", "dati nang sakit": "pre-existing",
    "lasing": "inebriated alcohol", "nakainom": "alcohol", "kapansanan": "disability",
    "naputol": "amputation", "naputulan": "amputation", "bulag": "blindness sight",
    # travel
    "biyahe": "trip travel", "byahe": "trip travel", "bumiyahe": "travel trip", "lakad": "trip",
    "eroplano": "aircraft flight", "lipad": "flight", "flight": "flight", "naantala": "delay delayed",
    "na-delay": "delay delayed", "late": "delay", "naiwan": "missed connection", "na-miss": "missed connection",
    "kinansela": "cancellation cancel", "kanselado": "cancellation", "i-cancel": "cancellation cancel",
    "kanselahin": "cancellation", "nag-cancel": "cancellation", "hindi natuloy": "cancellation",
    "di natuloy": "cancellation", "umuwi ng maaga": "curtailment cut short return home early",
    "umuwi": "return home", "pauwi": "return home journey back", "pag-uwi": "return home", "pagbalik": "return",
    "pinas": "philippines", "pilipinas": "philippines", "ibang bansa": "abroad overseas international",
    "abroad": "overseas abroad", "barko": "cruise ship", "pantalan": "port", "hotel": "hotel accommodation",
    "tirahan": "accommodation residence", "matutuluyan": "accommodation", "pasaporte": "passport",
    "embahada": "embassy passport", "motor": "motorcycle", "motorsiklo": "motorcycle", "kotse": "car vehicle",
    "sasakyan": "vehicle", "inarkila": "rented rental", "nirentahan": "rented rental", "nagrent": "rented rental",
    "bundok": "mountaineering trekking", "nagdive": "diving", "sumisid": "diving",
    # property
    "bagahe": "baggage luggage", "bag": "baggage luggage", "bags": "baggage luggage", "wallet": "money cash",
    "pitaka": "money cash", "pitaka ko": "money cash", "maleta": "luggage baggage suitcase", "gamit": "belongings personal effects",
    "nawala": "lost loss", "nawalan": "lost loss", "naiwala": "lost", "ninakaw": "stolen robbery theft burglary",
    "nanakaw": "stolen robbery theft burglary", "nanakawan": "stolen robbery theft", "na-holdap": "robbery",
    "snatch": "robbery", "nasira": "damage damaged", "sira": "damage", "nabasag": "damage fragile",
    "pera": "money cash", "cash": "money cash", "alahas": "jewelry valuable", "relo": "watch valuable",
    "cellphone": "mobile phone valuable", "selpon": "mobile phone", "resibo": "receipts official receipt",
    "pulis": "police report", "ari-arian": "property",
    # money & admin
    "magkano": "amount limit benefit amount", "bayad": "payment premium", "babayaran": "pay payable",
    "sagot": "covered cover", "sasagutin": "covered", "sinasagot": "covered", "sakop": "covered",
    "ibalik": "refund", "refund": "refund", "premyum": "premium", "presyo": "premium price",
    "deadline": "within days notice", "ilang araw": "days period", "gaano katagal": "period days time",
    "kailan": "when date period", "patunay": "proof", "dokumento": "documents", "papeles": "documents",
    "mag-claim": "claim", "i-claim": "claim", "kumuha": "obtain", "reklamo": "complaint dispute",
    "tinanggihan": "denied", "na-deny": "denied", "pandaraya": "fraud", "tseke": "check",
    "bangko": "bank", "palitan": "exchange rate currency", "dolyar": "dollar currency",
    # disasters & exclusions
    "lindol": "earthquake natural disaster", "baha": "flood natural disaster", "bagyo": "storm typhoon natural disaster",
    "giyera": "war", "digmaan": "war", "kaguluhan": "riot", "welga": "strike", "nag-strike": "strike",
    "terorismo": "terrorism", "kalamidad": "natural disaster",
}

MULTIWORD = sorted([k for k in GLOSSARY if " " in k], key=len, reverse=True)
HYPHEN_PREFIX = re.compile(r"\b(?:na|i|mag|nag|ma|pa|ipa|maka|naka)-([a-z]{3,})\b", re.I)


def expand_query(query: str) -> str:
    """Return the query followed by English equivalents of recognised Tagalog words."""
    q = query.lower()
    extra = []
    for phrase in MULTIWORD:
        if phrase in q:
            extra.append(GLOSSARY[phrase])
    for tok in re.findall(r"[a-zñ\-]+", q):
        if tok in GLOSSARY:
            extra.append(GLOSSARY[tok])
    # Taglish verbs with an English root: "na-delay" -> "delay"
    extra += [m.group(1) for m in HYPHEN_PREFIX.finditer(q)]
    if not extra:
        return query
    return query + " || " + " ".join(dict.fromkeys(" ".join(extra).split()))
