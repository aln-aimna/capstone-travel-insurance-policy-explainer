"""Policy PDF -> clean pages -> section-aware chunks with page numbers (the 'R' in RAG).

Why section-aware: the policy is 47 pages of numbered clauses ("SECTION 14: LOSS OR DAMAGE OF
BAGGAGE...", "ADDITIONAL EXCLUSIONS APPLICABLE TO: ...", "EXCLUSIONS APPLICABLE TO ALL SECTIONS").
Tagging each chunk with the heading it falls under lets the chatbot cite "[Section 14, p. 27]"
and lets retrieval match questions to the right clause even when the chunk text itself does not
repeat the benefit name.
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

from . import config

FOOTER_PATTERNS = [
    r"Prudential Guarantee and Assurance, Inc\.\s*\n\s*COYIUTO HOUSE.*?www\.prudentialguarantee\.com",
]
HEADER_RE = re.compile(r"^\s*\d+\s*\n\s*(Covid-19 Endorsement|Travel Shield Insurance Policy)\s*\n\s*Page\s*of\s*\d+\s*\n", re.I)

# Headings that start a new logical part of the policy, most specific first.
HEADING_RES = [
    (re.compile(r"SECTION\s+(\d+)\s*:\s*([A-Z][A-Z0-9 /&,()\-’']+)"), "section"),
    (re.compile(r"ADDITIONAL EXCLUSIONS APPLICABLE TO\s*:?\s*\n?\s*([^\n]+)"), "add_excl"),
    (re.compile(r"(EXCLUSIONS APPLICABLE TO ALL SECTIONS)"), "plain"),
    (re.compile(r"(OPTIONAL BENEFITS)"), "plain"),
    (re.compile(r"\n([A-F])\)\s+([A-Z][A-Z0-9 ,()\-]{3,})"), "optional"),
    (re.compile(r"(GENERAL CONDITIONS UNDER THE POLICY|GENERAL PROVISIONS|COVERAGE LIMITS|"
                r"EFFECTIVE AND TERMINATION DATES OF INDIVIDUAL INSURANCE|LIMITS OF COVERAGE|"
                r"DESCRIPTION OF BENEFITS|DEFINITIONS|COVID-19 ENDORSEMENT|"
                r"SPECIFIC EXCLUSIONS COVID-19 HEALTH EXPENSES|TERMS AND CONDITIONS OF COVID-19 RIDER|"
                r"AGGRAVATION OF THE RISK|OBLIGATIONS IN CASE OF LOSS)"), "plain"),
]
# Title-case sub-headings inside General Conditions (one per line in the PDF).
GENERAL_CONDITION_TITLES = [
    "Premium Payment", "Grace Period", "Currency", "Aggregate Limit of Liability", "Assistance and Cooperation",
    "Due Diligence", "Notice of Claim or Loss", "Claim Forms", "Proof of Loss", "Physical Examination and Autopsy",
    "To Whom Claims Paid", "Time of Payment of Claim", "Subrogation", "Right of Recovery", "Cumulative Insurances",
    "Misstatement of Age", "Reinstatement of Policy", "Renewal Conditions", "Cancellation", "Termination of Insurance",
    "Fraud", "Fraudulent Claims", "Clerical Error", "Assignment", "Age Limitation", "Geographical Limits",
    "Terms and Conditions", "Complying with Policy Conditions", "Entire Contract", "Governing Law", "Arbitration",
    "Legal Action", "Free Look Provision", "Civil Code 1250 Waiver Clause",
]
GC_RE = re.compile(r"\n(" + "|".join(re.escape(t) for t in GENERAL_CONDITION_TITLES) + r")\s*\n")


def _clean(text: str) -> str:
    for pat in FOOTER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.S)
    text = HEADER_RE.sub("", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)     # re-join words hyphenated at line ends
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def extract_pages(pdf_path: Path = config.POLICY_PDF) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    return [{"page": i + 1, "text": _clean(p.extract_text() or "")} for i, p in enumerate(reader.pages)]


def _title(match, kind) -> str:
    if kind == "section":
        name = match.group(2).strip().title().replace("’S", "’s")
        name = re.sub(r"\s+(If|Up|In|Including)\b.*$", "", name) if len(name) > 70 else name
        name = re.sub(r"\s*\(\w*$", "", name).strip()
        return f"Section {match.group(1)} - {name}"
    if kind == "add_excl":
        return "Additional Exclusions - " + match.group(1).strip()[:80]
    if kind == "optional":
        name = re.sub(r"\s*\(\w*$", "", match.group(2).strip().title()).strip()
        return f"Optional Benefit {match.group(1)} - {name}"
    return match.group(1).strip().title()


def _headings(text: str):
    """Yield (char_offset, heading) for every heading found in a page."""
    found = []
    padded = "\n" + text
    for rx, kind in HEADING_RES:
        for m in rx.finditer(padded):
            found.append((max(m.start() - 1, 0), _title(m, kind)))
    for m in GC_RE.finditer(padded):
        found.append((max(m.start() - 1, 0), "General Conditions - " + m.group(1)))
    found.sort()
    # drop near-duplicates (two patterns matching the same spot)
    out = []
    for off, h in found:
        if out and abs(out[-1][0] - off) < 5:
            continue
        out.append((off, h))
    return out


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping windows that prefer to cut at a line or sentence end (as in full_app.py)."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
    """Split each page at headings, then into windows. Each chunk keeps its page and the heading
    in force at its start (headings carry over from previous pages)."""
    chunks, current = [], "COVID-19 Endorsement"
    for p in pages:
        text, page = p["text"], p["page"]
        if page >= 4 and current.startswith("COVID"):
            current = "Travel Shield Insurance Policy"
        marks = _headings(text)
        bounds = [0] + [off for off, _ in marks] + [len(text)]
        titles = [current] + [h for _, h in marks]
        for i in range(len(bounds) - 1):
            seg = text[bounds[i]:bounds[i + 1]].strip()
            if titles[i]:
                current = titles[i]
            if len(seg) < 40:
                continue
            for piece in _split(seg, size, overlap):
                chunks.append({"chunk_id": f"p{page:02d}-{len(chunks):03d}", "page": page,
                               "section": current, "text": piece})
        if marks:
            current = marks[-1][1]
    return chunks


def save_chunks(chunks: list[dict], path: Path = config.CHUNKS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path = config.CHUNKS_FILE) -> list[dict]:
    if not Path(path).exists():
        chunks = build_chunks(extract_pages())
        save_chunks(chunks, path)
        return chunks
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    pages = extract_pages()
    chunks = build_chunks(pages)
    save_chunks(chunks)
    print(f"{len(pages)} pages -> {len(chunks)} chunks -> {config.CHUNKS_FILE}")
