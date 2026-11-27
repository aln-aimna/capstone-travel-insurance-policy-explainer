"""Travel Shield Policy Explainer + Claim Assistant (Streamlit).

    streamlit run app.py

Secrets (.streamlit/secrets.toml, see secrets.toml.example):
    ANTHROPIC_API_KEY   - optional; without it the chatbot runs in offline mode (reference answers)
    SMTP_* / CLAIMS_TO  - optional; without them the claim PDF can still be downloaded
"""
import json
import os
from collections import Counter
from datetime import date

import pandas as pd
import streamlit as st

try:                                   # also accept keys from a .env file, not only secrets.toml
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src import config
from src.chatbot import PolicyChatbot
from src.claims import (LOSS_TYPES, POLICY_TIPS, claim_email_bodies, reference_number, render_claim_pdf,
                        required_documents, suggest_loss_types, total_claimed, validate_claim)
from src.emailer import SMTPSettings, build_message, send
from src.ingest import load_chunks
from src.plans import ASSISTANCE_HOTLINE, limit_for_loss_types, load_schedule

st.set_page_config(page_title="Travel Shield Policy Explainer", page_icon="🛡️", layout="wide")


def secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


@st.cache_resource(show_spinner="Loading the policy and models...")
def get_bot(api_key: str, model: str) -> PolicyChatbot:
    return PolicyChatbot(api_key=api_key or None, model=model)


EXAMPLES = [
    "Nawala ng airline yung maleta ko, ano pwede ko i-claim?",
    "Sagot ba ang COVID-19 kung ma-ospital ako sa Japan?",
    "My flight was delayed 9 hours. What can I claim?",
    "Covered ba ako pag nag-scuba diving ako?",
    "Gaano katagal bago ko dapat i-file ang claim?",
    "Is a pre-existing condition covered?",
]

ss = st.session_state
for k, v in {"messages": [], "meta": [], "pending": None, "claim_desc": "", "loss_types": [],
             "submitted": None}.items():
    ss.setdefault(k, v)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ Travel Shield")
    st.caption("Policy Explainer & Claim Assistant")
    api_key = secret("ANTHROPIC_API_KEY")
    # model = st.selectbox("Claude model", ["claude-haiku-5-5", "claude-sonnet-5-5"], index=0,
    #                      help="Haiku is fast and cheap; Sonnet for harder questions.")
    model = "claude-haiku-5-5"
    
    bot = get_bot(api_key, model)
    if bot.online:
        # st.success("Claude connected", icon="✅")
        st.success("Online mode", icon="✅")
    else:
        # st.warning("Offline mode: no ANTHROPIC_API_KEY found, so answers come from the checked Q&A dataset only. "
        #            "Put the key in .streamlit/secrets.toml (or .env) and restart to let Claude answer any question.",
        #            icon="📴")
        st.warning("Offline mode: no ANTHROPIC_API_KEY found, so answers come from the checked Q&A dataset only. ",
                    icon="📴")
    # st.caption(f"Model: {model}")
    # st.metric("API cost this session", f"${bot.total_cost:.4f}")
    st.divider()
    st.caption("Ask in **English, Tagalog or Taglish**. Answers explain the policy wording only; "
               "the insurer's claims team decides every claim.")
    with st.expander("💡 Example questions"):
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"side_ex{i}", use_container_width=True):
                ss.pending = ex
                st.rerun()
    if st.button("Clear chat", use_container_width=True):
        ss.messages, ss.meta = [], []
        st.rerun()
    st.divider()
    st.markdown("**🚨 Emergency abroad?**")
    st.caption(f"Call the 24-hour travel assistance hotline first: **{ASSISTANCE_HOTLINE}**. "
               "For medical, dental and accident cases, not calling can lead to a denied claim. "
               f"Head office: {config.INSURER_PHONES} · {config.INSURER_EMAIL}")
st.title("🛡️ Travel Shield Policy Explainer")
# Remove Insights as not part of scope of the system
# tab_chat, tab_claim, tab_insights = st.tabs(["💬 Ask about your policy", "📝 Claim Assistant", "📊 Insights"])
tab_chat, tab_claim, tab_cov, tab_policy = st.tabs(["💬 Ask about your policy", "📝 Claim Assistant", "📋 Coverages", "📄 Policy"])

# ----------------------------------------------------------------------------- chat
MODE_BADGE = {"claude": "🤖 Claude", "offline": "📚 reference answer", "guardrail": "🛑 out of scope", "error": "⚠️ fallback"}


def show_sources(meta):
    if not meta:
        return
    st.caption(f"{MODE_BADGE.get(meta['mode'], meta['mode'])} · language: **{meta['language']}** · "
               f"topic: **{config.TOPIC_LABELS.get(meta['topic'], meta['topic'])}** ({meta['topic_confidence']:.0%})"
               + (f" · {meta['input_tokens']}+{meta['output_tokens']} tokens · ${meta['cost_usd']:.5f}" if meta["mode"] == "claude" else ""))
    with st.expander("Sources from the policy"):
        for c in meta["chunks"][:4]:
            st.markdown(f"**p. {c['page']} · {c['section']}**")
            st.caption(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))


with tab_chat:
    # The conversation is drawn into this container, which sits ABOVE the input box. The input is read
    # first (so we know whether a question is coming) and the messages are written into the container
    # afterwards, so the box always stays under the latest answer.
    chat_area = st.container()
    prompt = st.chat_input("Itanong ang tungkol sa iyong policy / Ask about your policy...")
    if ss.pending and not prompt:
        prompt, ss.pending = ss.pending, None

    with chat_area:
        if not ss.messages and not prompt:
            st.info("Kumusta! Ask me anything about your Travel Shield coverage, in English, Tagalog or Taglish.", icon="👋")
            cols = st.columns(3)
            for i, ex in enumerate(EXAMPLES):
                if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
                    ss.pending = ex
                    st.rerun()

        for msg, meta in zip(ss.messages, ss.meta):
            with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    show_sources(meta)

        if prompt:
            history = list(ss.messages)
            ss.messages.append({"role": "user", "content": prompt})
            ss.meta.append(None)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🛡️"):
                answer = st.write_stream(bot.answer_stream(prompt, history))
                result = bot.last_result.to_dict()
                show_sources(result)
                wants_claim = any(w in prompt.lower() for w in ("claim", "file", "reimburse", "i-claim", "mag-claim"))
                if result["suggest_claim"] or (wants_claim and result["topic"] in config.TOPICS[:12]):
                    st.info("Ready to file? Open the **📝 Claim Assistant** tab: your description is already filled in.", icon="📝")
                    ss.claim_desc = prompt
                    ss.loss_types = suggest_loss_types(prompt, result["topic"])
            ss.messages.append({"role": "assistant", "content": result["answer"] or (answer if isinstance(answer, str) else "")})
            ss.meta.append(result)

# ----------------------------------------------------------------------------- claim assistant
with tab_claim:
    st.subheader("Notification of Claim - Travel Insurance")
    st.caption("This follows the insurer's Notification of Claim form. Submit within **90 days** of the assistance "
               "(written notice within 30 days, Policy p. 42). Submitting does not guarantee approval.")
    if ss.submitted:
        st.success(f"Claim **{ss.submitted['ref']}** {ss.submitted['status']}", icon="✅")

    with st.expander("1 · Insured's information", expanded=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        ins = {"name": c1.text_input("Insured's name *"), "age": c2.number_input("Age", 0, 99, 30),
               "sex": c3.selectbox("Sex", ["", "Male", "Female"])}
        c1, c2 = st.columns(2)
        ins["policy_number"] = c1.text_input("Policy number *")
        ins["address"] = c2.text_input("Complete address *")
        c1, c2, c3 = st.columns(3)
        ins["home"], ins["office"], ins["mobile"] = c1.text_input("Home tel."), c2.text_input("Office tel."), c3.text_input("Mobile *")
        c1, c2 = st.columns(2)
        ins["email"], ins["fax"] = c1.text_input("E-mail address * (a copy is sent here)"), c2.text_input("Fax")

    with st.expander("2 · Claimant's information"):
        same = st.checkbox("The claimant is the insured", value=True)
        cl = {"same_as_insured": same}
        if not same:
            c1, c2, c3 = st.columns([3, 1, 1])
            cl.update(name=c1.text_input("Claimant's name"), age=c2.number_input("Claimant age", 0, 120, 30),
                      sex=c3.selectbox("Claimant sex", ["", "Male", "Female"]))
            c1, c2, c3 = st.columns([3, 1, 2])
            cl.update(address=c1.text_input("Claimant's complete address"),
                      birthday=str(c2.date_input("Birthday", value=None, min_value=date(1900, 1, 1)) or ""),
                      relationship=c3.text_input("Relationship to insured"))
            c1, c2, c3 = st.columns(3)
            cl.update(home=c1.text_input("Claimant home tel."), office=c2.text_input("Claimant office tel."),
                      mobile=c3.text_input("Claimant mobile"))

    with st.expander("3 · What happened & type of loss", expanded=True):
        desc = st.text_area("Describe what happened (English, Tagalog or Taglish) *", value=ss.claim_desc, height=100,
                            placeholder="e.g. Na-delay ng 10 oras ang flight ko sa Incheon dahil sa bagyo...")
        if st.button("✨ Suggest type of loss from my description"):
            info = bot.analyze(desc) if desc.strip() else {"topic": None}
            ss.loss_types = suggest_loss_types(desc, info["topic"])
            ss.claim_desc = desc
            st.rerun()
        loss_types = st.multiselect("Type of loss *", LOSS_TYPES, default=[t for t in ss.loss_types if t in LOSS_TYPES])
        loss_other = st.text_input("Others (please specify)") if "Others" in loss_types else ""
        limits = limit_for_loss_types(loss_types)
        for lt in loss_types:
            lim = next((x for x in limits if x["loss_type"] == lt), None)
            tip = POLICY_TIPS.get(lt, "")
            if lim or tip:
                st.caption(f"ℹ️ **{lt}:** " + (f"limit {lim['limit_text']} ({lim['section']}). " if lim else "") + tip)

    with st.expander("4 · Details of injury, sickness or incident", expanded=True):
        c1, c2 = st.columns(2)
        inc = {"description": desc, "date_of_loss": str(c1.date_input("Date of loss / assistance *", value=date.today())),
               "place": c2.text_input("Place / address where it happened")}
        medical = any(t in loss_types for t in ("Medical Expenses", "Repatriation Expenses", "Personal Accident"))
        if medical:
            inc["nature"] = st.text_input("Nature and condition of injury or sickness")
            c1, c2 = st.columns(2)
            inc.update(hospital_dates=c1.text_input("Hospitalization / consultation dates"),
                       hospital_name=c2.text_input("Name of hospital"))
            c1, c2, c3 = st.columns([2, 2, 1])
            inc.update(physician_name=c1.text_input("Attending physician"), hospital_address=c2.text_input("Hospital address"),
                       hospital_phone=c3.text_input("Hospital tel. / fax"))
            inc["prior_treatment"] = st.text_input("Date/s of any prior treatment of the same illness (if none, write 'none')")

    phy = {"applicable": False}
    if medical:
        with st.expander("5 · Attending physician statement (if applicable)"):
            phy["applicable"] = st.checkbox("Include physician statement details")
            if phy["applicable"]:
                c1, c2 = st.columns([1, 3])
                phy.update(patient_type=c1.radio("Patient", ["Out-Patient", "In-Patient"]),
                           diagnosis=c2.text_area("Complete diagnosis of medical condition", height=70))
                c1, c2, c3 = st.columns(3)
                phy.update(admission_date=str(c1.date_input("Admission date", value=None) or ""),
                           discharge_date=str(c2.date_input("Discharged date", value=None) or ""),
                           consultation_date=str(c3.date_input("Date of consultation", value=None) or ""))
                c1, c2 = st.columns(2)
                phy["chronic"] = c1.radio("Continuous treatment for a chronic disease?", ["NO", "YES"], horizontal=True)
                phy["other_disease"] = c2.radio("Other disease affecting present condition?", ["NO", "YES"], horizontal=True)
                if phy["other_disease"] == "YES":
                    phy["other_disease_details"] = st.text_input("If YES, please describe")
                st.caption("The doctor signs the physician statement on the original form.")

    with st.expander("6 · Official receipts submitted", expanded=True):
        receipts_df = st.data_editor(
            pd.DataFrame([{"or_number": "", "description": "", "currency": "PHP", "amount": 0.0}]),
            num_rows="dynamic", use_container_width=True, key="receipts",
            column_config={"or_number": st.column_config.TextColumn("O.R. number"),
                           "description": st.column_config.TextColumn("Description", width="large"),
                           "currency": st.column_config.SelectboxColumn("Currency", options=["PHP", "USD", "EUR", "JPY", "SGD", "KRW", "HKD", "AUD", "Other"]),
                           "amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="%.2f")})
        receipts = receipts_df.fillna("").to_dict("records")
        st.metric("Total amount claimed", f"{total_claimed({'receipts': receipts}):,.2f}")
        php_total = sum(float(r.get("amount") or 0) for r in receipts if r.get("currency") == "PHP")
        cap = sum(x["max_php"] for x in limit_for_loss_types(loss_types))
        if cap and php_total > cap:
            st.warning(f"The PHP receipts total {php_total:,.2f}, above the PHP {cap:,.0f} maximum in your Schedule of Coverages "
                       "for the selected type(s) of loss. You can still submit; the insurer pays at most the limit.", icon="⚠️")
        total_words = st.text_input("Total amount claimed in words (optional)")
        if len({r.get('currency') for r in receipts if float(r.get('amount') or 0) > 0}) > 1:
            st.caption("Mixed currencies: the insurer converts at the rate on the date of each expense (Currency, p. 41).")

    with st.expander("7 · Payment details", expanded=True):
        c1, c2 = st.columns(2)
        pay = {"payee_name": c1.text_input("Name of payee as it should appear on the check *", value=ins["name"]),
               "payee_relationship": c2.text_input("If payee is not the insured, relationship to insured")}
        pay["method"] = st.radio("Mode of payment", ["Check", "Direct credit to bank account"], horizontal=True)
        if pay["method"] == "Direct credit to bank account":
            c1, c2, c3 = st.columns([2, 2, 1])
            pay.update(bank_account_name=c1.text_input("Bank account name *"),
                       bank_account_number=c2.text_input("Bank account number *", type="password",
                                                         help="Shown only in the attached PDF; masked in the e-mail body."),
                       account_type=c3.selectbox("Account type", ["Savings", "Current / Checking"]))
            c1, c2 = st.columns(2)
            pay.update(bank_name_address=c1.text_input("Bank complete address *"),
                       bank_relationship=c2.text_input("Relationship to patient (if account is not the patient's)"))
            st.caption("Per the form: direct credit applies only up to the insurer's limit; inter-branch crediting costs may be "
                       "deducted; a processing fee applies to incorrect details.")

    with st.expander("8 · Documents checklist & attachments", expanded=True):
        docs = required_documents(loss_types)
        ready = [d for d in docs if st.checkbox(d, key=f"doc_{d}")]
        police = False
        if "Luggage and Personal Effects" in loss_types:
            police = st.checkbox("I reported the theft/robbery to the police within 24 hours (if applicable)")
        files = st.file_uploader("Attach scanned documents (PDF, JPG, PNG; max 20 MB total)", type=["pdf", "jpg", "jpeg", "png"],
                                 accept_multiple_files=True)
        size_mb = sum(f.size for f in files or []) / 1e6
        if size_mb > 20:
            st.error(f"Attachments are {size_mb:.1f} MB; please keep the total under 20 MB.")

    with st.expander("9 · Authority, release and declaration", expanded=True):
        st.markdown(
            "- **Authority:** you allow the insurer to request medical records and test results related to this claim "
            "from hospitals, clinics and doctors.\n"
            "- **Release and subrogation:** payment of this claim is a full and final settlement, and the insurer takes over "
            "your right to recover the loss from anyone responsible.\n"
            "- **Declaration:** everything in this form is complete and true and the amounts are due under the policy.\n\n"
            "Full wording: Claim-Form.pdf, page 2.")
        agreed = st.checkbox("I have read and agree to the Authority, Release and Declaration statements *")
        c1, c2 = st.columns([3, 1])
        sig_name = c1.text_input("Signature: type your full name as printed name of insured / claimant *")
        sig_date = str(c2.date_input("Date", value=date.today()))

    claim = {"insured": ins, "claimant": cl, "loss_types": loss_types, "loss_other": loss_other, "incident": inc,
             "physician": phy, "receipts": receipts, "total_in_words": total_words, "payment": pay,
             "documents_ready": ready, "police_report": police, "attachment_names": [f.name for f in files or []],
             "declaration_agreed": agreed, "signature_name": sig_name, "signature_date": sig_date}

    c1, c2 = st.columns(2)
    preview = c1.button("👁️ Check & preview PDF", use_container_width=True)
    submit = c2.button("📨 Submit claim & send e-mail", type="primary", use_container_width=True)
    if preview or submit:
        errors, warnings = validate_claim(claim)
        if size_mb > 20:
            errors.append("Attachments exceed 20 MB.")
        for w in warnings:
            st.warning(w, icon="⏰")
        if errors:
            st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            ref = reference_number(claim)
            pdf = render_claim_pdf(claim, ref)
            st.download_button("⬇️ Download Notification of Claim (PDF)", pdf, file_name=f"{ref}.pdf", mime="application/pdf")
            if submit:
                smtp = SMTPSettings.load(st.secrets if hasattr(st, "secrets") else None)
                text, html = claim_email_bodies(claim, ref)
                attachments = [(f"{ref}.pdf", pdf)] + [(f.name, f.getvalue()) for f in files or []]
                attachments.append((f"{ref}.json", json.dumps({k: v for k, v in claim.items() if k != "payment"},
                                                               ensure_ascii=False, indent=2, default=str).encode()))
                subject = f"[Travel Shield Claim] {ref} - {ins['name']} - {', '.join(loss_types)}"
                if not smtp.configured:
                    st.warning("E-mail is not configured (SMTP_* and CLAIMS_TO in secrets). Download the PDF above and "
                               "send it with your documents, or ask the app owner to set up e-mail.", icon="✉️")
                else:
                    try:
                        with st.spinner("Sending e-mail..."):
                            send(smtp, build_message(smtp, subject, text, html, attachments, claimant_email=ins["email"]))
                        ss.submitted = {"ref": ref, "status": f"was e-mailed to the claims team ({smtp.claims_to})"
                                        + (f" with a copy to {ins['email']}." if smtp.copy_to_claimant else ".")}
                        st.success(f"Sent! Reference **{ref}**. Keep your original receipts: the insurer may ask for them.", icon="✅")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Could not send the e-mail: {type(e).__name__}: {e}. Your PDF is ready to download above.")

# ----------------------------------------------------------------------------- insights
# with tab_insights:
#     st.subheader("Model evaluation (held-out test set)")
#     mfile = config.RESULTS / "metrics.json"
#     if mfile.exists():
#         m = json.loads(mfile.read_text())
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Language ID accuracy", f"{m['language']['ml']['accuracy']:.0%}")
#         c2.metric("Topic accuracy (unseen items)", f"{m['topic']['logreg_weak']['accuracy']:.0%}")
#         c3.metric("Retrieval Hit@3 (unseen questions)", f"{m['retrieval']['unseen']['full']['hit@3']:.0%}")
#         c4.metric("Retrieval Hit@3 (FAQ paraphrases)", f"{m['retrieval']['paraphrase']['full']['hit@3']:.0%}")
#         for img in ["retrieval_ablation.png", "topic_by_language.png", "language_confusion.png"]:
#             p = config.CHARTS / img
#             if p.exists():
#                 st.image(str(p), use_container_width=True)
#     else:
#         st.info("Run `python scripts/evaluate.py` to generate the metrics.")
#     st.subheader("This session")
#     asked = [m for m in ss.meta if m]
#     if asked:
#         c1, c2 = st.columns(2)
#         c1.bar_chart(pd.Series(Counter(config.TOPIC_LABELS[m["topic"]] for m in asked)), horizontal=True)
#         c2.bar_chart(pd.Series(Counter(m["language"] for m in asked)))
#     else:
#         st.caption("Ask a few questions to see topics and languages here.")

# ----------------------------------------------------------------------------- coverages
with tab_cov:
    st.subheader(" Coverage Limits")
    st.caption("Maximum amounts per benefit, in Philippine Pesos, from the Certificate of Insurance. "
               "These are limits, not guaranteed payouts: every claim is still subject to the policy's terms, "
               "conditions and exclusions.")
    sched = load_schedule()
    search = st.text_input("Search a benefit", placeholder="e.g. baggage, dental, flight delay")
    view = sched
    if search.strip():
        view = sched[sched["benefit"].str.contains(search.strip(), case=False, na=False)]
    st.dataframe(
        view[["benefit", "limit_text", "policy_section"]].rename(columns={
            "benefit": "Benefit", "limit_text": "Sum insured (PHP)", "policy_section": "Policy section"}),
        hide_index=True, use_container_width=True, height=min(38 * (len(view) + 1) + 4, 800))
    if view.empty:
        st.info("No benefit matches that search.")
    st.info(f"🚨 Medical, dental or accident case abroad? Call the 24-hour assistance hotline first: "
            f"**{ASSISTANCE_HOTLINE}**. Not calling can lead to a denied claim.")

# ----------------------------------------------------------------------------- full policy
import re as _re


@st.cache_data(show_spinner=False)
def policy_sections() -> list[dict]:
    """Group the section-tagged chunks into sections (in page order) and merge them into readable text.
    Chunks overlap by ~150 characters, so the repeated start of each chunk is trimmed."""
    sections = []
    for c in load_chunks():
        if not sections or sections[-1]["section"] != c["section"]:
            sections.append({"section": c["section"], "pages": [], "text": ""})
        sec = sections[-1]
        if c["page"] not in sec["pages"]:
            sec["pages"].append(c["page"])
        prev, nxt = sec["text"], c["text"]
        cut = 0
        for n in range(min(len(prev), len(nxt), 300), 20, -1):     # longest overlap of prev-end / next-start
            if prev.endswith(nxt[:n]):
                cut = n
                break
        sec["text"] = (prev + ("\n" if prev and not cut else "") + nxt[cut:]).strip()
    return sections


def page_label(pages):
    return f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}-{pages[-1]}"


def highlight(text: str, query: str) -> str:
    safe = _re.sub(r"([*_`#>\[\]])", r"\\\1", text).replace("\n", "  \n")
    for w in sorted({w for w in _re.findall(r"[A-Za-z]{3,}", query)}, key=len, reverse=True):
        safe = _re.sub(rf"(?i)\b({_re.escape(w)}\w*)", r"**\1**", safe)
    return safe


with tab_policy:
    st.subheader("Travel Shield Insurance Policy")
    c1, c2 = st.columns([3, 1])
    c1.caption("The full policy wording, including the COVID-19 Endorsement (47 pages). "
               "Tip: for a plain-language answer, ask in the 💬 chat tab instead.")
    if config.POLICY_PDF.exists():
        c2.download_button("⬇️ Download full policy (PDF)", config.POLICY_PDF.read_bytes(),
                           file_name="Travel-Shield-ePolicy.pdf", mime="application/pdf", use_container_width=True)
    sections = policy_sections()
    mode = st.radio("View", ["🔎 Search", "📚 Browse by section"], horizontal=True, label_visibility="collapsed")
    if mode == "🔎 Search":
        q = st.text_input("Search the policy (English, Tagalog or Taglish)",
                          placeholder="e.g. baggage delay, pre-existing, nawala ang pasaporte")
        if q.strip():
            hits = bot.retriever.retrieve(q, k=8).chunks
            if not hits:
                st.info("No matching clause found. Try other words, or ask in the chat tab.")
            for h in hits:
                with st.container(border=True):
                    st.markdown(f"**{h['section']}** · p. {h['page']}")
                    st.markdown(highlight(h["text"], q))
        else:
            st.caption("Type a word or a question; the most relevant clauses appear here with their page numbers.")
    else:
        labels = [f"{s['section']}  ({page_label(s['pages'])})" for s in sections]
        pick = st.selectbox("Section", range(len(sections)), format_func=lambda i: labels[i])
        sec = sections[pick]
        st.markdown(f"#### {sec['section']}")
        st.caption(page_label(sec["pages"]))
        with st.container(border=True, height=520):
            st.markdown(highlight(sec["text"], ""))
