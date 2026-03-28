import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from graph.workflow import run_query
from vectorstore.store import ingest_documents, is_ingested

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: transparent;
        border-left: 4px solid #4a90d9;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.85rem;
        color: inherit;
    }
    .agent-trail {
        background-color: transparent;
        border-left: 4px solid #2ecc71;
        padding: 8px 12px;
        margin: 3px 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.8rem;
        font-family: monospace;
        color: inherit;
    }
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    .badge-hr { background-color: #e74c3c; }
    .badge-it { background-color: #3498db; }
    .badge-finance { background-color: #2ecc71; }
    .badge-general { background-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/building.png", width=60)
    st.markdown("### TechNova Solutions")
    st.markdown("**Enterprise Knowledge Assistant**")
    st.markdown("---")

    st.markdown("#### How it works")
    st.markdown("""
    1. **Router Agent** classifies your query
    2. **Retriever Agent** fetches relevant documents
    3. **Specialist Agent** generates an answer with citations
    4. **Quality Agent** validates the response
    """)

    st.markdown("---")
    st.markdown("#### Sample Questions")
    sample_questions = [
        "How many annual leaves do I get?",
        "What is the password policy?",
        "How do I submit travel expenses?",
        "Where are the company offices located?",
        "What is the employee referral bonus?",
        "How does the performance review work?",
        "What VPN should I use for remote access?",
        "What is the corporate credit card limit?",
    ]
    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_query"] = q

    st.markdown("---")
    if st.button("Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            stats = ingest_documents(force=True)
        st.success(f"Ingested: {stats}")

# ── Data Ingestion (auto on first run) ───────────────────────────────────────
if "ingested" not in st.session_state:
    if not is_ingested():
        with st.spinner("First run — ingesting documents into vector store..."):
            ingest_documents()
    st.session_state["ingested"] = True

# ── Chat History ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about HR policies, IT guidelines, finance procedures, and more.</div>',
    unsafe_allow_html=True,
)

# Display chat history
for entry in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        cat = entry.get("category", "general")
        badge_class = f"badge-{cat}"
        st.markdown(
            f'<span class="category-badge {badge_class}">{cat.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(entry["response"])

        if entry.get("citations"):
            with st.expander("View Citations", expanded=False):
                for c in entry["citations"]:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>{c["doc_id"]}</strong>: {c["title"]} '
                        f'({c["department"]}) | Updated: {c["last_updated"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if entry.get("agent_trail"):
            with st.expander("Agent Trail", expanded=False):
                for step in entry["agent_trail"]:
                    st.markdown(
                        f'<div class="agent-trail">{step}</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
user_query = st.chat_input("Ask a question about company policies...")

if prefill:
    user_query = prefill

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working on your query..."):
            try:
                history = [
                    {"query": h["query"], "response": h["response"]}
                    for h in st.session_state["chat_history"]
                ]
                result = run_query(user_query, chat_history=history)

                cat = result.get("category", "general")
                badge_class = f"badge-{cat}"
                st.markdown(
                    f'<span class="category-badge {badge_class}">{cat.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(result["response"])

                citations = result.get("citations", [])
                if citations:
                    with st.expander("View Citations", expanded=True):
                        for c in citations:
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>{c["doc_id"]}</strong>: {c["title"]} '
                                f'({c["department"]}) | Updated: {c["last_updated"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                agent_trail = result.get("agent_trail", [])
                if agent_trail:
                    with st.expander("Agent Trail", expanded=False):
                        for step in agent_trail:
                            st.markdown(
                                f'<div class="agent-trail">{step}</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state["chat_history"].append({
                    "query": user_query,
                    "response": result["response"],
                    "category": cat,
                    "citations": citations,
                    "agent_trail": agent_trail,
                })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Make sure AWS credentials are configured for Bedrock access.")
