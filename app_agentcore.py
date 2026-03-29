"""
Streamlit UI that connects to the AgentCore Runtime endpoint
instead of running the LangGraph workflow locally.

Usage:
    streamlit run app_agentcore.py

Set environment variables before running:
    export AGENTCORE_AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/your-agent-id
    export AWS_REGION=us-east-1
"""
import json
import os
import uuid

import boto3
import streamlit as st

AGENT_RUNTIME_ARN = os.getenv("AGENTCORE_AGENT_RUNTIME_ARN", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 2rem; }
    .citation-box {
        background-color: transparent; border-left: 4px solid #4a90d9;
        padding: 10px 15px; margin: 5px 0; border-radius: 0 5px 5px 0;
        font-size: 0.85rem; color: inherit;
    }
    .agent-trail {
        background-color: transparent; border-left: 4px solid #2ecc71;
        padding: 8px 12px; margin: 3px 0; border-radius: 0 5px 5px 0;
        font-size: 0.8rem; font-family: monospace; color: inherit;
    }
    .category-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600; color: white;
    }
    .badge-hr { background-color: #e74c3c; }
    .badge-it { background-color: #3498db; }
    .badge-finance { background-color: #2ecc71; }
    .badge-general { background-color: #9b59b6; }
    .badge-blocked { background-color: #95a5a6; }
</style>
""", unsafe_allow_html=True)


def invoke_agentcore(query: str, chat_history: list) -> dict:
    """Invoke the AgentCore Runtime endpoint."""
    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    payload = json.dumps({
        "prompt": query,
        "chat_history": chat_history,
    }).encode("utf-8")

    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.get("session_id", str(uuid.uuid4())),
        payload=payload,
        qualifier="DEFAULT",
    )

    # Read streaming response
    chunks = []
    for chunk in response.get("response", []):
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(str(chunk))

    raw = "".join(chunks)
    try:
        parsed = json.loads(raw)
        # Handle double-encoded JSON (agent returns json.dumps string)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {"response": raw, "category": "general", "citations": [], "agent_trail": []}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### TechNova Solutions")
    st.markdown("**Enterprise Knowledge Assistant**")
    st.markdown(f"*Connected to AgentCore Runtime*")
    st.markdown("---")

    runtime_arn = st.text_input(
        "Agent Runtime ARN",
        value=AGENT_RUNTIME_ARN,
        type="password",
        help="ARN of the deployed AgentCore Runtime agent",
    )
    if runtime_arn:
        AGENT_RUNTIME_ARN = runtime_arn

    st.markdown("---")
    st.markdown("#### Sample Questions")
    sample_questions = [
        "How many annual leaves do I get?",
        "What is the password policy?",
        "How do I submit travel expenses?",
        "Where are the company offices located?",
    ]
    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_query"] = q

# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about HR policies, IT guidelines, finance procedures, and more.</div>',
    unsafe_allow_html=True,
)

if not AGENT_RUNTIME_ARN:
    st.warning("Please set the AGENTCORE_AGENT_RUNTIME_ARN environment variable or enter it in the sidebar.")
    st.stop()

# Display chat history
for entry in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        cat = entry.get("category", "general")
        st.markdown(
            f'<span class="category-badge badge-{cat}">{cat.upper()}</span>',
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
                result = invoke_agentcore(user_query, history)

                cat = result.get("category", "general")
                st.markdown(
                    f'<span class="category-badge badge-{cat}">{cat.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(result.get("response", ""))

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
                    "response": result.get("response", ""),
                    "category": cat,
                    "citations": citations,
                    "agent_trail": agent_trail,
                })

            except Exception as e:
                st.error(f"Error invoking AgentCore: {str(e)}")
                st.info("Check that AGENTCORE_AGENT_RUNTIME_ARN is correct and the agent is deployed.")
