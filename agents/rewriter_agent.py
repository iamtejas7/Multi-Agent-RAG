from agents.state import AgentState
from utils.llm import get_llm

REWRITER_PROMPT = """You are a query rewriter for an enterprise knowledge assistant.
Given the conversation history and the latest user query, rewrite the query into a fully self-contained standalone question.

If the latest query is already self-contained and does not reference prior conversation, return it as-is.
If it contains pronouns like "it", "that", "they", "this" or implicit references to earlier topics, resolve them using the conversation history.

Conversation history:
{chat_history}

Latest user query: {query}

Rewritten standalone query:"""


def _format_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "(No prior conversation)"
    lines = []
    for turn in chat_history[-6:]:
        lines.append(f"User: {turn['query']}")
        response_preview = turn["response"][:200]
        lines.append(f"Assistant: {response_preview}...")
    return "\n".join(lines)


def rewrite_query(state: AgentState) -> AgentState:
    chat_history = state.get("chat_history", [])

    if not chat_history:
        return {
            **state,
            "original_query": state["query"],
            "agent_trail": state.get("agent_trail", []) + [
                "Rewriter -> No history, query unchanged"
            ],
        }

    llm = get_llm(temperature=0.0)
    prompt = REWRITER_PROMPT.format(
        chat_history=_format_history(chat_history),
        query=state["query"],
    )
    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    changed = rewritten.lower() != state["query"].lower()

    return {
        **state,
        "original_query": state["query"],
        "query": rewritten,
        "agent_trail": state.get("agent_trail", []) + [
            f"Rewriter -> {'Rewritten: ' + rewritten if changed else 'Query unchanged'}"
        ],
    }
