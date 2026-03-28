from langgraph.graph import END, StateGraph

from agents.guardrail_agent import check_relevance
from agents.quality_agent import check_quality
from agents.retriever_agent import retrieve_documents
from agents.rewriter_agent import rewrite_query
from agents.router_agent import route_query
from agents.specialist_agent import generate_response
from agents.state import AgentState


def _after_guardrail(state: AgentState) -> str:
    return "router" if state.get("is_relevant", False) else END


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("guardrail", check_relevance)
    workflow.add_node("router", route_query)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("specialist", generate_response)
    workflow.add_node("quality_check", check_quality)

    workflow.set_entry_point("rewriter")
    workflow.add_edge("rewriter", "guardrail")
    workflow.add_conditional_edges("guardrail", _after_guardrail)
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "specialist")
    workflow.add_edge("specialist", "quality_check")
    workflow.add_edge("quality_check", END)

    return workflow.compile()


def run_query(query: str, chat_history: list[dict[str, str]] | None = None) -> AgentState:
    graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "chat_history": chat_history or [],
        "is_relevant": False,
        "category": "",
        "retrieved_docs": [],
        "context": "",
        "response": "",
        "citations": [],
        "agent_trail": [],
        "error": None,
    }

    result = graph.invoke(initial_state)
    return result
