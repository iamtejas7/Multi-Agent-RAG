from agents.state import AgentState
from vectorstore.store import query_collection


def retrieve_documents(state: AgentState) -> AgentState:
    category = state["category"]
    query = state["query"]

    docs = query_collection(category=category, query=query)

    return {
        **state,
        "retrieved_docs": docs,
        "agent_trail": state.get("agent_trail", []) + [
            f"Retriever ({category}) -> Found {len(docs)} documents"
        ],
    }
