from typing import Any, TypedDict


class AgentState(TypedDict):
    query: str
    original_query: str
    chat_history: list[dict[str, str]]
    is_relevant: bool
    category: str
    retrieved_docs: list[dict[str, Any]]
    context: str
    response: str
    citations: list[dict[str, str]]
    agent_trail: list[str]
    error: str | None
