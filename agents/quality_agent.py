from agents.state import AgentState
from utils.llm import get_llm

QUALITY_CHECK_PROMPT = """You are a quality assurance agent. Review the following response to an employee query and evaluate it.

Employee Query: {query}
Category: {category}

Generated Response:
{response}

Check the following:
1. Does the response actually answer the question?
2. Does the response include source citations?
3. Is the response professional and clear?

If the response is satisfactory, respond with: PASS
If the response needs improvement, respond with: FAIL - [brief reason]

Verdict:"""


def check_quality(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    prompt = QUALITY_CHECK_PROMPT.format(
        query=state["query"],
        category=state["category"],
        response=state["response"],
    )
    result = llm.invoke(prompt)
    verdict = result.content.strip()

    passed = verdict.upper().startswith("PASS")

    return {
        **state,
        "agent_trail": state.get("agent_trail", []) + [
            f"Quality Check -> {'PASSED' if passed else verdict}"
        ],
        "error": None if passed else verdict,
    }
