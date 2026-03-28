from agents.state import AgentState
from utils.llm import get_llm

GUARDRAIL_PROMPT = """You are a guardrail agent for an enterprise knowledge assistant at TechNova Solutions.
Your job is to determine if the employee query is relevant to the enterprise knowledge base.

RELEVANT topics include:
- HR: leave, benefits, insurance, performance reviews, onboarding, work from home, conduct, hiring
- IT: passwords, VPN, software, devices, security, cloud, support, technical infrastructure
- Finance: expenses, reimbursements, procurement, credit cards, payroll, salary, budgets, invoices
- General company: company info, offices, learning programs, referrals, CSR, communication, policies

IRRELEVANT topics include:
- Entertainment (movies, TV shows, music, games, sports scores)
- Personal advice (dating, cooking recipes, fitness routines)
- General world knowledge unrelated to the company (history, geography, science trivia)
- Homework or academic questions
- Any topic clearly outside an enterprise workplace context

Respond with ONLY one word: RELEVANT or IRRELEVANT

Employee query: {query}

Verdict:"""

REJECTION_MESSAGE = (
    "I'm sorry, but this question is outside the scope of the Enterprise Knowledge Assistant. "
    "I can only help with company-related topics such as HR policies, IT guidelines, "
    "finance procedures, and general company information. Please ask a work-related question."
)


def check_relevance(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    prompt = GUARDRAIL_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()

    is_relevant = verdict.startswith("RELEVANT")

    new_state = {**state, "is_relevant": is_relevant}
    trail = state.get("agent_trail", []) + [
        f"Guardrail -> {'RELEVANT' if is_relevant else 'IRRELEVANT (blocked)'}"
    ]
    new_state["agent_trail"] = trail

    if not is_relevant:
        new_state["response"] = REJECTION_MESSAGE
        new_state["citations"] = []
        new_state["category"] = "blocked"

    return new_state
