from agents.state import AgentState
from utils.llm import get_llm

ROUTER_PROMPT = """You are a query routing agent for an enterprise knowledge assistant at TechNova Solutions.
Your job is to classify the employee query into exactly ONE of these categories:

- hr: Questions about leave policies, benefits, health insurance, performance reviews, onboarding, work from home, code of conduct, employee welfare
- it: Questions about passwords, VPN, software, laptops, devices, data security, cloud services, IT support, technical infrastructure
- finance: Questions about travel expenses, reimbursements, procurement, credit cards, payroll, salary, budgets, invoices, accounts payable
- general: Questions about company overview, office locations, learning programs, referral programs, CSR, communication channels, or anything else

Respond with ONLY the category name (hr, it, finance, or general). Nothing else.

Employee query: {query}

Category:"""


def route_query(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    prompt = ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)
    category = response.content.strip().lower().replace(".", "")

    valid_categories = {"hr", "it", "finance", "general"}
    if category not in valid_categories:
        category = "general"

    return {
        **state,
        "category": category,
        "agent_trail": state.get("agent_trail", []) + [f"Router -> {category}"],
    }
