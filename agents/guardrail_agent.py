from agents.state import AgentState
from utils.llm import get_llm

GUARDRAIL_PROMPT = """You are a guardrail agent for an enterprise knowledge assistant at TechNova Solutions.
Classify the employee query into exactly one of these three categories:

GREETING - General greetings, pleasantries, or conversational openers like "hello", "hi", "hey", "good morning", "how are you", "thanks", "thank you", "what can you do", "help", "who are you", etc.

RELEVANT - Questions related to the enterprise knowledge base:
- HR: leave, benefits, insurance, performance reviews, onboarding, work from home, conduct, hiring
- IT: passwords, VPN, software, devices, security, cloud, support, technical infrastructure
- Finance: expenses, reimbursements, procurement, credit cards, payroll, salary, budgets, invoices
- General company: company info, offices, learning programs, referrals, CSR, communication, policies

IRRELEVANT - Topics completely outside the workplace:
- Entertainment (movies, TV shows, music, games, sports scores)
- Personal advice (dating, cooking recipes, fitness routines)
- General world knowledge unrelated to the company (history, geography, science trivia)
- Homework or academic questions

Respond with ONLY one word: GREETING, RELEVANT, or IRRELEVANT

Employee query: {query}

Verdict:"""

GREETING_MESSAGE = (
    "Hello! I'm the TechNova Enterprise Knowledge Assistant. "
    "I can help you with questions about:\n\n"
    "- **HR Policies** - Leave, benefits, performance reviews, onboarding, WFH\n"
    "- **IT Guidelines** - Passwords, VPN, software, devices, security\n"
    "- **Finance Procedures** - Expenses, reimbursements, payroll, procurement\n"
    "- **General Company Info** - Offices, learning programs, referrals, CSR\n\n"
    "How can I assist you today?"
)

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

    if verdict.startswith("GREETING"):
        return {
            **state,
            "is_relevant": False,
            "response": GREETING_MESSAGE,
            "citations": [],
            "category": "greeting",
            "agent_trail": state.get("agent_trail", []) + ["Guardrail -> GREETING"],
        }

    if verdict.startswith("RELEVANT"):
        return {
            **state,
            "is_relevant": True,
            "agent_trail": state.get("agent_trail", []) + ["Guardrail -> RELEVANT"],
        }

    return {
        **state,
        "is_relevant": False,
        "response": REJECTION_MESSAGE,
        "citations": [],
        "category": "blocked",
        "agent_trail": state.get("agent_trail", []) + ["Guardrail -> IRRELEVANT (blocked)"],
    }
