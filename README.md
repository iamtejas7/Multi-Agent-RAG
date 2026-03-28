# TechNova Enterprise Knowledge Assistant

A multi-agent RAG (Retrieval-Augmented Generation) system that answers employee queries with citations, built using LangChain, LangGraph, ChromaDB, AWS Bedrock, and Streamlit.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  User Query  │────▶│ Router Agent │────▶│ Retriever Agent │────▶│ Specialist Agent │────▶│ Quality Agent  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────────┘     └────────────────┘
                     Classifies query     Fetches relevant docs    Generates answer         Validates response
                     into a domain        from ChromaDB             with citations           quality & citations
```

### Agents

| Agent | Role |
|-------|------|
| **Router Agent** | Uses LLM to classify the query into one of four domains: `hr`, `it`, `finance`, `general` |
| **Retriever Agent** | Queries the domain-specific ChromaDB collection and returns top-k relevant documents |
| **Specialist Agent** | Uses a domain-tuned prompt to generate a cited answer from retrieved context |
| **Quality Agent** | Validates that the response answers the question and includes proper citations |

### Tech Stack

- **Orchestration**: LangGraph (StateGraph)
- **LLM**: AWS Bedrock (Claude 3 Sonnet)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) or AWS Bedrock Titan
- **Vector Store**: ChromaDB (in-memory / persistent)
- **Frontend**: Streamlit
- **Framework**: LangChain

## Project Structure

```
├── config.py                  # Configuration (AWS, ChromaDB, models)
├── requirements.txt           # Python dependencies
├── ingest.py                  # Standalone data ingestion script
├── main.py                    # CLI entry point
├── app.py                     # Streamlit web UI
├── data/
│   ├── hr_policies.json       # 8 HR policy documents
│   ├── it_policies.json       # 7 IT policy documents
│   ├── finance_policies.json  # 6 Finance policy documents
│   └── general_knowledge.json # 6 General knowledge documents
├── agents/
│   ├── state.py               # Shared AgentState (TypedDict)
│   ├── router_agent.py        # Query classification agent
│   ├── retriever_agent.py     # Document retrieval agent
│   ├── specialist_agent.py    # Domain-specific response generation
│   └── quality_agent.py       # Response quality validation
├── vectorstore/
│   └── store.py               # ChromaDB ingestion & query operations
├── graph/
│   └── workflow.py            # LangGraph workflow definition
└── utils/
    ├── citations.py           # Citation formatting utilities
    └── llm.py                 # AWS Bedrock LLM factory
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

The system uses AWS Bedrock for the LLM. Ensure your AWS credentials are configured:

```bash
export AWS_REGION=us-east-2
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

Or configure via `~/.aws/credentials`.

### 3. Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | Bedrock LLM model ID |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `USE_BEDROCK_EMBEDDINGS` | `false` | Set `true` to use Bedrock embeddings instead of local Sentence Transformers |

## Usage

### Streamlit Web UI

```bash
streamlit run app.py
```

Features:
- Chat interface with conversation history
- Clickable sample questions in the sidebar
- Citation display with document IDs, titles, and update dates
- Agent trail showing the full processing pipeline
- Category badges (HR / IT / Finance / General)
- One-click document re-ingestion

### CLI Mode

```bash
python main.py
```

### Ingest Data Only

```bash
python ingest.py
```

## Dummy Data

The system ships with 27 realistic enterprise policy documents for a fictional company **TechNova Solutions**:

- **HR** (8 docs): Leave policies, sick leave, maternity/paternity, performance reviews, health insurance, WFH policy, onboarding, code of conduct
- **IT** (7 docs): Passwords/MFA, VPN, software licensing, data classification, laptop policy, IT service desk, cloud/SaaS usage
- **Finance** (6 docs): Travel expenses, procurement, corporate credit cards, payroll, budget planning, invoice processing
- **General** (6 docs): Company overview, office locations, L&D programs, referral program, CSR initiatives, internal communication

## Sample Queries

```
How many annual leaves do I get?
What is the password policy?
How do I submit travel expenses?
Where are the company offices located?
What is the employee referral bonus?
How does the performance review process work?
What VPN should I use for remote access?
What is the corporate credit card limit for managers?
```

## How Citations Work

Each response includes inline citations in the format `[Source N | DOC_ID: Title]` referencing the exact policy documents used. The citation panel shows:
- **Document ID** (e.g., HR-001, IT-003)
- **Document title**
- **Department**
- **Last updated date**
