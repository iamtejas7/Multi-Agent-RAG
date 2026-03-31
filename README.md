# TechNova Enterprise Knowledge Assistant

A multi-agent RAG (Retrieval-Augmented Generation) system that answers employee queries with citations. Built using LangChain, LangGraph, ChromaDB, AWS Bedrock, and Streamlit. Deployable to AWS Bedrock AgentCore Runtime via direct code deployment.

## Architecture

```
                                    Multi-Agent RAG Pipeline
                                    
User Query                                                                          Response
   │                                                                                   ▲
   ▼                                                                                   │
┌──────────┐    ┌───────────┐    ┌────────┐    ┌───────────┐    ┌────────────┐    ┌─────────┐
│ Rewriter │───▶│ Guardrail │───▶│ Router │───▶│ Retriever │───▶│ Specialist │───▶│ Quality │
│  Agent   │    │   Agent   │    │ Agent  │    │   Agent   │    │   Agent    │    │  Agent  │
└──────────┘    └───────────┘    └────────┘    └───────────┘    └────────────┘    └─────────┘
 Resolves        Blocks           Classifies    Fetches docs     Generates         Validates
 follow-up       irrelevant       query into    from ChromaDB    cited answer      response
 questions       queries          hr/it/fin/gen                  per domain        quality
```

### Agents

| Agent | Role |
|-------|------|
| **Rewriter Agent** | Resolves follow-up questions using conversation history into standalone queries |
| **Guardrail Agent** | Blocks irrelevant/off-topic queries (e.g., "best Netflix movies") before any processing |
| **Router Agent** | Uses LLM to classify the query into one of four domains: `hr`, `it`, `finance`, `general` |
| **Retriever Agent** | Queries the domain-specific ChromaDB collection and returns top-k relevant documents |
| **Specialist Agent** | Uses a domain-tuned prompt to generate a cited answer from retrieved context |
| **Quality Agent** | Validates that the response answers the question and includes proper citations |

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph (StateGraph) |
| **LLM** | AWS Bedrock (Claude) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) or AWS Bedrock Titan |
| **Vector Store** | ChromaDB (persistent locally, in-memory on AgentCore) |
| **Frontend** | Streamlit |
| **Framework** | LangChain |
| **Deployment** | AWS Bedrock AgentCore Runtime (direct code deploy) |

## Project Structure

```
.
├── agent.py                    # AgentCore Runtime entrypoint (@app.entrypoint)
├── app.py                      # Streamlit UI (local mode - runs graph locally)
├── app_agentcore.py            # Streamlit UI (cloud mode - calls AgentCore endpoint)
├── main.py                     # CLI entry point for local testing
├── config.py                   # Configuration (AWS, ChromaDB, models)
├── deploy.py                   # Deployment helper (wraps agentcore CLI)
├── ingest.py                   # Standalone data ingestion script
├── requirements.txt            # Full dependencies (local development)
├── requirements-agentcore.txt  # Slim dependencies (AgentCore deployment)
├── .bedrock_agentcore.yaml     # AgentCore CLI configuration (auto-generated)
├── .streamlit/
│   └── config.toml             # Streamlit config (suppresses venv warnings)
├── data/
│   ├── hr_policies.json        # 8 HR policy documents
│   ├── it_policies.json        # 7 IT policy documents
│   ├── finance_policies.json   # 6 Finance policy documents
│   └── general_knowledge.json  # 6 General knowledge documents
├── agents/
│   ├── state.py                # Shared AgentState (TypedDict)
│   ├── rewriter_agent.py       # Follow-up query resolution using chat history
│   ├── guardrail_agent.py      # Irrelevant query blocking
│   ├── router_agent.py         # Query classification (hr/it/finance/general)
│   ├── retriever_agent.py      # Document retrieval from ChromaDB
│   ├── specialist_agent.py     # Domain-specific response generation with citations
│   └── quality_agent.py        # Response quality validation
├── vectorstore/
│   └── store.py                # ChromaDB operations (persistent + ephemeral modes)
├── graph/
│   └── workflow.py             # LangGraph StateGraph workflow definition
└── utils/
    ├── citations.py            # Citation formatting utilities
    └── llm.py                  # AWS Bedrock LLM factory
```

## Local Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

The system uses AWS Bedrock for the LLM. Ensure your credentials are configured:

```bash
# Option A: Environment variables 
export AWS_REGION=us-east-2
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# CMD
set AWS_REGION=us-east-2
set AWS_ACCESS_KEY_ID=your-access-key
set AWS_SECRET_ACCESS_KEY=your-secret-key

# Option B: AWS CLI
aws configure
```

### 4. Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-2` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock LLM model ID |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `USE_BEDROCK_EMBEDDINGS` | `false` | Set `true` to use Bedrock Titan embeddings instead of local Sentence Transformers |
| `DEPLOYMENT_MODE` | `local` | Set `agentcore` for in-memory ChromaDB (auto-set by `agent.py`) |

## Running Locally

### Streamlit Web UI

```bash
streamlit run app.py
```

Features:
- Chat interface with conversation history and follow-up support
- Clickable sample questions in the sidebar
- Citation display with document IDs, titles, and update dates
- Agent trail showing the full processing pipeline
- Category badges (HR / IT / Finance / General)
- Guardrail blocking for irrelevant queries
- One-click document re-ingestion

### CLI Mode

```bash
python main.py
```

### Ingest Data Only

```bash
python ingest.py
```

On first run, documents are automatically ingested into ChromaDB and persisted to `chroma_db/`. Subsequent runs skip ingestion.

## Deploy to AWS Bedrock AgentCore

### Prerequisites

- AWS account with credentials configured (`aws configure`)
- AWS Bedrock model access enabled for your chosen model
- Python 3.10+ installed
- IAM permissions for AgentCore Runtime ([required permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html#runtime-permissions-starter-toolkit))

### Step 1: Install the AgentCore Starter Toolkit

```bash
pip install bedrock-agentcore-starter-toolkit
```

Verify installation:

```bash
agentcore --help
```

### Step 2: Configure the Agent

```bash
agentcore configure --entrypoint agent.py --name technova_assistant --requirements-file requirements-agentcore.txt --deployment-type direct_code_deploy --runtime PYTHON_3_13 --region us-east-2 --execution-role your_role --disable-memory --disable-otel --non-interactive
```

This creates `.bedrock_agentcore.yaml` with all deployment configuration.

### Step 3: Test Locally (Optional)

```bash
# Start the agent locally on port 8080
python agent.py

# In another terminal, test with curl
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many annual leaves do I get?"}'
```

### Step 4: Deploy to AgentCore Runtime

```bash
agentcore deploy --auto-update-on-conflict
```

This packages the code as a zip, uploads to S3, and creates the AgentCore Runtime. The first deployment takes ~30 seconds; subsequent updates are ~10 seconds.

### Step 5: Test the Deployed Agent

```bash
# Invoke the deployed agent
agentcore invoke '{"prompt": "How many annual leaves do I get?"}'

# Test with follow-up
agentcore invoke '{"prompt": "Can I carry forward unused days?"}' --session-id my-session

# Test guardrail (should be blocked)
agentcore invoke '{"prompt": "What are the best Netflix movies?"}'
```

### Step 6: Check Status

```bash
agentcore status
```

### Helper Script (Alternative)

Run configure + deploy in one command:

```bash
python deploy.py --region us-east-2 --name technova-assistant
```

### Connecting Streamlit to AgentCore Endpoint

Once deployed, use `app_agentcore.py` to point the Streamlit UI at the cloud endpoint:

```bash
# Get the ARN from agentcore status or .bedrock_agentcore.yaml
export AGENTCORE_AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-east-2:123456789012:runtime/your-agent-id
export AWS_REGION=us-east-2

streamlit run app_agentcore.py
```

### Manage the Deployment

```bash
# Stop the active session (saves costs)
agentcore stop-session

# Update after code changes
agentcore deploy

# Destroy all resources
agentcore destroy
```

## Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI for **local** mode -- runs the LangGraph pipeline directly |
| `app_agentcore.py` | Streamlit UI for **cloud** mode -- calls the deployed AgentCore endpoint via boto3 |
| `agent.py` | AgentCore Runtime entrypoint -- uses `@app.entrypoint` from `bedrock-agentcore` SDK |
| `deploy.py` | Helper script that runs `agentcore configure` + `agentcore deploy` |
| `requirements.txt` | Full dependencies including Streamlit and toolkit (for local dev) |
| `requirements-agentcore.txt` | Slim dependencies (for AgentCore deployment, no Streamlit) |

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

**Follow-up examples** (the Rewriter Agent resolves these using chat history):

```
User: What is the leave policy?
User: How many days can I carry forward?     ← resolved to "How many unused annual leave days can I carry forward?"
User: What about sick leave?                 ← resolved to "What is the sick leave policy?"
```

**Blocked queries** (the Guardrail Agent rejects these):

```
What are the best Netflix movies?
Give me a recipe for pasta
Who won the 2024 world cup?
```

## How Citations Work

Each response includes inline citations in the format `[Source N | DOC_ID: Title]` referencing the exact policy documents used. The citation panel shows:

- **Document ID** (e.g., HR-001, IT-003)
- **Document title**
- **Department**
- **Last updated date**

## Conversational Memory

The system maintains chat history within a session. The **Rewriter Agent** uses the last 6 conversation turns to resolve pronouns and implicit references in follow-up questions into fully self-contained standalone queries before routing them through the pipeline. This works in both the Streamlit UI and CLI modes.
