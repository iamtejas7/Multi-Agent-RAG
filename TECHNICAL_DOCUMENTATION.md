# Technical Code Documentation - TechNova Enterprise Knowledge Assistant

## Table of Contents

1. [System Overview](#1-system-overview)
2. [End-to-End Flow Explanation](#2-end-to-end-flow-explanation)
3. [Project Structure](#3-project-structure)
4. [Configuration - config.py](#4-configuration---configpy)
5. [Vector Store - vectorstore/store.py](#5-vector-store---vectorstorepy)
6. [Utilities](#6-utilities)
   - [LLM Factory - utils/llm.py](#61-llm-factory---utilsllmpy)
   - [Citations - utils/citations.py](#62-citations---utilscitationspy)
7. [Agent State - agents/state.py](#7-agent-state---agentsstatepy)
8. [Agents (Detailed)](#8-agents-detailed)
   - [Rewriter Agent - agents/rewriter_agent.py](#81-rewriter-agent)
   - [Guardrail Agent - agents/guardrail_agent.py](#82-guardrail-agent)
   - [Router Agent - agents/router_agent.py](#83-router-agent)
   - [Retriever Agent - agents/retriever_agent.py](#84-retriever-agent)
   - [Specialist Agent - agents/specialist_agent.py](#85-specialist-agent)
   - [Quality Agent - agents/quality_agent.py](#86-quality-agent)
9. [LangGraph Workflow - graph/workflow.py](#9-langgraph-workflow---graphworkflowpy)
10. [Entry Points](#10-entry-points)
    - [CLI - main.py](#101-cli---mainpy)
    - [Data Ingestion - ingest.py](#102-data-ingestion---ingestpy)
    - [Streamlit Local - app.py](#103-streamlit-local---apppy)
    - [AgentCore Entrypoint - agent.py](#104-agentcore-entrypoint---agentpy)
    - [Streamlit Cloud - app_agentcore.py](#105-streamlit-cloud---app_agentcorepy)
    - [Deployment Script - deploy.py](#106-deployment-script---deploypy)
11. [Data Files](#11-data-files)
12. [Key Concepts for Beginners](#12-key-concepts-for-beginners)

---

## 1. System Overview

This is a **Multi-Agent RAG (Retrieval-Augmented Generation)** system. Instead of using one monolithic AI to answer questions, the system uses **6 specialized agents** working in a pipeline. Each agent has one specific job. They pass data to each other through a shared state dictionary.

**What is RAG?**
RAG is a technique where instead of relying solely on an LLM's training data, you first *retrieve* relevant documents from a database, then pass those documents to the LLM as context so it can *generate* an accurate, grounded answer.

**What is Multi-Agent?**
Instead of one AI doing everything, you split the work into specialized agents. Think of it like an office where a receptionist (Router) directs your question to the right department (HR/IT/Finance), and that department's expert (Specialist) answers you.

**Technology Stack:**
- **LangGraph**: Orchestrates the agent pipeline as a directed graph
- **LangChain**: Provides the LLM wrapper (`ChatBedrock`) to talk to AWS Bedrock
- **ChromaDB**: Vector database that stores document embeddings for similarity search
- **AWS Bedrock**: Cloud service providing the LLM (Claude) and embedding model (Titan)
- **Streamlit**: Python web framework for the chat UI
- **AWS Bedrock AgentCore**: Cloud hosting platform for deploying the agent

---

## 2. End-to-End Flow Explanation

Here is exactly what happens when a user types a question like *"Can I carry forward unused leave?"* after previously asking *"What is the leave policy?"*:

```
Step 1: USER INPUT
        User types: "Can I carry forward unused leave?"
        Chat history contains: [{"query": "What is the leave policy?", "response": "..."}]
                    |
                    v
Step 2: REWRITER AGENT
        Sees chat history exists.
        Sends to LLM: "Given this conversation, rewrite the query as standalone"
        LLM returns: "Can I carry forward unused annual leave days to the next year?"
        State updated: query = rewritten version, original_query = original
                    |
                    v
Step 3: GUARDRAIL AGENT
        Sends rewritten query to LLM: "Is this GREETING, RELEVANT, or IRRELEVANT?"
        LLM returns: "RELEVANT"
        State updated: is_relevant = True
                    |
                    v
Step 4: ROUTER AGENT (only runs if is_relevant=True)
        Sends query to LLM: "Classify into hr, it, finance, or general"
        LLM returns: "hr"
        State updated: category = "hr"
                    |
                    v
Step 5: RETRIEVER AGENT
        Queries ChromaDB collection "hr_policies" with the rewritten query
        ChromaDB converts query to embedding vector, finds 4 most similar documents
        Returns: [HR-001 Annual Leave Policy, HR-002 Sick Leave Policy, ...]
        State updated: retrieved_docs = [...]
                    |
                    v
Step 6: SPECIALIST AGENT
        Formats retrieved docs as numbered sources
        Sends to LLM with HR-specific prompt: "You are the HR Policy Specialist..."
        LLM generates answer using ONLY the provided sources, with citations
        State updated: response = "Yes, you can carry forward up to 5 days... [Source 1 | HR-001]"
                    |
                    v
Step 7: QUALITY AGENT
        Sends the generated response to LLM: "Does this answer the question? Has citations?"
        LLM returns: "PASS"
        State updated: agent_trail += "Quality Check -> PASSED"
                    |
                    v
Step 8: RESPONSE RETURNED
        Final state contains: response, citations, category, agent_trail
        UI displays the answer with citation panel and agent trail
```

**What happens for irrelevant queries (e.g., "Best Netflix movies")?**

```
Step 1: User types: "Best Netflix movies"
Step 2: Rewriter -> No history, query unchanged
Step 3: Guardrail -> LLM says "IRRELEVANT"
        State: is_relevant=False, response=rejection message, category="blocked"
Step 4: Graph hits END (conditional edge skips Router/Retriever/Specialist/Quality)
Step 5: Rejection message returned directly
```

**What happens for greetings (e.g., "Hello")?**

```
Step 1: User types: "Hello"
Step 2: Rewriter -> No history, query unchanged
Step 3: Guardrail -> LLM says "GREETING"
        State: is_relevant=False, response=greeting message with capabilities list
Step 4: Graph hits END
Step 5: Greeting message returned directly
```

---

## 3. Project Structure

```
Multi Agentic RAG/
|
|-- config.py                    # Central configuration (AWS, models, paths)
|
|-- vectorstore/
|   |-- __init__.py              # Makes this a Python package
|   |-- store.py                 # All ChromaDB operations (embed, store, query)
|
|-- utils/
|   |-- __init__.py
|   |-- llm.py                   # Creates LLM instances (Bedrock Claude)
|   |-- citations.py             # Formats citations for display
|
|-- agents/
|   |-- __init__.py
|   |-- state.py                 # Defines the shared data structure (AgentState)
|   |-- rewriter_agent.py        # Agent 1: Resolves follow-up questions
|   |-- guardrail_agent.py       # Agent 2: Blocks irrelevant queries
|   |-- router_agent.py          # Agent 3: Classifies query domain
|   |-- retriever_agent.py       # Agent 4: Fetches documents from vector DB
|   |-- specialist_agent.py      # Agent 5: Generates cited answer
|   |-- quality_agent.py         # Agent 6: Validates response quality
|
|-- graph/
|   |-- __init__.py
|   |-- workflow.py              # Wires all agents into a LangGraph pipeline
|
|-- data/
|   |-- hr_policies.json         # 8 HR policy documents
|   |-- it_policies.json         # 7 IT policy documents
|   |-- finance_policies.json    # 6 Finance policy documents
|   |-- general_knowledge.json   # 6 General knowledge documents
|
|-- main.py                      # CLI entry point (for terminal testing)
|-- app.py                       # Streamlit UI (local mode)
|-- app_agentcore.py             # Streamlit UI (cloud mode, calls AgentCore)
|-- agent.py                     # AgentCore Runtime entrypoint
|-- deploy.py                    # Deployment helper script
|-- ingest.py                    # Standalone data ingestion script
|-- requirements.txt             # Full dependencies (local dev)
|-- requirements-agentcore.txt   # Slim dependencies (cloud deployment)
|-- .bedrock_agentcore.yaml      # AgentCore CLI configuration
|-- .streamlit/config.toml       # Streamlit settings
```

---

## 4. Configuration - config.py

This file holds all configurable values used across the project. Centralizing configuration here means you only need to change one file to switch models, regions, or settings.

```python
import os
```
- `os` module is used to read environment variables and construct file paths.

```python
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
```
- Reads the `AWS_REGION` environment variable. If not set, defaults to `"us-east-2"`.
- This tells boto3 (AWS SDK) which AWS data center to connect to.

```python
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
```
- The model ID for the **LLM** used by all agents (Router, Specialist, etc.).
- Claude Haiku is a fast, cost-efficient model. You can change this to Claude Sonnet for better quality.

```python
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
```
- The model ID for the **embedding model** that converts text into numerical vectors.
- Titan Embed v2 produces 1024-dimensional vectors.

```python
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
```
- `os.path.dirname(__file__)` = the folder containing config.py (project root).
- `os.path.join(...)` = joins it with `"chroma_db"` to get `<project_root>/chroma_db`.
- This is where ChromaDB saves its database files on disk (local mode only).

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
```
- Path to the `data/` folder containing the JSON policy documents.

```python
COLLECTION_MAP = {
    "hr": "hr_policies",
    "it": "it_policies",
    "finance": "finance_policies",
    "general": "general_knowledge",
}
```
- Maps each domain category to its ChromaDB collection name.
- When the Router Agent says `"hr"`, we look up `"hr_policies"` as the collection to query.

```python
TOP_K_RESULTS = 4
```
- How many similar documents to retrieve from ChromaDB for each query.
- Higher = more context but slower and more tokens. 4 is a good balance for 27 documents.

---

## 5. Vector Store - vectorstore/store.py

This is the most complex utility file. It handles everything related to ChromaDB: creating clients, embedding text, storing documents, and querying for similar documents.

### Imports and Module-Level Variables

```python
import json
import os

import chromadb
from chromadb.config import Settings
```
- `json`: For reading JSON data files and serializing API requests.
- `chromadb`: The vector database library. It stores text as numerical vectors and finds similar ones.
- `Settings`: ChromaDB configuration (imported but used in older code paths; kept for compatibility).

```python
from config import (
    AWS_REGION,
    BEDROCK_EMBEDDING_MODEL_ID,
    CHROMA_PERSIST_DIR,
    COLLECTION_MAP,
    DATA_DIR,
    TOP_K_RESULTS,
)
```
- Imports all configuration values from `config.py`.

```python
_client: chromadb.ClientAPI | None = None
_embedding_fn = None
```
- **Module-level singletons** (global variables). These ensure we only create ONE ChromaDB client and ONE embedding function per process, instead of creating new ones for every query.
- The `_` prefix is a Python convention meaning "private, don't use outside this module".

### BedrockTitanEmbeddingFunction Class

```python
FALLBACK_HF_MODEL = "all-MiniLM-L6-v2"
```
- Name of the open-source Sentence Transformers model used as fallback if Bedrock Titan is unavailable.
- MiniLM-L6-v2 produces 384-dimensional vectors and runs locally without AWS.

```python
class BedrockTitanEmbeddingFunction:
    """Custom Bedrock Titan embedding function that calls boto3 directly,
    avoiding ChromaDB's wrapper which double-encodes the model ID."""
```
- We wrote our own embedding class instead of using ChromaDB's built-in `AmazonBedrockEmbeddingFunction` because the built-in one has a bug where it double URL-encodes the colon in `amazon.titan-embed-text-v2:0`, causing AWS signature errors.

```python
    def __init__(self, model_id: str, region: str):
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
```
- **`__init__`**: Called when you create a new instance: `BedrockTitanEmbeddingFunction(model_id=..., region=...)`.
- `boto3.client("bedrock-runtime", ...)`: Creates a connection to the AWS Bedrock Runtime API, which is the service that runs ML models.
- Stores the client and model ID as instance variables (`self._client`, `self._model_id`).

```python
    def name(self) -> str:
        return "bedrock-titan"
```
- ChromaDB requires embedding functions to have a `name()` method. It uses this internally to track which embedding function a collection was created with.

```python
    def __call__(self, input: list[str]) -> list[list[float]]:
        import json
        embeddings = []
        for text in input:
            response = self._client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text}),
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])
        return embeddings
```
- **`__call__`**: Makes the object callable like a function: `fn(["some text"])`.
- ChromaDB calls this method with a list of text strings and expects back a list of embedding vectors (lists of floats).
- For each text:
  1. Calls `invoke_model` on Bedrock with the text as `inputText`.
  2. Reads the response body (a JSON stream).
  3. Extracts the `"embedding"` field (a list of 1024 floats for Titan v2).
- Returns all embeddings as a list of lists.

### _get_embedding_function()

```python
def _get_embedding_function():
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
```
- **Singleton pattern**: If we already created an embedding function, return it immediately. This avoids recreating expensive resources on every call.
- `global _embedding_fn`: Tells Python that `_embedding_fn` refers to the module-level variable, not a local one.

```python
    try:
        fn = BedrockTitanEmbeddingFunction(
            model_id=BEDROCK_EMBEDDING_MODEL_ID,
            region=AWS_REGION,
        )
        fn(["test"])
        _embedding_fn = fn
```
- Creates a Titan embedding function and immediately tests it with the word `"test"`.
- If the test succeeds (AWS credentials work, model is accessible), we use Titan.

```python
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Bedrock Titan embeddings unavailable (%s). Falling back to SentenceTransformers.", e
        )
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=FALLBACK_HF_MODEL
        )
```
- If Titan fails (no credentials, no network, model access disabled), we fall back to the open-source SentenceTransformers model.
- `SentenceTransformerEmbeddingFunction` downloads the model (~80MB) on first use and runs it locally on CPU.
- A warning log is printed so you know the fallback is being used.

**Important**: Titan embeddings (1024-dim) and MiniLM embeddings (384-dim) are **incompatible**. If you switch between them, you must delete the `chroma_db/` folder and re-ingest.

### get_chroma_client()

```python
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")
```
- `"local"` (default) = use persistent storage (files on disk).
- `"agentcore"` = use ephemeral storage (in-memory only), because AgentCore Runtime has no persistent filesystem.

```python
def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        if DEPLOYMENT_MODE == "agentcore":
            _client = chromadb.EphemeralClient()
        else:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client
```
- **Singleton pattern** again for the ChromaDB client.
- `EphemeralClient()`: All data lives in RAM. Fast, but lost when the process restarts.
- `PersistentClient(path=...)`: Saves data to disk in `chroma_db/`. Survives restarts.
- `os.makedirs(..., exist_ok=True)`: Creates the directory if it doesn't exist; doesn't error if it does.

### is_ingested()

```python
def is_ingested() -> bool:
    client = get_chroma_client()
    for col_name in COLLECTION_MAP.values():
        try:
            col = client.get_collection(col_name, embedding_function=_get_embedding_function())
            if col.count() == 0:
                return False
        except Exception:
            return False
    return True
```
- Checks if ALL four collections (hr_policies, it_policies, finance_policies, general_knowledge) exist and have documents.
- Used by `main.py` and `app.py` to skip ingestion if data is already loaded.
- `client.get_collection(...)`: Gets an existing collection. Throws an exception if not found.
- `col.count()`: Returns the number of documents in the collection.
- Returns `False` if any collection is missing or empty.

### get_collection()

```python
def get_collection(category: str) -> chromadb.Collection:
    client = get_chroma_client()
    collection_name = COLLECTION_MAP.get(category, category)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
    )
```
- Gets a ChromaDB collection by category name (e.g., `"hr"` -> `"hr_policies"`).
- `get_or_create_collection`: Returns existing collection or creates a new one.
- The `embedding_function` parameter tells ChromaDB how to convert text to vectors when inserting or querying.

### ingest_documents()

```python
def ingest_documents(force: bool = False) -> dict[str, int]:
    stats = {}
    file_category_map = {
        "hr_policies.json": "hr",
        "it_policies.json": "it",
        "finance_policies.json": "finance",
        "general_knowledge.json": "general",
    }
```
- `force=False`: If True, re-ingests even if data already exists. If False, skips populated collections.
- `file_category_map`: Maps each JSON filename to its category.

```python
    for filename, category in file_category_map.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            stats[category] = 0
            continue
```
- Loops through each data file. Skips if the file doesn't exist.

```python
        collection = get_collection(category)

        if not force and collection.count() > 0:
            stats[category] = collection.count()
            continue
```
- If `force` is False and the collection already has documents, skip it and record the existing count.

```python
        with open(filepath, "r") as f:
            documents = json.load(f)
```
- Opens the JSON file and parses it into a Python list of dictionaries.

```python
        ids = [doc["doc_id"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "department": doc["department"],
                "effective_date": doc.get("effective_date", ""),
                "last_updated": doc.get("last_updated", ""),
            }
            for doc in documents
        ]
```
- Prepares three parallel lists for ChromaDB:
  - `ids`: Unique identifier for each document (e.g., `"HR-001"`).
  - `texts`: The actual content text that gets embedded into vectors.
  - `metadatas`: Extra info stored alongside the vector (title, department, dates). This is returned in search results but NOT embedded.

```python
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        stats[category] = len(documents)
```
- `upsert`: Insert if new, update if the ID already exists. Safer than `add` which would throw an error on duplicates.
- ChromaDB automatically calls the embedding function on each text in `documents` to create vectors.

### query_collection()

```python
def query_collection(category: str, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    collection = get_collection(category)
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
```
- `query_texts=[query]`: ChromaDB embeds this query text into a vector, then finds the `n_results` closest document vectors using cosine similarity.
- `min(top_k, collection.count())`: Can't request more results than documents that exist.

```python
    documents = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            doc = {
                "doc_id": results["ids"][0][i],
                "content": results["documents"][0][i] if results["documents"] else "",
                "title": results["metadatas"][0][i].get("title", "") if results["metadatas"] else "",
                "department": results["metadatas"][0][i].get("department", "") if results["metadatas"] else "",
                "last_updated": results["metadatas"][0][i].get("last_updated", "") if results["metadatas"] else "",
                "relevance_score": 1 - results["distances"][0][i] if results["distances"] else 0,
            }
            documents.append(doc)
    return documents
```
- ChromaDB returns results in a nested structure: `results["ids"][0]` is the list of IDs for the first query (we only send one query).
- Flattens the results into a simple list of dictionaries.
- `1 - results["distances"][0][i]`: ChromaDB returns *distance* (lower = more similar). We convert to *similarity score* (higher = more similar) by subtracting from 1.

---

## 6. Utilities

### 6.1 LLM Factory - utils/llm.py

```python
from langchain_aws import ChatBedrock
from config import AWS_REGION, BEDROCK_MODEL_ID
```
- `ChatBedrock`: LangChain's wrapper for AWS Bedrock chat models. It handles the API calls, formatting, and response parsing.

```python
def get_llm(temperature: float = 0.1) -> ChatBedrock:
    return ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": temperature, "max_tokens": 2048},
    )
```
- **Factory function**: Creates a new LLM instance each time it's called.
- `temperature`: Controls randomness. 0.0 = deterministic (same input = same output), 1.0 = very creative/random. We use 0.0 for Router/Guardrail (need consistent classification) and 0.1 for Specialist (slight creativity in phrasing).
- `max_tokens`: Maximum length of the generated response (2048 tokens is roughly 1500 words).
- Every agent calls `get_llm()` to get its own LLM instance.

### 6.2 Citations - utils/citations.py

```python
from typing import Any
```
- `Any` is a type hint meaning "can be any type".

#### format_citations()

```python
def format_citations(documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    citations = []
    seen = set()
    for doc in documents:
        doc_id = doc.get("doc_id", "N/A")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        citations.append({
            "doc_id": doc_id,
            "title": doc.get("title", "Untitled"),
            "department": doc.get("department", "Unknown"),
            "last_updated": doc.get("last_updated", "N/A"),
        })
    return citations
```
- Takes a list of retrieved documents and extracts citation metadata.
- Uses a `set` called `seen` to **deduplicate** -- if the same document appears twice in results, it only shows once in citations.
- Returns a clean list of citation objects for the UI to display.

#### format_context_with_sources()

```python
def format_context_with_sources(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        doc_id = doc.get("doc_id", "N/A")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        context_parts.append(
            f"[Source {i} | {doc_id}: {title}]\n{content}"
        )
    return "\n\n".join(context_parts)
```
- Formats retrieved documents into a numbered text block that gets inserted into the LLM prompt.
- Example output:
  ```
  [Source 1 | HR-001: Annual Leave Policy]
  All full-time employees are entitled to 24 days of paid annual leave...

  [Source 2 | HR-002: Sick Leave Policy]
  Employees are entitled to 12 days of paid sick leave per year...
  ```
- The `[Source N | DOC_ID: Title]` format gives the LLM a consistent way to reference sources in its answer.

---

## 7. Agent State - agents/state.py

```python
from typing import Any, TypedDict
```
- `TypedDict`: A Python type that defines a dictionary with specific keys and their types. Unlike regular dicts, TypedDicts provide type checking.

```python
class AgentState(TypedDict):
    query: str                            # The current query (may be rewritten)
    original_query: str                   # The user's original query before rewriting
    chat_history: list[dict[str, str]]    # Previous conversation turns
    is_relevant: bool                     # Whether the guardrail approved the query
    category: str                         # Domain: "hr", "it", "finance", "general", "greeting", "blocked"
    retrieved_docs: list[dict[str, Any]]  # Documents found by the Retriever Agent
    context: str                          # Formatted source text sent to the LLM
    response: str                         # The final answer text
    citations: list[dict[str, str]]       # Citation metadata for the UI
    agent_trail: list[str]                # Log of what each agent did (for debugging)
    error: str | None                     # Error message from Quality Agent, or None if passed
```
- This is the **single source of truth** that flows through the entire pipeline.
- Every agent function takes an `AgentState` as input and returns an updated `AgentState`.
- LangGraph automatically merges the returned state into the existing state.
- Think of it as a form that each agent fills in their section of.

---

## 8. Agents (Detailed)

### 8.1 Rewriter Agent

**File**: `agents/rewriter_agent.py`
**Purpose**: Resolves follow-up questions into standalone queries using conversation history.
**When it matters**: User asks "What is the leave policy?" then follows up with "How many days can I carry forward?" -- the Rewriter converts the second question to "How many unused annual leave days can I carry forward to the next year?"

#### REWRITER_PROMPT

```python
REWRITER_PROMPT = """You are a query rewriter for an enterprise knowledge assistant.
Given the conversation history and the latest user query, rewrite the query into a fully self-contained standalone question.

If the latest query is already self-contained and does not reference prior conversation, return it as-is.
If it contains pronouns like "it", "that", "they", "this" or implicit references to earlier topics, resolve them using the conversation history.

Conversation history:
{chat_history}

Latest user query: {query}

Rewritten standalone query:"""
```
- `{chat_history}` and `{query}` are placeholders filled by `.format()`.
- The prompt instructs the LLM to resolve pronouns and implicit references.
- If the query is already standalone (e.g., "What is the password policy?"), the LLM returns it unchanged.

#### _format_history()

```python
def _format_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "(No prior conversation)"
    lines = []
    for turn in chat_history[-6:]:
        lines.append(f"User: {turn['query']}")
        response_preview = turn["response"][:200]
        lines.append(f"Assistant: {response_preview}...")
    return "\n".join(lines)
```
- Takes the last **6 turns** of conversation (to stay within token limits).
- Truncates each assistant response to **200 characters** (we only need enough context to understand references, not the full answer).
- Returns a formatted string like:
  ```
  User: What is the leave policy?
  Assistant: All full-time employees are entitled to 24 days of paid annual leave per calendar year...
  ```

#### rewrite_query()

```python
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
```
- If there's no chat history, skip the LLM call entirely (saves time and cost).
- `{**state, ...}`: Python spread operator -- creates a new dict with all existing state values plus the updated ones.

```python
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
```
- Uses `temperature=0.0` for deterministic rewriting.
- `response.content`: The text generated by the LLM.
- `.strip()`: Removes leading/trailing whitespace.
- Compares case-insensitively to detect if the query actually changed.
- Updates the `query` field (which all subsequent agents use) and records the original in `original_query`.

### 8.2 Guardrail Agent

**File**: `agents/guardrail_agent.py`
**Purpose**: Classifies queries into GREETING, RELEVANT, or IRRELEVANT. Blocks irrelevant queries from reaching the RAG pipeline. Responds to greetings with a capabilities message.

#### Three-Way Classification

```python
GUARDRAIL_PROMPT = """You are a guardrail agent for an enterprise knowledge assistant at TechNova Solutions.
Classify the employee query into exactly one of these three categories:

GREETING - General greetings, pleasantries, or conversational openers like "hello", "hi", "hey", "good morning", "how are you", "thanks", "thank you", "what can you do", "help", "who are you", etc.

RELEVANT - Questions related to the enterprise knowledge base:
...

IRRELEVANT - Topics completely outside the workplace:
...

Respond with ONLY one word: GREETING, RELEVANT, or IRRELEVANT
"""
```
- Gives the LLM clear examples for each category.
- "Respond with ONLY one word" ensures we get a parseable response.

#### check_relevance()

```python
def check_relevance(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    prompt = GUARDRAIL_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
```
- `.upper()`: Normalizes the response to uppercase for reliable comparison.

```python
    if verdict.startswith("GREETING"):
        return {
            **state,
            "is_relevant": False,
            "response": GREETING_MESSAGE,
            "citations": [],
            "category": "greeting",
            "agent_trail": state.get("agent_trail", []) + ["Guardrail -> GREETING"],
        }
```
- For greetings: Sets `is_relevant=False` (so the graph goes to END, not Router), and fills in the response directly.
- `GREETING_MESSAGE` introduces the bot and lists its capabilities.

```python
    if verdict.startswith("RELEVANT"):
        return {
            **state,
            "is_relevant": True,
            "agent_trail": state.get("agent_trail", []) + ["Guardrail -> RELEVANT"],
        }
```
- For relevant queries: Sets `is_relevant=True` so the graph continues to the Router Agent.
- Does NOT fill in `response` -- that's the Specialist Agent's job.

```python
    return {
        **state,
        "is_relevant": False,
        "response": REJECTION_MESSAGE,
        "citations": [],
        "category": "blocked",
        "agent_trail": state.get("agent_trail", []) + ["Guardrail -> IRRELEVANT (blocked)"],
    }
```
- For irrelevant queries (or any unexpected LLM output): Blocks with a polite rejection message.

### 8.3 Router Agent

**File**: `agents/router_agent.py`
**Purpose**: Classifies the query into one of four domains so we query the right ChromaDB collection.

```python
ROUTER_PROMPT = """You are a query routing agent for an enterprise knowledge assistant at TechNova Solutions.
Your job is to classify the employee query into exactly ONE of these categories:

- hr: Questions about leave policies, benefits, health insurance...
- it: Questions about passwords, VPN, software, laptops...
- finance: Questions about travel expenses, reimbursements, procurement...
- general: Questions about company overview, office locations...

Respond with ONLY the category name (hr, it, finance, or general). Nothing else.
"""
```
- The prompt lists keywords/topics for each category to guide the LLM.

```python
def route_query(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    prompt = ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)
    category = response.content.strip().lower().replace(".", "")
```
- `.lower()`: Normalizes to lowercase (LLM might return "HR" or "hr").
- `.replace(".", "")`: Removes any trailing period the LLM might add.

```python
    valid_categories = {"hr", "it", "finance", "general"}
    if category not in valid_categories:
        category = "general"
```
- **Safety fallback**: If the LLM returns something unexpected (e.g., "human resources"), default to `"general"`.

### 8.4 Retriever Agent

**File**: `agents/retriever_agent.py`
**Purpose**: Queries the ChromaDB vector store to find documents similar to the user's query.

```python
from vectorstore.store import query_collection

def retrieve_documents(state: AgentState) -> AgentState:
    category = state["category"]
    query = state["query"]

    docs = query_collection(category=category, query=query)
```
- Uses the `category` from the Router Agent to pick the right collection.
- Uses the `query` (potentially rewritten by the Rewriter Agent) for similarity search.
- `query_collection` handles the actual ChromaDB query (see Section 5).

```python
    return {
        **state,
        "retrieved_docs": docs,
        "agent_trail": state.get("agent_trail", []) + [
            f"Retriever ({category}) -> Found {len(docs)} documents"
        ],
    }
```
- Stores the retrieved documents in state and logs how many were found.

### 8.5 Specialist Agent

**File**: `agents/specialist_agent.py`
**Purpose**: Generates the final answer using the retrieved documents as context. Uses domain-specific prompts for each category.

#### Domain-Specific Prompts

```python
SPECIALIST_PROMPTS = {
    "hr": """You are the HR Policy Specialist at TechNova Solutions...""",
    "it": """You are the IT Support Specialist at TechNova Solutions...""",
    "finance": """You are the Finance Policy Specialist at TechNova Solutions...""",
    "general": """You are the General Knowledge Specialist at TechNova Solutions...""",
}
```
- Each prompt gives the LLM a specific persona and domain expertise.
- All prompts share the same structure:
  1. Role definition ("You are the HR Policy Specialist...")
  2. Grounding instruction ("Use ONLY the provided source documents")
  3. Citation format instruction ("[Source N | DOC_ID: Title]")
  4. The actual sources (inserted via `{context}`)
  5. The question (inserted via `{query}`)

#### generate_response()

```python
def generate_response(state: AgentState) -> AgentState:
    category = state["category"]
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {
            **state,
            "response": "I could not find any relevant documents...",
            "citations": [],
            "context": "",
            ...
        }
```
- Early return if no documents were retrieved. This prevents the LLM from hallucinating without context.

```python
    context = format_context_with_sources(docs)
    citations = format_citations(docs)
```
- `format_context_with_sources`: Creates the numbered source text block for the prompt.
- `format_citations`: Creates the citation metadata for the UI panel.

```python
    prompt_template = SPECIALIST_PROMPTS.get(category, SPECIALIST_PROMPTS["general"])
    prompt = prompt_template.format(context=context, query=query)
```
- `.get(category, SPECIALIST_PROMPTS["general"])`: Falls back to the general prompt if category doesn't match.
- `.format(...)`: Inserts the context and query into the prompt template.

```python
    llm = get_llm(temperature=0.1)
    response = llm.invoke(prompt)
```
- Uses `temperature=0.1` -- slightly more creative than the classifier agents (0.0) for more natural-sounding answers, but still mostly deterministic.

### 8.6 Quality Agent

**File**: `agents/quality_agent.py`
**Purpose**: Final validation step. Checks if the response actually answers the question and includes citations.

```python
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
"""
```

```python
def check_quality(state: AgentState) -> AgentState:
    ...
    passed = verdict.upper().startswith("PASS")

    return {
        **state,
        "agent_trail": state.get("agent_trail", []) + [
            f"Quality Check -> {'PASSED' if passed else verdict}"
        ],
        "error": None if passed else verdict,
    }
```
- If the check fails, the `error` field contains the reason (e.g., "FAIL - No citations included").
- Currently, the pipeline still returns the response even if quality fails -- the `error` field is informational. You could extend this to retry the Specialist Agent on failure.

---

## 9. LangGraph Workflow - graph/workflow.py

This file wires all 6 agents into a directed graph using LangGraph's `StateGraph`.

### Imports

```python
from langgraph.graph import END, StateGraph
```
- `StateGraph`: A graph where nodes are functions that modify a shared state.
- `END`: A special constant meaning "the pipeline is done, return the final state".

### Conditional Routing Function

```python
def _after_guardrail(state: AgentState) -> str:
    return "router" if state.get("is_relevant", False) else END
```
- This function is called after the Guardrail Agent finishes.
- If `is_relevant` is True, go to the `"router"` node.
- If `is_relevant` is False (greeting or blocked), go to `END` and return the state immediately.

### build_graph()

```python
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
```
- Creates a new graph that uses `AgentState` as its data structure.

```python
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("guardrail", check_relevance)
    workflow.add_node("router", route_query)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("specialist", generate_response)
    workflow.add_node("quality_check", check_quality)
```
- Registers each agent function as a named node in the graph.
- The string name (e.g., `"rewriter"`) is how we reference the node in edges.

```python
    workflow.set_entry_point("rewriter")
```
- The first node to execute when the graph is invoked.

```python
    workflow.add_edge("rewriter", "guardrail")
```
- **Unconditional edge**: After `rewriter` finishes, always go to `guardrail`.

```python
    workflow.add_conditional_edges("guardrail", _after_guardrail)
```
- **Conditional edge**: After `guardrail` finishes, call `_after_guardrail(state)` to decide where to go.
- If relevant -> `"router"`. If not -> `END`.

```python
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "specialist")
    workflow.add_edge("specialist", "quality_check")
    workflow.add_edge("quality_check", END)
```
- The remaining edges are all unconditional: Router -> Retriever -> Specialist -> Quality -> END.

```python
    return workflow.compile()
```
- `compile()` validates the graph (checks for unreachable nodes, cycles) and returns a callable `CompiledStateGraph`.

### run_query()

```python
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
```
- **The main entry point** for the entire pipeline. Everything starts here.
- Creates a fresh initial state with all fields set to empty/default values.
- `graph.invoke(initial_state)`: Runs the graph from start to END, passing state through each agent.
- Returns the final state containing the response, citations, and agent trail.

---

## 10. Entry Points

### 10.1 CLI - main.py

**Purpose**: Terminal-based testing interface. Good for quick testing without a browser.

```python
sys.path.insert(0, os.path.dirname(__file__))
```
- Adds the project root to Python's module search path so imports like `from graph.workflow import run_query` work correctly.

```python
    if not is_ingested():
        print("\nFirst run — ingesting documents into vector store...")
        stats = ingest_documents()
        ...
    else:
        print("\nVector store already populated. Skipping ingestion.")
```
- On first run, ingests the JSON documents into ChromaDB.
- On subsequent runs, skips ingestion because `chroma_db/` already exists (persistent mode).

```python
    chat_history = []

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = run_query(query, chat_history=chat_history)
        ...
        chat_history.append({"query": query, "response": result["response"]})
```
- Interactive loop: reads user input, runs the pipeline, prints the result.
- `chat_history` accumulates conversation turns for the Rewriter Agent.

### 10.2 Data Ingestion - ingest.py

```python
def main():
    print("Ingesting enterprise documents into vector store...")
    stats = ingest_documents(force=True)
    ...
```
- Standalone script to force re-ingest all documents. Useful after modifying the JSON data files.
- `force=True` overwrites existing documents even if the collection already has data.

### 10.3 Streamlit Local - app.py

**Purpose**: Web UI that runs the LangGraph pipeline locally (no cloud dependency except Bedrock for the LLM).

**Key sections:**

1. **Page Config & CSS**: Sets up the page title, layout, and custom styles for citation boxes, agent trail, and category badges.

2. **Sidebar**: Shows sample questions as clickable buttons, company branding, and a "Re-ingest Documents" button.

3. **Auto-ingestion**: On first Streamlit load, checks `is_ingested()` and ingests if needed. Uses `st.session_state["ingested"]` to avoid re-checking on every Streamlit rerun.

4. **Chat History Display**: Loops through `st.session_state["chat_history"]` and renders each past Q&A pair with citations and agent trail expandable sections.

5. **Input Handling**:
```python
prefill = st.session_state.pop("prefill_query", "")
user_query = st.chat_input("Ask a question about company policies...")
if prefill:
    user_query = prefill
```
- `st.chat_input`: Streamlit's built-in chat input widget.
- `prefill_query`: Set when a sidebar sample question button is clicked. Overrides the chat input.

6. **Query Execution**:
```python
history = [
    {"query": h["query"], "response": h["response"]}
    for h in st.session_state["chat_history"]
]
result = run_query(user_query, chat_history=history)
```
- Extracts minimal chat history (just query/response pairs) and passes to the pipeline.
- `run_query` executes the full LangGraph pipeline locally.

### 10.4 AgentCore Entrypoint - agent.py

**Purpose**: The file that AWS Bedrock AgentCore Runtime executes when the deployed agent receives a request.

```python
os.environ["DEPLOYMENT_MODE"] = "agentcore"
```
- Forces the vector store to use `EphemeralClient()` (in-memory) instead of `PersistentClient`.

```python
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp(debug=True)
```
- `BedrockAgentCoreApp`: The AgentCore SDK class that sets up an HTTP server on port 8080 with `/invocations` (POST) and `/ping` (GET) endpoints.

```python
_ingested = False

@app.entrypoint
def invoke(payload):
    global _ingested
    if not _ingested:
        logger.info("First invocation: ingesting documents into in-memory vector store...")
        ingest_documents(force=True)
        _ingested = True
```
- **Lazy initialization**: Documents are ingested on the first request, not during module import.
- This is important because AgentCore has a 30-second initialization timeout. Ingesting 27 documents via Bedrock Titan API calls takes time, so we do it during the first invocation instead.
- `_ingested` flag ensures ingestion only happens once per runtime session.

```python
    result = run_query(query, chat_history=chat_history)

    response = {
        "query": result.get("original_query", query),
        "rewritten_query": result.get("query", query),
        "category": result.get("category", ""),
        "response": result.get("response", ""),
        "citations": result.get("citations", []),
        "agent_trail": result.get("agent_trail", []),
        "is_relevant": result.get("is_relevant", True),
    }

    return json.dumps(response)
```
- Runs the same `run_query` pipeline as local mode.
- Returns a JSON string (AgentCore expects string or bytes, not a dict).

### 10.5 Streamlit Cloud - app_agentcore.py

**Purpose**: Streamlit UI that calls the deployed AgentCore endpoint instead of running the pipeline locally.

```python
def invoke_agentcore(query: str, chat_history: list) -> dict:
    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    payload = json.dumps({
        "prompt": query,
        "chat_history": chat_history,
    }).encode("utf-8")

    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_RUNTIME_ARN,
        runtimeSessionId=st.session_state.get("session_id", str(uuid.uuid4())),
        payload=payload,
        qualifier="DEFAULT",
    )
```
- `boto3.client("bedrock-agentcore")`: Creates a client for the AgentCore API.
- `invoke_agent_runtime`: Sends the payload to the deployed agent and gets back a streaming response.
- `runtimeSessionId`: A unique session ID. Same session ID reuses the same runtime instance (warm start).
- `qualifier="DEFAULT"`: Uses the default endpoint version.

```python
    raw = "".join(chunks)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed
```
- Handles **double-encoded JSON**: The agent returns `json.dumps(dict)` which is a string. AgentCore may wrap it again, resulting in `"\"{ ... }\""`. The `isinstance(parsed, str)` check detects this and parses again.

### 10.6 Deployment Script - deploy.py

**Purpose**: Automates the two-step deployment process by running `agentcore configure` then `agentcore deploy`.

```python
    run([
        "agentcore", "configure",
        "--entrypoint", "agent.py",
        "--name", args.name,
        "--requirements-file", "requirements-agentcore.txt",
        "--deployment-type", "direct_code_deploy",
        "--runtime", "PYTHON_3_13",
        "--region", args.region,
        "--disable-memory",
        "--disable-otel",
        "--non-interactive",
    ])
```
- `--entrypoint agent.py`: The Python file containing `@app.entrypoint`.
- `--requirements-file requirements-agentcore.txt`: Slim requirements without Streamlit/SentenceTransformers.
- `--deployment-type direct_code_deploy`: Zip-based deployment (no Docker needed).
- `--runtime PYTHON_3_13`: Python version on the runtime.
- `--disable-memory`: Don't provision AgentCore Memory (we handle memory ourselves via chat_history).
- `--disable-otel`: Don't add OpenTelemetry tracing.
- `--non-interactive`: Don't prompt for input; use defaults.

---

## 11. Data Files

Located in `data/`. Each file is a JSON array of policy documents with this structure:

```json
{
    "doc_id": "HR-001",
    "title": "Annual Leave Policy",
    "department": "HR",
    "content": "All full-time employees are entitled to 24 days of paid annual leave...",
    "effective_date": "2024-01-01",
    "last_updated": "2024-06-15"
}
```

| Field | Purpose |
|-------|---------|
| `doc_id` | Unique identifier, used as ChromaDB document ID and in citations |
| `title` | Human-readable name, shown in citations |
| `department` | Category label, shown in citations |
| `content` | The actual policy text that gets **embedded into vectors** and used as LLM context |
| `effective_date` | When the policy became active (metadata only) |
| `last_updated` | When the policy was last revised (shown in citations) |

**Document counts:**
- `hr_policies.json`: 8 documents (HR-001 to HR-008)
- `it_policies.json`: 7 documents (IT-001 to IT-007)
- `finance_policies.json`: 6 documents (FIN-001 to FIN-006)
- `general_knowledge.json`: 6 documents (GEN-001 to GEN-006)
- **Total: 27 documents**

---

## 12. Key Concepts for Beginners

### What is an Embedding?
An embedding converts text into a list of numbers (vector) that captures its meaning. Similar texts produce similar vectors. For example:
- "annual leave policy" -> [0.12, -0.45, 0.78, ...]
- "vacation days rules" -> [0.11, -0.44, 0.77, ...]  (very similar!)
- "password requirements" -> [-0.33, 0.67, -0.12, ...] (very different!)

ChromaDB stores these vectors and can quickly find the most similar ones to any query vector.

### What is a Vector Database?
A specialized database optimized for storing and searching embedding vectors. ChromaDB is one of many (others: Pinecone, Weaviate, FAISS). Instead of SQL queries like `WHERE department = 'HR'`, you do similarity searches like "find the 4 documents most similar to this query".

### What is LangGraph?
LangGraph (by LangChain) lets you define AI workflows as directed graphs. Each node is a function, and edges define the execution order. It supports conditional edges (branching based on state), which is how our Guardrail Agent can short-circuit the pipeline for irrelevant queries.

### What is AWS Bedrock?
A managed service where you can call various LLMs (Claude, Llama, Titan, etc.) via API without managing GPU servers. You pay per token (input + output). Bedrock also hosts embedding models like Titan Embed.

### What is AgentCore Runtime?
AWS's serverless hosting for AI agents. You upload your code as a zip, and AWS handles scaling, infrastructure, and networking. Your agent gets an HTTP endpoint that accepts POST requests and returns responses.

### What is a TypedDict?
A Python dictionary with predefined keys and types. Unlike a regular `dict`, it provides IDE autocomplete and type checking. Our `AgentState` is a TypedDict so every agent knows exactly what fields are available.

### What does `{**state, "key": "value"}` mean?
This creates a new dictionary by copying all key-value pairs from `state` and then adding/overriding specific keys. It's the standard pattern for immutable state updates in Python. Each agent returns a new state dict rather than mutating the input.

### What is a Singleton Pattern?
A design pattern that ensures only ONE instance of something exists. In `store.py`, `_client` and `_embedding_fn` are singletons -- the first call creates them, and subsequent calls return the same instance. This avoids the cost of creating new database connections or downloading models repeatedly.

### Why Two Requirements Files?
- `requirements.txt`: Everything needed for local development (includes Streamlit, SentenceTransformers, AgentCore toolkit).
- `requirements-agentcore.txt`: Only what the deployed agent needs (no Streamlit, no SentenceTransformers). Keeping this slim ensures the deployment zip stays under AgentCore's 250MB limit.
