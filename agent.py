"""
AWS Bedrock AgentCore Runtime entrypoint for the Enterprise Knowledge Assistant.
Deployed via direct code deployment (zip).
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))
os.environ["DEPLOYMENT_MODE"] = "agentcore"

from bedrock_agentcore import BedrockAgentCoreApp
from graph.workflow import run_query
from vectorstore.store import ingest_documents, is_ingested

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp(debug=True)

_ingested = False


@app.entrypoint
def invoke(payload):
    """AgentCore invocation handler."""
    global _ingested
    if not _ingested:
        logger.info("First invocation: ingesting documents into in-memory vector store...")
        ingest_documents(force=True)
        _ingested = True
        logger.info("Document ingestion complete.")

    query = payload.get("prompt", payload.get("query", ""))
    chat_history = payload.get("chat_history", [])

    if not query:
        return json.dumps({
            "error": "No query provided. Send {\"prompt\": \"your question\"}"
        })

    logger.info("Query: %s", query)
    logger.info("Chat history length: %d", len(chat_history))

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

    logger.info("Category: %s | Relevant: %s", response["category"], response["is_relevant"])
    return json.dumps(response)


if __name__ == "__main__":
    app.run()
