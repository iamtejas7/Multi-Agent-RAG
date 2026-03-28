"""CLI entry point for testing the multi-agent RAG pipeline without Streamlit."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vectorstore.store import ingest_documents, is_ingested
from graph.workflow import run_query


def main():
    print("=" * 70)
    print("  TechNova Enterprise Knowledge Assistant (CLI)")
    print("=" * 70)

    if not is_ingested():
        print("\nFirst run — ingesting documents into vector store...")
        stats = ingest_documents()
        for cat, count in stats.items():
            print(f"  [{cat}] {count} documents")
    else:
        print("\nVector store already populated. Skipping ingestion.")

    print("\nReady! Type your question (or 'quit' to exit).\n")

    chat_history = []

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nProcessing...\n")
        result = run_query(query, chat_history=chat_history)

        print(f"Category: {result['category'].upper()}")
        print(f"\nAnswer:\n{result['response']}\n")

        if result.get("citations"):
            print("Citations:")
            for c in result["citations"]:
                print(f"  - [{c['doc_id']}] {c['title']} ({c['department']}) | Updated: {c['last_updated']}")

        if result.get("agent_trail"):
            print("\nAgent Trail:")
            for step in result["agent_trail"]:
                print(f"  > {step}")

        chat_history.append({"query": query, "response": result["response"]})

        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
