"""Script to ingest dummy data into ChromaDB vector store."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vectorstore.store import ingest_documents


def main():
    print("Ingesting enterprise documents into vector store...")
    stats = ingest_documents(force=True)
    for category, count in stats.items():
        print(f"  {category}: {count} documents ingested")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
