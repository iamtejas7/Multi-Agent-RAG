import json
import os

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_MAP,
    DATA_DIR,
    HF_EMBEDDING_MODEL,
    TOP_K_RESULTS,
    USE_BEDROCK_EMBEDDINGS,
)

_client: chromadb.ClientAPI | None = None
_embedding_fn = None


def _get_embedding_function():
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    if USE_BEDROCK_EMBEDDINGS:
        from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction
        from config import AWS_REGION, BEDROCK_EMBEDDING_MODEL_ID

        _embedding_fn = AmazonBedrockEmbeddingFunction(
            model_name=BEDROCK_EMBEDDING_MODEL_ID,
            region_name=AWS_REGION,
        )
    else:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=HF_EMBEDDING_MODEL
        )
    return _embedding_fn


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


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


def get_collection(category: str) -> chromadb.Collection:
    client = get_chroma_client()
    collection_name = COLLECTION_MAP.get(category, category)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
    )


def ingest_documents(force: bool = False) -> dict[str, int]:
    stats = {}
    file_category_map = {
        "hr_policies.json": "hr",
        "it_policies.json": "it",
        "finance_policies.json": "finance",
        "general_knowledge.json": "general",
    }

    for filename, category in file_category_map.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            stats[category] = 0
            continue

        collection = get_collection(category)

        if not force and collection.count() > 0:
            stats[category] = collection.count()
            continue

        with open(filepath, "r") as f:
            documents = json.load(f)

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

        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        stats[category] = len(documents)

    return stats


def query_collection(category: str, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    collection = get_collection(category)
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))

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
