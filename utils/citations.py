from typing import Any


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
