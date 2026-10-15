from typing import Any

import chromadb

from core.config import CHROMA_DIR


def get_chroma_client() -> Any:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "idamp_memory") -> Any:
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, embedding_function=None)


def store_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        collection = get_collection()
        record: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [text],
            # Chroma requires a vector when its embedding function is disabled.
            "embeddings": [[0.0]],
        }
        if metadata:
            record["metadatas"] = [metadata]
        collection.upsert(**record)
    except Exception:
        pass


def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        get_options: dict[str, Any] = {
            "limit": n_results,
            "include": ["documents", "metadatas"],
        }
        if query_text:
            get_options["where_document"] = {"$contains": query_text}
        results = collection.get(**get_options)

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        return [
            {
                "id": doc_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else None,
            }
            for index, doc_id in enumerate(ids)
        ]
    except Exception:
        return []
