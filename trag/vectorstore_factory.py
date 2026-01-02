# trag/vectorstore_factory.py
from __future__ import annotations

import os
import threading
from typing import Optional, Dict, Any, List

from config import (
    VECTORDB_PROVIDER,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)

# Embeddings / VectorStore (LangChain 최신/구버전 호환)
def _load_ollama_embeddings():
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception:
        # 구버전 fallback
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

def _load_chroma(embedding_function):
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    try:
        # 최신 분리 패키지
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function,
        )
    except Exception:
        from langchain_community.vectorstores import Chroma
        return Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function,
        )

class VectorStoreManager:
    """
    Thread-safe wrapper around VectorStore.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._emb = _load_ollama_embeddings()

        if VECTORDB_PROVIDER.lower() == "chroma":
            self._vs = _load_chroma(self._emb)
        else:
            raise ValueError(f"Unsupported VECTORDB_PROVIDER={VECTORDB_PROVIDER}")

    @property
    def lock(self):
        return self._lock

    def add_documents(self, docs) -> None:
        with self._lock:
            self._vs.add_documents(docs)
            # Chroma persist는 내부적으로 자동 persist 되는 경우도 있으나 명시적으로 호출 시도
            try:
                self._vs.persist()
            except Exception:
                pass

    def similarity_search(self, query: str, k: int):
        with self._lock:
            return self._vs.similarity_search(query, k=k)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            info: Dict[str, Any] = {}
            # Chroma collection count
            try:
                # langchain_chroma / community 모두 _collection 접근이 되는 경우가 많음
                count = self._vs._collection.count()
                info["count"] = int(count)
            except Exception:
                info["count"] = None
            info["persist_directory"] = CHROMA_PERSIST_DIR
            info["collection_name"] = CHROMA_COLLECTION_NAME
            return info

# singleton
_singleton: Optional[VectorStoreManager] = None

def get_vectorstore() -> VectorStoreManager:
    global _singleton
    if _singleton is None:
        _singleton = VectorStoreManager()
    return _singleton