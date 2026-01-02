import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def vectordb_dir(cfg: Dict[str, Any], embed_model: str) -> str:
    name = f"chroma_db_ollama_{_sanitize(embed_model)}"
    p = os.path.join(str(cfg["VECTOR_DB_ROOT"]), name)
    os.makedirs(p, exist_ok=True)
    return p


def fingerprints_path(cfg: Dict[str, Any], embed_model: str) -> str:
    return os.path.join(vectordb_dir(cfg, embed_model), "fingerprints.json")


def load_fingerprints(cfg: Dict[str, Any], embed_model: str) -> Dict[str, Dict[str, str]]:
    fp = fingerprints_path(cfg, embed_model)
    if not os.path.exists(fp):
        return {"pdf": {}, "news": {}}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("pdf", {})
        data.setdefault("news", {})
        return data
    except Exception:
        return {"pdf": {}, "news": {}}


def save_fingerprints(cfg: Dict[str, Any], embed_model: str, data: Dict[str, Dict[str, str]]) -> None:
    fp = fingerprints_path(cfg, embed_model)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_fingerprint(path: str) -> str:
    st_ = os.stat(path)
    return f"{st_.st_size}:{int(st_.st_mtime)}"


@st.cache_resource
def get_embeddings(embed_model: str):
    return OllamaEmbeddings(model=embed_model)


@st.cache_resource
def get_vectorstore_cached(persist_directory: str, embed_model: str) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(embed_model),
    )


def get_vectorstore(cfg: Dict[str, Any], embed_model: str, logger: logging.Logger) -> Chroma:
    if str(cfg.get("VECTOR_DB", "chroma")).lower() != "chroma":
        logger.info(f"VECTOR_DB='{cfg.get('VECTOR_DB')}' not supported yet. Fallback to 'chroma'.")
    pdir = vectordb_dir(cfg, embed_model)
    return get_vectorstore_cached(pdir, embed_model)


def split_docs(docs, cfg: Dict[str, Any]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg["CHUNK_SIZE"]),
        chunk_overlap=int(cfg["CHUNK_OVERLAP"]),
    )
    return splitter.split_documents(docs)


def delete_by_source(vs: Chroma, source_path: str, logger: logging.Logger) -> None:
    try:
        vs._collection.delete(where={"source": source_path})  # type: ignore[attr-defined]
    except Exception as e:
        logger.info(f"[VectorDB] delete(where=source) skipped/failed: {e}")


def _log_vectordb_status(vs: Chroma, logger: logging.Logger, prefix: str, added_chunks: int, reason: str) -> None:
    try:
        total = vs._collection.count()  # type: ignore[attr-defined]
    except Exception:
        total = -1
    msg = f"[{prefix}] VectorDB updated: added_chunks={added_chunks}, total_vectors={total}, reason={reason}"
    logger.info(msg)
    try:
        st.sidebar.info(msg)
    except Exception:
        pass


def add_documents(
    vs: Chroma,
    docs,
    cfg: Dict[str, Any],
    logger: logging.Logger,
    prefix: str,
    reason: str
) -> int:
    if not docs:
        return 0
    chunks = split_docs(docs, cfg)
    vs.add_documents(chunks)
    _log_vectordb_status(vs, logger, prefix=prefix, added_chunks=len(chunks), reason=reason)
    return len(chunks)


def list_pdfs(cfg: Dict[str, Any]) -> List[str]:
    data_dir = str(cfg["DATA_DIR"])
    if not os.path.exists(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, n)
        for n in os.listdir(data_dir)
        if n.lower().endswith(".pdf")
    )


def save_uploaded_pdf(uploaded, cfg: Dict[str, Any], logger: logging.Logger) -> Optional[str]:
    if uploaded is None:
        return None
    name = uploaded.name
    if not name.lower().endswith(".pdf"):
        try:
            st.sidebar.error("PDF 파일만 업로드할 수 있습니다.")
        except Exception:
            pass
        return None

    os.makedirs(str(cfg["DATA_DIR"]), exist_ok=True)
    dst = os.path.join(str(cfg["DATA_DIR"]), name)

    data = uploaded.getvalue()
    with open(dst, "wb") as f:
        f.write(data)

    try:
        st.sidebar.success(f"업로드 완료: {name} → {cfg['DATA_DIR']}")
    except Exception:
        pass

    logger.info(f"[UPLOAD] pdf saved: {dst} ({len(data)} bytes)")
    return dst


def sync_pdf_dir(vs: Chroma, cfg: Dict[str, Any], embed_model: str, logger: logging.Logger) -> int:
    fps = load_fingerprints(cfg, embed_model)
    before = dict(fps.get("pdf", {}))

    total_added = 0
    for p in list_pdfs(cfg):
        fp = file_fingerprint(p)
        if before.get(p) == fp:
            continue

        delete_by_source(vs, p, logger)

        loader = PyPDFLoader(p)
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["source_type"] = "pdf"

        total_added += add_documents(
            vs, docs, cfg, logger,
            prefix="PDF",
            reason=f"changed_or_new: {os.path.basename(p)}"
        )
        fps["pdf"][p] = fp

    save_fingerprints(cfg, embed_model, fps)
    return total_added

def log_vectordb_state(vs, logger):
    """Log current vector DB state after writes."""
    try:
        coll = getattr(vs, "_collection", None)
        if coll is not None:
            cnt = coll.count()
            logger.info(f"[VDB] collection_count={cnt}")
            return
    except Exception as e:
        logger.info(f"[VDB] state read failed: {e}")
    logger.info("[VDB] state=unknown")