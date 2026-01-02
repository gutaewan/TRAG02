import os
import re
import json
import time
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import requests
import feedparser

# Optional: full article extraction
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None  # type: ignore

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain 버전에 따라 import 경로가 달라질 수 있어 호환 처리
#try:
#    from langchain.chains import create_history_aware_retriever
#except Exception:
#    try:
#        from langchain.chains.history_aware_retriever import create_history_aware_retriever
#    except Exception:#
#        create_history_aware_retriever = None


# =========================
# 0) Config (./config/config.py)
#    - 단일 설정 소스: ./config/config.py
#    - BaseRAG_v3.py 내부 클래스로 설정하지 않음
# =========================

CONFIG_DIR = "./config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.py")

DEFAULTS: Dict[str, Any] = {
    # Directories
    "DATA_DIR": "./data",
    "NEWS_DIR": "./news_texts",
    "LOG_DIR": "./logs",
    "VECTOR_DB_ROOT": "./vector_db",

    # Vector DB
    "VECTOR_DB": "chroma",

    # Models
    "LLM_MODEL": "llama3.2",
    "EMBED_MODEL": "qwen3-embedding",

    # Chunking
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,

    # Periodic intervals
    "PDF_SYNC_INTERVAL_SEC": 600,
    "NEWS_CRAWL_INTERVAL_SEC": 600,

    # News
    "NEWS_KEYWORDS": ["소프트웨어 공학", "AI 안전", "자동차 기능안전", "SDV"],
    "NEWS_MAX_ITEMS_PER_KEYWORD": 10,
    "NEWS_TIMEOUT_SEC": 20,

    # Auto refresh
    "AUTO_REFRESH_ENABLED": True,
    "AUTO_REFRESH_TICK_SEC": 30,

    # Safety: reset stuck "generating" state if a run is interrupted
    "GENERATION_STALE_SEC": 180,
}

TEMPLATE_CONFIG = """# TRAG01 설정 파일\n# 여기 값을 수정하면 BaseRAG_v3.py가 그대로 반영합니다.\n\n# Directories\nDATA_DIR = \"./data\"\nNEWS_DIR = \"./news_texts\"\nLOG_DIR = \"./logs\"\nVECTOR_DB_ROOT = \"./vector_db\"\n\n# Vector DB\nVECTOR_DB = \"chroma\"\n\n# Models\nLLM_MODEL = \"llama3.2\"\nEMBED_MODEL = \"qwen3-embedding\"\n\n# Chunking\nCHUNK_SIZE = 1000\nCHUNK_OVERLAP = 200\n\n# Periodic intervals (seconds)\nPDF_SYNC_INTERVAL_SEC = 600\nNEWS_CRAWL_INTERVAL_SEC = 600\n\n# News\nNEWS_KEYWORDS = [\"소프트웨어 공학\", \"AI 안전\", \"자동차 기능안전\", \"SDV\"]\nNEWS_MAX_ITEMS_PER_KEYWORD = 10\nNEWS_TIMEOUT_SEC = 20\n\n# Auto refresh\nAUTO_REFRESH_ENABLED = True\nAUTO_REFRESH_TICK_SEC = 30\n\n# Safety: reset stuck \"generating\" state if a run is interrupted\nGENERATION_STALE_SEC = 180\n"""


def ensure_config_file() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(TEMPLATE_CONFIG)


def load_config() -> Dict[str, Any]:
    ensure_config_file()
    cfg = dict(DEFAULTS)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("trag_user_config", CONFIG_FILE)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            for k in list(DEFAULTS.keys()):
                if hasattr(mod, k):
                    cfg[k] = getattr(mod, k)
    except Exception as e:
        print(f"[WARN] Failed to load config.py: {e}")
    # normalize types
    if isinstance(cfg.get("NEWS_KEYWORDS"), tuple):
        cfg["NEWS_KEYWORDS"] = list(cfg["NEWS_KEYWORDS"])
    return cfg


CFG: Dict[str, Any] = load_config()

# Ensure dirs
os.makedirs(str(CFG["DATA_DIR"]), exist_ok=True)
os.makedirs(str(CFG["NEWS_DIR"]), exist_ok=True)
os.makedirs(str(CFG["LOG_DIR"]), exist_ok=True)
os.makedirs(str(CFG["VECTOR_DB_ROOT"]), exist_ok=True)


# =========================
# =========================
# 0.5) Persistent storage for chats (survive browser refresh)
# =========================

CHAT_STORE_DIR = os.path.join(str(CFG["VECTOR_DB_ROOT"]), "chat_store")
os.makedirs(CHAT_STORE_DIR, exist_ok=True)

def _chat_registry_path() -> str:
    return os.path.join(CHAT_STORE_DIR, "chat_registry.json")

def _chat_messages_path(session_id: str) -> str:
    return os.path.join(CHAT_STORE_DIR, f"chat_{session_id}.json")

# =========================
# 1) Logging
# =========================
def _load_chat_registry_from_disk() -> Dict[str, str]:
    p = _chat_registry_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # ensure string->string
            out: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
            return out
        return {}
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to load registry: {e}")
        return {}


def _save_chat_registry_to_disk(reg: Dict[str, str]) -> None:
    try:
        os.makedirs(CHAT_STORE_DIR, exist_ok=True)
        tmp = _chat_registry_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _chat_registry_path())
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to save registry: {e}")


def _serialize_messages(msgs) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in msgs or []:
        try:
            # StreamlitChatMessageHistory uses dict-like messages with .type/.content
            t = getattr(m, "type", None) or m.get("type")
            c = getattr(m, "content", None) or m.get("content")
            if t and c is not None:
                out.append({"type": str(t), "content": str(c)})
        except Exception:
            continue
    return out


def _load_messages_from_disk(session_id: str) -> List[Dict[str, str]]:
    p = _chat_messages_path(session_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            out = []
            for it in data:
                if isinstance(it, dict) and "type" in it and "content" in it:
                    out.append({"type": str(it["type"]), "content": str(it["content"])})
            return out
        return []
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to load messages for {session_id}: {e}")
        return []


def _save_messages_to_disk(session_id: str, msgs) -> None:
    try:
        os.makedirs(CHAT_STORE_DIR, exist_ok=True)
        tmp = _chat_messages_path(session_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_serialize_messages(msgs), f, ensure_ascii=False, indent=2)
        os.replace(tmp, _chat_messages_path(session_id))
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to save messages for {session_id}: {e}")


def _hydrate_history_from_disk(session_id: str, history: StreamlitChatMessageHistory) -> None:
    """If Streamlit session is fresh but disk has messages, load them into StreamlitChatMessageHistory."""
    try:
        if history.messages:
            return
        disk_msgs = _load_messages_from_disk(session_id)
        if not disk_msgs:
            return
        for m in disk_msgs:
            if m.get("type") in ("human", "user"):
                history.add_user_message(m.get("content", ""))
            else:
                history.add_ai_message(m.get("content", ""))
        logger.info(f"[CHAT_STORE] hydrated {len(disk_msgs)} messages into session {session_id}")
    except Exception as e:
        logger.info(f"[CHAT_STORE] hydrate failed for {session_id}: {e}")

os.makedirs(str(CFG["LOG_DIR"]), exist_ok=True)
LOG_PATH = os.path.join(str(CFG["LOG_DIR"]), "app.log")

logger = logging.getLogger("trag")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _log_vectordb_status(vs: Chroma, prefix: str, added_chunks: int, reason: str) -> None:
    """(5)(14) 벡터DB 상태 로그"""
    try:
        total = vs._collection.count()  # type: ignore[attr-defined]
    except Exception:
        total = -1

    msg = f"[{prefix}] VectorDB updated: added_chunks={added_chunks}, total_vectors={total}, reason={reason}"
    logger.info(msg)
    # sidebar에도 보여주면 디버깅 편함
    try:
        st.sidebar.info(msg)
    except Exception:
        pass


# =========================
# 2) Vector DB path (21) + fingerprints
# =========================


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace(":", "_").replace(" ", "_")
    return s


def vectordb_dir(embed_model: str) -> str:
    # (21) ./vector_db/chroma_db_ollama_{임베딩모델}
    name = f"chroma_db_ollama_{_sanitize(embed_model)}"
    p = os.path.join(CFG["VECTOR_DB_ROOT"], name)
    os.makedirs(p, exist_ok=True)
    return p


def fingerprints_path(embed_model: str) -> str:
    return os.path.join(vectordb_dir(embed_model), "fingerprints.json")


def load_fingerprints(embed_model: str) -> Dict[str, Dict[str, str]]:
    fp = fingerprints_path(embed_model)
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


def save_fingerprints(embed_model: str, data: Dict[str, Dict[str, str]]) -> None:
    fp = fingerprints_path(embed_model)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def file_fingerprint(path: str) -> str:
    st_ = os.stat(path)
    return f"{st_.st_size}:{int(st_.st_mtime)}"


# =========================
# 3) Vectorstore + embeddings + chunking
# =========================

@st.cache_resource
def get_embeddings(embed_model: str):
    return OllamaEmbeddings(model=embed_model)


@st.cache_resource
def get_vectorstore(embed_model: str) -> Chroma:
    # (6) vector db type configurable, but this implementation is chroma only
    if str(CFG["VECTOR_DB"]).lower() != "chroma":
        logger.info(f"VECTOR_DB='{CFG['VECTOR_DB']}' not supported yet. Fallback to 'chroma'.")
    return Chroma(
        persist_directory=vectordb_dir(embed_model),
        embedding_function=get_embeddings(embed_model),
    )


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(CFG["CHUNK_SIZE"]),
        chunk_overlap=int(CFG["CHUNK_OVERLAP"]),
    )
    return splitter.split_documents(docs)


def delete_by_source(vs: Chroma, source_path: str) -> None:
    """같은 source를 가진 기존 문서를 삭제하고 재임베딩 (변경 반영)"""
    try:
        vs._collection.delete(where={"source": source_path})  # type: ignore[attr-defined]
    except Exception as e:
        logger.info(f"[VectorDB] delete(where=source) skipped/failed: {e}")


def add_documents(vs: Chroma, docs, prefix: str, reason: str) -> int:
    if not docs:
        return 0
    chunks = split_docs(docs)
    vs.add_documents(chunks)
    _log_vectordb_status(vs, prefix=prefix, added_chunks=len(chunks), reason=reason)
    return len(chunks)


# =========================
# 4) PDF upload + sync
# =========================


def list_pdfs() -> List[str]:
    return sorted(
        os.path.join(CFG["DATA_DIR"], n)
        for n in os.listdir(CFG["DATA_DIR"])
        if n.lower().endswith(".pdf")
    )


def save_uploaded_pdf(uploaded) -> Optional[str]:
    """(1) 업로드된 PDF를 ./data에 저장"""
    if uploaded is None:
        return None
    name = uploaded.name
    if not name.lower().endswith(".pdf"):
        st.sidebar.error("PDF 파일만 업로드할 수 있습니다.")
        return None

    os.makedirs(CFG["DATA_DIR"], exist_ok=True)
    dst = os.path.join(CFG["DATA_DIR"], name)

    data = uploaded.getvalue()
    with open(dst, "wb") as f:
        f.write(data)

    st.sidebar.success(f"업로드 완료: {name} → {CFG['DATA_DIR']}")
    logger.info(f"[UPLOAD] pdf saved: {dst} ({len(data)} bytes)")
    return dst


def sync_pdf_dir(vs: Chroma, embed_model: str) -> int:
    """(3) 변경된 PDF만 임베딩"""
    fps = load_fingerprints(embed_model)
    before = dict(fps.get("pdf", {}))

    total_added = 0
    for p in list_pdfs():
        fp = file_fingerprint(p)
        if before.get(p) == fp:
            continue

        # 변경/신규이면 기존 source 삭제 후 재추가
        delete_by_source(vs, p)

        loader = PyPDFLoader(p)
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["source_type"] = "pdf"

        total_added += add_documents(vs, docs, prefix="PDF", reason=f"changed_or_new: {os.path.basename(p)}")
        fps["pdf"][p] = fp

    save_fingerprints(embed_model, fps)
    return total_added


# =========================
# 5) News crawl (Google News RSS) + save (dedup) + sync
# =========================

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def google_news_rss_url(query: str) -> str:
    from urllib.parse import quote_plus

    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def strip_html(text: str) -> str:
    """HTML 제거(간단)"""
    s = text or ""
    if BeautifulSoup is not None:
        try:
            s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
        except Exception:
            s = re.sub(r"<[^>]+>", " ", s)
    else:
        s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_url(u: str) -> str:
    return (u or "").strip()


def representative_sentence(title: str, summary_html: str) -> str:
    # 간단/견고한 1문장 추출
    summary = strip_html(summary_html)
    if not summary:
        return (title or "").strip()

    m = re.search(r"[.!?](?:\s+|$)|다\.(?:\s+|$)", summary)
    first = summary[: m.end()].strip() if m else summary.strip()

    if len(first) < 25:
        return f"{(title or '').strip()} - {first}".strip(" -")
    return first


def stable_news_id(title: str, link: str) -> str:
    # (13) 중복 방지 key
    base = ((title or "").strip().lower() + "|" + (link or "").strip())
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def fetch_article_fulltext(url: str, timeout_sec: int) -> str:
    """뉴스 링크에서 가능한 한 '전체 본문'을 추출합니다.

    우선순위:
    1) trafilatura (가능하면)로 본문 추출
    2) BeautifulSoup로 <article> 또는 본문 후보 텍스트 수집
    3) 실패 시 빈 문자열 반환

    주의: 사이트 정책/구조/차단에 따라 전체 본문이 제한될 수 있어, 실패하면 RSS summary로 폴백합니다.
    """
    url = normalize_url(url)
    if not url:
        return ""

    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_sec, allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logger.info(f"[NEWS] article fetch failed: {url} ({e})")
        return ""

    # 1) trafilatura
    if trafilatura is not None:
        try:
            downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
            if downloaded:
                text = re.sub(r"\s+", " ", downloaded).strip()
                # 너무 짧으면 실패로 간주
                if len(text) >= 400:
                    return text
        except Exception as e:
            logger.info(f"[NEWS] trafilatura extract failed: {url} ({e})")

    # 2) BeautifulSoup heuristic
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            # remove scripts/styles
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            # Prefer <article>
            article = soup.find("article")
            if article is not None:
                text = article.get_text(" ", strip=True)
            else:
                # Heuristic: collect paragraphs from common containers
                candidates = []
                for sel in [
                    "main",
                    "#content",
                    ".content",
                    ".article",
                    ".news",
                    ".post",
                    "body",
                ]:
                    node = soup.select_one(sel)
                    if node is None:
                        continue
                    ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
                    joined = " ".join([p for p in ps if p])
                    if len(joined) > len(" ".join(candidates)):
                        candidates = ps
                text = " ".join([p for p in candidates if p])

            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 400:
                return text
        except Exception as e:
            logger.info(f"[NEWS] soup extract failed: {url} ({e})")

    return ""


def fetch_google_news(keyword: str) -> List[Dict[str, Any]]:
    url = google_news_rss_url(keyword)
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=int(CFG["NEWS_TIMEOUT_SEC"]))
    r.raise_for_status()
    feed = feedparser.parse(r.text)

    out: List[Dict[str, Any]] = []
    for e in feed.entries[: int(CFG["NEWS_MAX_ITEMS_PER_KEYWORD"])]:
        title = getattr(e, "title", "").strip()
        link = normalize_url(getattr(e, "link", "").strip())
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        summary = getattr(e, "summary", "") or getattr(e, "description", "")

        fulltext = ""
        # 가능한 경우 링크 본문을 가져와 full text 추출
        if link:
            fulltext = fetch_article_fulltext(link, timeout_sec=int(CFG["NEWS_TIMEOUT_SEC"]))

        out.append(
            {
                "keyword": keyword,
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
                "fulltext": fulltext,
            }
        )
    return out


def ensure_news_dir():
    os.makedirs(CFG["NEWS_DIR"], exist_ok=True)


def news_index_path() -> str:
    return os.path.join(CFG["NEWS_DIR"], "news_index.json")


def load_news_index() -> Dict[str, str]:
    # news_id -> filename
    p = news_index_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_news_index(idx: Dict[str, str]) -> None:
    with open(news_index_path(), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def save_news_items(items: List[Dict[str, Any]]) -> List[str]:
    """(10)(13) 뉴스 txt 생성 및 중복 방지"""
    ensure_news_dir()
    idx = load_news_index()

    created: List[str] = []
    for e in items:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        published = (e.get("published") or "").strip()
        keyword = (e.get("keyword") or "").strip()
        summary = (e.get("summary") or "").strip()
        fulltext = (e.get("fulltext") or "").strip()

        nid = stable_news_id(title, link)
        if nid in idx:
            # 이미 저장된 뉴스
            continue

        # 파일명은 임의로 생성(요구사항 10)하지만, index로 중복을 관리
        fname = f"news_{uuid.uuid4().hex}.txt"
        path = os.path.join(CFG["NEWS_DIR"], fname)

        rep = representative_sentence(title, fulltext if fulltext else summary)

        body = strip_html(fulltext) if fulltext else strip_html(summary)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# title: {title}\n")
            f.write(f"# url: {link}\n")
            f.write(f"# published: {published}\n")
            f.write(f"# keyword: {keyword}\n")
            f.write(f"# representative: {rep}\n")
            f.write(f"# has_fulltext: {bool(fulltext)}\n\n")
            f.write(rep + "\n\n")
            f.write("===== FULL CONTENT =====\n")
            f.write(body + "\n")

        logger.info(f"[NEWS] saved: {fname} fulltext={bool(fulltext)} len={len(body)}")

        idx[nid] = fname
        created.append(path)

    if created:
        save_news_index(idx)
        logger.info(f"[NEWS] created {len(created)} new files")

    return created


def crawl_news_once() -> List[str]:
    """(11) 주기 크롤링 단발"""
    items: List[Dict[str, Any]] = []
    for kw in CFG["NEWS_KEYWORDS"]:
        try:
            items.extend(fetch_google_news(kw))
        except Exception as e:
            logger.info(f"[NEWS] fetch failed for '{kw}': {e}")
    return save_news_items(items)


def list_news_txts() -> List[str]:
    ensure_news_dir()
    return sorted(
        os.path.join(CFG["NEWS_DIR"], n)
        for n in os.listdir(CFG["NEWS_DIR"])
        if n.lower().endswith(".txt")
    )


def sync_news_dir(vs: Chroma, embed_model: str) -> int:
    """(12) 변경된/신규 뉴스 txt만 임베딩"""
    fps = load_fingerprints(embed_model)
    before = dict(fps.get("news", {}))

    total_added = 0
    for p in list_news_txts():
        fp = file_fingerprint(p)
        if before.get(p) == fp:
            continue

        delete_by_source(vs, p)

        loader = TextLoader(p, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["source_type"] = "news_text"

        total_added += add_documents(vs, docs, prefix="NEWS", reason=f"changed_or_new: {os.path.basename(p)}")
        fps["news"][p] = fp

    save_fingerprints(embed_model, fps)
    return total_added


# =========================
# 6) Periodic tasks in Streamlit
# =========================


def init_periodic_state():
    now = time.time()
    # 첫 실행(세션 최초 로드)에서는 주기 작업을 즉시 돌리지 않도록 '현재 시각'으로 초기화합니다.
    # 이렇게 하면 앱이 뜨는 속도가 빨라지고, 설정한 간격이 지난 뒤에만 크롤링/임베딩이 수행됩니다.
    st.session_state.setdefault("_last_pdf_sync", now)
    st.session_state.setdefault("_last_news_crawl", now)
    st.session_state.setdefault("_last_news_sync", now)


def periodic_tasks(vs: Chroma, embed_model: str) -> None:
    init_periodic_state()
    now = time.time()

    # (4) PDF 주기 체크
    if now - st.session_state["_last_pdf_sync"] >= int(CFG["PDF_SYNC_INTERVAL_SEC"]):
        try:
            added = sync_pdf_dir(vs, embed_model)
            logger.info(f"[PDF] periodic sync done. added_chunks={added}")
        except Exception as e:
            logger.info(f"[PDF] periodic sync failed: {e}")
        st.session_state["_last_pdf_sync"] = now

    # (11) 뉴스 주기 크롤링
    if now - st.session_state["_last_news_crawl"] >= int(CFG["NEWS_CRAWL_INTERVAL_SEC"]):
        try:
            created = crawl_news_once()
            logger.info(f"[NEWS] periodic crawl done. created_files={len(created)}")
        except Exception as e:
            logger.info(f"[NEWS] periodic crawl failed: {e}")
        st.session_state["_last_news_crawl"] = now

    # (12) 뉴스 임베딩 동기화
    if now - st.session_state["_last_news_sync"] >= int(CFG["NEWS_CRAWL_INTERVAL_SEC"]):
        try:
            added = sync_news_dir(vs, embed_model)
            logger.info(f"[NEWS] periodic sync done. added_chunks={added}")
        except Exception as e:
            logger.info(f"[NEWS] periodic sync failed: {e}")
        st.session_state["_last_news_sync"] = now


# --- LLM caching + Korean-only post-processing helpers ---
@st.cache_resource
def get_llm(model_name: str):
    # Deterministic output helps reduce language-mixing
    # Also bias toward Korean via system prompt and sampling controls.
    try:
        return ChatOllama(
            model=model_name,
            temperature=0.0,
            top_p=0.9,
            num_predict=512,
            repeat_penalty=1.1,
        )
    except TypeError:
        # Fallback for older langchain_ollama versions
        return ChatOllama(model=model_name, temperature=0.0, top_p=0.9)


_ALLOWED_ASCII_ACRONYMS = {
    "SDV", "LLM", "AI", "ISO", "IEC", "ASIL", "HARA", "FMEA", "FMEDA", "FTA", "GSN",
}


def _needs_korean_rewrite(text: str) -> bool:
    """Detect if the answer contains too much non-Korean text.

    - Allow short ASCII acronyms (SDV, ISO 26262, AI, LLM, etc.)
    - If there are many Latin words or CJK (non-Hangul) characters, request rewrite.
    - Be aggressive for any non-Hangul CJK, Japanese kana, Cyrillic, Turkish, or Latin words (>=3 letters, not in allowlist).
    """
    if not text:
        return False

    s = text.strip()
    hangul = len(re.findall(r"[가-힣]", s))

    # Latin words (length >= 3) excluding allowed acronyms
    latin_words = re.findall(r"[A-Za-z]{3,}", s)
    latin_words_filtered = [w for w in latin_words if w.upper() not in _ALLOWED_ASCII_ACRONYMS]

    # Japanese Kana + CJK ideographs + Cyrillic etc. (anything we don't want mixed in)
    jp_kana = re.findall(r"[\u3040-\u30ff]", s)
    cjk_ideographs = re.findall(r"[\u4e00-\u9fff]", s)
    cyrillic = re.findall(r"[\u0400-\u04FF]", s)
    turkish = re.findall(r"[ğĞşŞıİöÖüÜ]", s)

    # If ANY Japanese kana appear, rewrite.
    if jp_kana:
        return True

    # If CJK ideographs appear together with Hangul, it's likely mixed-language (e.g., 自動運転).
    if cjk_ideographs and hangul > 0:
        return True

    # If Cyrillic/Turkish letters appear, rewrite.
    if cyrillic or turkish:
        return True

    # If there are any non-allowed Latin words and some Hangul exists, rewrite.
    if latin_words_filtered and hangul > 0:
        return True

    # If Hangul is very scarce but foreign tokens exist, rewrite.
    if hangul < 10 and (latin_words_filtered or cjk_ideographs or cyrillic or turkish):
        return True

    return False


def _rewrite_to_korean_only(llm_model: str, answer: str) -> str:
    """Rewrite answer into natural Korean. Keep technical acronyms as-is."""
    llm = get_llm(llm_model)

    sys = (
        "당신은 한국어 편집자이자 번역가입니다. 아래 텍스트를 '자연스러운 한국어'로만 다시 작성하세요. "
        "영어/일본어/중국어/베트남어/터키어/러시아어 등 어떤 외국어 문장도 섞지 마세요. "
        "외국어 단어/문장/한자 표기(예: - 進める, 自動運転, 私の, expertise, özellikle 등)가 있으면 의미를 유지한 채 한국어로 번역해 바꿔 쓰세요. "
        "다만 기술 약어/표준명/제품명(예: SDV, LLM, AI, ISO 26262, UNECE R155)은 필요할 때만 그대로 유지할 수 있습니다. "
        "문체는 존댓말로 공손하게 유지하고, 문장은 매끄럽고 자연스럽게 연결하세요."
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", sys), ("human", "다음 텍스트를 한국어로만 자연스럽게 다시 써 주세요:\n\n{answer}")]
    )

    try:
        msg = prompt.invoke({"answer": answer})
        out = llm.invoke(msg)
        return getattr(out, "content", str(out)).strip()
    except Exception:
        return answer


def ensure_korean_output(llm_model: str, answer: str) -> str:
    if _needs_korean_rewrite(answer):
        out = _rewrite_to_korean_only(llm_model, answer)
        # If still mixed, do one more pass (prevents stubborn mixed-script outputs)
        if out and _needs_korean_rewrite(out):
            out = _rewrite_to_korean_only(llm_model, out)
        return out
    return answer



# =========================
# 7) RAG chain
# =========================


def build_rag_chain(vs: Chroma, llm_model: str):
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = (
        "당신은 한국어 질문-답변 도우미입니다. 아래의 검색된 문서 조각(Context)을 참고해서 답변하세요.\n"
        "- 답변은 반드시 자연스러운 한국어로만 작성합니다.\n"
        "- 영어/일본어/중국어 등 외국어 문장은 쓰지 않습니다. 외국어가 필요해 보이면 한국어로 번역해 설명합니다.\n"
        "- 기술 약어/표준명/제품명(예: SDV, ISO 26262, LLM, AI)은 필요한 경우에만 그대로 유지할 수 있습니다.\n"
        "- 근거가 부족하면 추측하지 말고 모른다고 말합니다.\n"
        "- 존댓말을 사용해 공손하게 답변합니다.\n\n"
        "[Context]\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = get_llm(llm_model)

    # (7) 검색 기반 질의 업데이트(히스토리 기반 재구성 가능)
    if create_history_aware_retriever is not None:
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    else:
        # Fallback: retriever는 문자열 query를 받으므로 input만 전달
        history_aware_retriever = RunnableLambda(lambda x: x["input"]) | retriever

    def _format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))

    def _build_inputs(x: Dict[str, Any]) -> Dict[str, Any]:
        # RunnableWithMessageHistory가 주입하는 history는 그대로 보존
        return {
            "input": x.get("input", ""),
            "history": x.get("history", []),
        }

    def _retrieve(x: Dict[str, Any]):
        # history-aware retriever는 dict를 입력으로 받고 문서 리스트를 반환
        return history_aware_retriever.invoke(x)

    def _to_qa_vars(x: Dict[str, Any]) -> Dict[str, Any]:
        docs = x.get("context_docs", []) or []
        return {
            "input": x.get("input", ""),
            "history": x.get("history", []),
            "context_docs": docs,
            "context": _format_docs(docs),
        }

    def _final(x: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": x.get("answer", ""), "context": x.get("context_docs", [])}

    rag_chain = (
        RunnableLambda(_build_inputs)
        .assign(context_docs=RunnableLambda(_retrieve))
        .assign(**{
            "context": RunnableLambda(lambda x: _format_docs(x.get("context_docs", []))),
        })
        .assign(
            answer=(
                qa_prompt
                | llm
                | RunnableLambda(lambda m: getattr(m, "content", str(m)))
            )
        )
        | RunnableLambda(_final)
    )

    return rag_chain


def build_pure_llm_chain(llm_model: str):
    #llm = ChatOllama(
    #    model=llm_model,
    #    temperature=0.0,
    #    top_p=0.9,
    #)
    llm = get_llm(llm_model)

    sys = "당신은 친절한 한국어 도우미입니다. 답변은 자연스러운 한국어로만 작성하고(외국어 문장 금지), 존댓말로 공손하게 답하세요."
    prompt_tpl = ChatPromptTemplate.from_messages(
        [("system", sys), MessagesPlaceholder("history"), ("human", "{input}")]
    )

    chain = (
        RunnablePassthrough
        .assign(
            prompt=RunnableLambda(
                lambda x: prompt_tpl.invoke({
                    "history": x.get("history", []),
                    "input": x.get("input", ""),
                })
            )
        )
        .assign(answer=RunnableLambda(lambda x: llm.invoke(x["prompt"])))
        | RunnableLambda(lambda x: {"answer": getattr(x["answer"], "content", str(x["answer"]))})
    )
    return chain


# =========================
# 8) Multi chat sessions UI
# =========================

# --- Chat title keyword extraction (titles derived from first message keywords) ---
import unicodedata

# Small Korean/English stopword set
_CHAT_TITLE_STOPWORDS = set([
    "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "a", "an", "is", "are", "was", "were", "be",
    "by", "as", "from", "that", "this", "it", "but", "if", "then", "so", "not", "do", "does", "did",
    "i", "you", "he", "she", "we", "they", "my", "your", "our", "their", "me", "him", "her", "them", "us",
    "can", "will", "would", "should", "could", "may", "might", "must",
    "how", "what", "when", "where", "who", "which", "why", "about",
    # Korean stopwords (common particles, pronouns, etc.)
    "의", "이", "가", "은", "는", "을", "를", "에", "에서", "에게", "께", "로", "으로", "와", "과", "도", "만",
    "보다", "처럼", "까지", "부터", "하고", "보다", "마다", "라도", "이나", "나", "든지", "조차", "마저", "밖에",
    "및", "등", "또는", "그리고", "그러나", "하지만", "그래서", "즉", "혹은", "때문에", "그러므로", "따라서",
    "저", "나", "너", "우리", "저희", "너희", "그", "그녀", "이것", "저것", "그것", "누구", "무엇", "어디", "언제", "어떻게", "왜",
])

def _extract_title_keywords(text: str, max_terms: int = 3) -> List[str]:
    """
    Extract up to max_terms representative keywords from the text for chat title.
    - Normalizes whitespace.
    - Extracts tokens using regex (Korean, alphanum, technical tokens).
    - Filters stopwords and very short tokens (len < 2 unless contains digit).
    - De-duplicates while preserving order.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    # Regex: Korean, alphanum, technical (e.g., ISO26262, SDV, R155)
    # Allow: 한글, 영어, 숫자, ISO/SDV/R155 등
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z0-9]{2,}(?:[.-][A-Za-z0-9]+)*", text)
    # Lowercase for stopword filtering, but preserve original for output
    seen = set()
    result = []
    for tok in tokens:
        tok_norm = tok.lower()
        # Remove stopwords
        if tok_norm in _CHAT_TITLE_STOPWORDS:
            continue
        # Remove very short tokens unless contains digit (e.g. "R155")
        if len(tok) < 2 and not any(c.isdigit() for c in tok):
            continue
        # De-duplicate
        if tok_norm in seen:
            continue
        seen.add(tok_norm)
        result.append(tok)
        if len(result) >= max_terms:
            break
    return result

def _make_chat_title(user_text: str, ai_text: str = "") -> str:
    """
    Make a chat title using representative keywords from user_text (and ai_text if needed).
    Joins keywords with " · ". Fallback: "새 채팅".
    """
    keywords = _extract_title_keywords(user_text, max_terms=3)
    if len(keywords) < 2 and ai_text:
        # Supplement from answer
        ai_kw = _extract_title_keywords(ai_text, max_terms=3)
        # Add only new keywords
        for k in ai_kw:
            if k not in keywords:
                keywords.append(k)
            if len(keywords) >= 3:
                break
    if keywords:
        return " · ".join(keywords)
    return "새 채팅"


def init_chat_registry():
    # Load from disk first (survive browser refresh)
    if "chat_registry" not in st.session_state:
        disk_reg = _load_chat_registry_from_disk()
        st.session_state["chat_registry"] = disk_reg

    st.session_state.setdefault("chat_registry", {})  # session_id -> title

    # Ensure there is at least one chat
    if not st.session_state.get("chat_registry"):
        sid = str(uuid.uuid4())
        st.session_state["chat_registry"][sid] = "새 채팅"
        st.session_state["active_chat_id"] = sid
        _save_chat_registry_to_disk(st.session_state["chat_registry"])
        return

    # Restore active chat id (prefer existing)
    if "active_chat_id" not in st.session_state:
        # Pick first chat deterministically
        first_sid = next(iter(st.session_state["chat_registry"].keys()))
        st.session_state["active_chat_id"] = first_sid


def new_chat():
    reg: Dict[str, str] = st.session_state["chat_registry"]
    sid = str(uuid.uuid4())
    # New chats are initially titled "새 채팅"
    reg[sid] = "새 채팅"
    st.session_state["active_chat_id"] = sid
    _save_chat_registry_to_disk(reg)


def sidebar_chat_list():
    st.sidebar.header("채팅 목록")

    busy = bool(st.session_state.get("_is_generating", False))
    active = st.session_state["active_chat_id"]

    # Busy notice (do NOT disable buttons; allow switch requests)
    if busy:
        st.sidebar.info("답변 생성 중입니다. 지금 전환/새 채팅을 누르면 '전환 예약'으로 처리되고, 답변이 끝나면 자동 전환됩니다.")

    # New chat button: allowed even when busy (will be scheduled)
    if st.sidebar.button("+ 새 채팅", key="btn_new_chat"):
        if busy:
            sid = str(uuid.uuid4())
            st.session_state["chat_registry"][sid] = "새 채팅"
            _save_chat_registry_to_disk(st.session_state["chat_registry"])
            # schedule switching after generation
            st.session_state["_switch_to_chat_id"] = sid
            st.session_state["_switch_requested_at"] = time.time()
            st.sidebar.success("새 채팅을 만들었습니다. 답변 생성이 끝나면 자동으로 전환됩니다.")
        else:
            new_chat()
            st.rerun()

    reg: Dict[str, str] = st.session_state["chat_registry"]

    for sid, title in reg.items():
        label = ("✅ " if sid == active else "") + title
        if st.sidebar.button(label, key=f"chat_{sid}"):
            if busy:
                st.session_state["_switch_to_chat_id"] = sid
                st.session_state["_switch_requested_at"] = time.time()
                st.sidebar.warning("전환을 예약했습니다. 답변 생성이 완료되면 자동 전환됩니다.")
                # rerun to reflect selection intent in UI immediately
                st.rerun()
            else:
                # save current chat messages before switching
                try:
                    cur = st.session_state.get("active_chat_id")
                    if cur:
                        _save_messages_to_disk(cur, get_chat_history(cur).messages)
                except Exception:
                    pass
                st.session_state["active_chat_id"] = sid
                st.rerun()

    st.sidebar.divider()


# =========================
# 9) Streamlit App
# =========================

st.set_page_config(page_title="TRAG BaseRAG_v3", layout="wide")

# Runtime flags: prevent rerun while generating
st.session_state.setdefault("_is_generating", False)
st.session_state.setdefault("_gen_started_at", None)
# Busy/switch control
st.session_state.setdefault("_busy_chat_id", None)
st.session_state.setdefault("_switch_to_chat_id", None)
st.session_state.setdefault("_switch_requested_at", None)

# If a previous run was interrupted (e.g., user switched chat / stopped the app while generating),
# Streamlit might not execute `finally:` blocks and `_is_generating` can remain True forever.
# This guard resets stale "generating" state so the UI won't be stuck.
GEN_STALE_SEC = int(CFG.get("GENERATION_STALE_SEC", 180))
now_ts = time.time()
if st.session_state.get("_is_generating", False):
    started = st.session_state.get("_gen_started_at")
    busy_chat = st.session_state.get("_busy_chat_id")
    if (started is None) or (busy_chat is None) or (now_ts - float(started) > GEN_STALE_SEC):
        logger.info(
            f"[GUARD] Reset stale generating state. started_at={started} now={now_ts} stale_sec={GEN_STALE_SEC}"
        )
        st.session_state["_is_generating"] = False
        st.session_state["_gen_started_at"] = None
        st.session_state["_busy_chat_id"] = None
        st.session_state["_switch_to_chat_id"] = None
        st.session_state["_switch_requested_at"] = None

# If config file missing, show guidance
if not os.path.exists(CONFIG_FILE):
    st.sidebar.warning("./config/config.py가 없습니다. 기본 템플릿을 생성합니다.")

# Sidebar chat list
init_chat_registry()
sidebar_chat_list()


# Apply scheduled switch only when not generating
if not st.session_state.get("_is_generating", False):
    sid = st.session_state.get("_switch_to_chat_id")
    if sid and sid in st.session_state.get("chat_registry", {}):
        st.session_state["active_chat_id"] = sid
        st.session_state["_switch_to_chat_id"] = None
        st.session_state["_switch_requested_at"] = None
        st.rerun()

# If a switch request is very old, clear it (prevents permanent "reserved" state)
req_at = st.session_state.get("_switch_requested_at")
if req_at and (time.time() - float(req_at) > int(CFG.get("GENERATION_STALE_SEC", 180))):
    st.session_state["_switch_to_chat_id"] = None
    st.session_state["_switch_requested_at"] = None


# Sidebar: upload + manual sync
st.sidebar.header("데이터")
up = st.sidebar.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=False)
if up is not None:
    save_uploaded_pdf(up)

if st.sidebar.button("지금 PDF/NEWS 동기화"):
    st.session_state["_force_sync"] = True
    st.rerun()

# Sidebar: show current config
st.sidebar.header("설정")
st.sidebar.caption(f"LLM={CFG['LLM_MODEL']} / EMB={CFG['EMBED_MODEL']}")
st.sidebar.caption(f"chunk={CFG['CHUNK_SIZE']} overlap={CFG['CHUNK_OVERLAP']}")
st.sidebar.caption(f"pdf_interval={CFG['PDF_SYNC_INTERVAL_SEC']}s news_interval={CFG['NEWS_CRAWL_INTERVAL_SEC']}s")
st.sidebar.caption("뉴스 키워드:")
for k in CFG["NEWS_KEYWORDS"]:
    st.sidebar.caption(f"- {k}")
st.sidebar.caption(f"config: {CONFIG_FILE}")

# Optional autorefresh so periodic tasks can run without user input
if bool(CFG["AUTO_REFRESH_ENABLED"]) and (not st.session_state.get("_is_generating", False)):
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(CFG["AUTO_REFRESH_TICK_SEC"]) * 1000, key="auto_refresh")
    except Exception:
        pass

# Vectorstore
vs = get_vectorstore(CFG["EMBED_MODEL"])

# Sync is intentionally NOT executed on first load.
# Only run sync when the user explicitly requests it (button), or via periodic tasks.
if st.session_state.pop("_force_sync", False):
    try:
        ap = sync_pdf_dir(vs, CFG["EMBED_MODEL"])
        created = crawl_news_once()
        an = sync_news_dir(vs, CFG["EMBED_MODEL"])
        logger.info(
            f"[FORCE] sync done. pdf_added_chunks={ap}, news_created_files={len(created)}, news_added_chunks={an}"
        )
        # Give immediate UI feedback
        st.sidebar.success(
            f"동기화 완료: PDF chunks +{ap}, News files +{len(created)}, News chunks +{an}"
        )
    except Exception as e:
        logger.info(f"[FORCE] sync failed: {e}")
        st.sidebar.error(f"동기화 실패: {e}")

# Periodic tasks (4)(11) - will run based on configured intervals
periodic_tasks(vs, CFG["EMBED_MODEL"])

# Status
pdf_count = len(list_pdfs())
news_count = len(list_news_txts())

st.header("TRAG Chatbot v01 💬📚")
st.caption(
    f"PDF: {pdf_count}개 / News txt: {news_count}개 / VectorDB: {CFG['VECTOR_DB']} ({vectordb_dir(CFG['EMBED_MODEL'])})"
)
st.caption(f"LLM: {CFG['LLM_MODEL']} / Embedding: {CFG['EMBED_MODEL']} / Log: {LOG_PATH}")

# Choose chain: (9) no docs -> pure LLM
no_docs = (pdf_count == 0 and news_count == 0)

if no_docs:
    chain = build_pure_llm_chain(CFG["LLM_MODEL"])
else:
    chain = build_rag_chain(vs, CFG["LLM_MODEL"])

# Per-session history (16)(17)(18)(19)
session_id = st.session_state["active_chat_id"]
logger.info(f"[SESSION] active_chat_id={session_id}")

def get_chat_history(sid: str) -> StreamlitChatMessageHistory:
    # One independent StreamlitChatMessageHistory per chat session
    return StreamlitChatMessageHistory(key=f"chat_messages_{sid}")

chat_history = get_chat_history(session_id)
_hydrate_history_from_disk(session_id, chat_history)

# Render history
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Run chat
if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt)
    # 답변 생성 시작
    st.session_state["_is_generating"] = True
    st.session_state["_gen_started_at"] = time.time()
    st.session_state["_busy_chat_id"] = st.session_state.get("active_chat_id")
    answer = ""

    try:
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                if no_docs:
                    resp = chain.invoke({"input": prompt, "history": chat_history.messages})
                    answer = resp.get("answer", "")
                    answer = ensure_korean_output(CFG["LLM_MODEL"], answer)
                    st.write(answer)
                else:
                    conversational = RunnableWithMessageHistory(
                        chain,
                        get_chat_history,
                        input_messages_key="input",
                        history_messages_key="history",
                        output_messages_key="answer",
                    )
                    try:
                        resp = conversational.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}},
                        )
                    except Exception as e:
                        logger.info(f"[CHAT] RAG invoke failed: {e}")
                        resp = {"answer": f"답변 생성 중 오류가 발생했습니다: {e}", "context": []}

                    answer = resp.get("answer", "")
                    answer = ensure_korean_output(CFG["LLM_MODEL"], answer)
                    st.write(answer)

                    with st.expander("참고 문서 확인"):
                        for doc in resp.get("context", []) or []:
                            src = doc.metadata.get("source", "unknown source")
                            st.markdown(src, help=doc.page_content)

        # (16) 현재 채팅창 종료 전까지 맥락 유지 (LLM 답변은 벡터 DB에 저장하지 않음)
        #chat_history.add_user_message(prompt)
        #chat_history.add_ai_message(answer)

        # --- Auto-rename chat after first exchange using keywords ---
        reg: Dict[str, str] = st.session_state.get("chat_registry", {})
        current_title = reg.get(session_id, "")
        # Only update if current title is "새 채팅" or starts with "Chat" (backward compatibility)
        if current_title == "새 채팅" or (current_title or "").startswith("Chat"):
            new_title = _make_chat_title(prompt, answer)
            st.session_state["chat_registry"][session_id] = new_title
            _save_chat_registry_to_disk(st.session_state["chat_registry"])
            # Do NOT st.rerun() here; sidebar will reflect on next rerun

        # ✅ Persist messages to disk so browser refresh won't lose them
        try:
            _save_messages_to_disk(session_id, chat_history.messages)
        except Exception:
            pass

    finally:
        st.session_state["_is_generating"] = False
        st.session_state["_gen_started_at"] = None
        st.session_state["_busy_chat_id"] = None

        # Apply scheduled switch if any
    sid = st.session_state.get("_switch_to_chat_id")
    if sid and sid in st.session_state.get("chat_registry", {}):
        st.session_state["active_chat_id"] = sid
        _save_chat_registry_to_disk(st.session_state["chat_registry"])
        st.session_state["_switch_to_chat_id"] = None
        st.session_state["_switch_requested_at"] = None
        st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.caption(f"로그 파일: {LOG_PATH}")
