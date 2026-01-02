import os
import time
import logging
import re
from typing import Dict, Any, List
import threading
import queue
import traceback

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama


from trag.vectorstore import (
    get_vectorstore,
    vectordb_dir,
    list_pdfs,
    save_uploaded_pdf,
    sync_pdf_dir,
)

from trag.news_crawler import (
    crawl_news_once,
    list_news_txts,
    sync_news_dir,
)

from trag.storage import (
    get_chat_store_dir,
    init_chat_registry_state,
    sidebar_chat_list_ui,
    hydrate_history,
    save_messages,
    save_chat_registry,
    make_chat_title,
)


# LangChain 버전에 따라 import 경로가 달라질 수 있어 호환 처리
try:
    from langchain.chains import create_history_aware_retriever
except Exception:
    try:
        from langchain.chains.history_aware_retriever import create_history_aware_retriever
    except Exception:
        create_history_aware_retriever = None


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



os.makedirs(str(CFG["LOG_DIR"]), exist_ok=True)
LOG_PATH = os.path.join(str(CFG["LOG_DIR"]), "app.log")

logger = logging.getLogger("trag")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# =========================
# Background Workers (threading)
# =========================

@st.cache_resource
def get_worker_bus():
    """Singleton bus for background workers.

    cmd_q: UI -> worker commands
    event_q: worker -> UI events
    db_lock: serialize Chroma access (avoid concurrent read/write corruption)
    stop_event: allow graceful stop (best-effort)
    """
    return {
        "cmd_q": queue.Queue(),
        "event_q": queue.Queue(),
        "db_lock": threading.Lock(),
        "stop_event": threading.Event(),
        "started": False,
        "thread": None,
        "status": {
            "busy": False,
            "last_ts": None,
            "pdf_added_chunks": 0,
            "news_created_files": 0,
            "news_added_chunks": 0,
            "last_error": None,
        },
    }


def _bus_emit(bus, typ: str, msg: str = "", **data):
    try:
        payload = {"ts": time.time(), "type": typ, "msg": msg}
        payload.update(data)
        bus["event_q"].put(payload)
    except Exception:
        pass


def _run_sync_cycle(bus):
    """One sync cycle: PDF sync + news crawl + news sync.

    NOTE: Must not call streamlit APIs.
    """
    try:
        cfg = load_config()
        embed_model = str(cfg.get("EMBED_MODEL", "qwen3-embedding"))

        _bus_emit(bus, "sync_start", "sync started")
        bus["status"]["busy"] = True
        bus["status"]["last_error"] = None

        # Build/open vectorstore inside lock to avoid concurrent open/write
        with bus["db_lock"]:
            vs_local = get_vectorstore(cfg, embed_model, logger)
            pdf_added = sync_pdf_dir(vs_local, cfg, embed_model, logger)
            created_files = crawl_news_once(cfg, logger)
            news_added = sync_news_dir(vs_local, cfg, embed_model, logger)

        bus["status"].update({
            "busy": False,
            "last_ts": time.time(),
            "pdf_added_chunks": int(pdf_added or 0),
            "news_created_files": int(len(created_files or [])),
            "news_added_chunks": int(news_added or 0),
            "last_error": None,
        })

        # Log vector DB state change (Req 5,14)
        logger.info(
            "[WORKER] sync done. pdf_added_chunks=%s, news_created_files=%s, news_added_chunks=%s",
            bus["status"]["pdf_added_chunks"],
            bus["status"]["news_created_files"],
            bus["status"]["news_added_chunks"],
        )
        _bus_emit(bus, "sync_end", "sync finished", **bus["status"])

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        bus["status"].update({
            "busy": False,
            "last_ts": time.time(),
            "pdf_added_chunks": 0,
            "news_created_files": 0,
            "news_added_chunks": 0,
            "last_error": err,
        })
        logger.info(f"[WORKER] sync failed: {err}\n{traceback.format_exc()}")
        _bus_emit(bus, "sync_error", err, **bus["status"])
    finally:
        bus["status"]["busy"] = False


def _worker_loop(bus):
    """Background loop.

    - Does NOT sync on first start (Req: do not sync on first run)
    - Periodically checks intervals for PDF/news and runs sync.
    - Responds to explicit sync_now commands.
    """
    _bus_emit(bus, "worker_started", "worker started")

    # Do not run immediately; set last_run to now.
    last_run = time.time()

    while not bus["stop_event"].is_set():
        try:
            # Handle commands
            cmd = None
            try:
                cmd = bus["cmd_q"].get_nowait()
            except queue.Empty:
                cmd = None

            if isinstance(cmd, dict) and cmd.get("type") == "sync_now":
                _run_sync_cycle(bus)
                last_run = time.time()

            cfg = load_config()
            pdf_iv = int(cfg.get("PDF_SYNC_INTERVAL_SEC", 600))
            news_iv = int(cfg.get("NEWS_CRAWL_INTERVAL_SEC", 600))
            interval = min(pdf_iv, news_iv)

            if (time.time() - last_run) >= max(5, interval):
                _run_sync_cycle(bus)
                last_run = time.time()

            # sleep in small increments for responsiveness
            for _ in range(10):
                if bus["stop_event"].is_set():
                    break
                time.sleep(1)

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.info(f"[WORKER] loop error: {err}\n{traceback.format_exc()}")
            _bus_emit(bus, "worker_error", err)
            time.sleep(2)


def start_worker_if_needed():
    bus = get_worker_bus()
    if bus.get("started") and bus.get("thread") is not None:
        return bus

    t = threading.Thread(target=_worker_loop, args=(bus,), daemon=True)
    bus["thread"] = t
    bus["started"] = True
    t.start()
    return bus


def drain_worker_events(bus):
    """Drain worker events into st.session_state for UI rendering."""
    st.session_state.setdefault("_worker_status", {})
    st.session_state.setdefault("_worker_last_event", None)

    drained = 0
    try:
        while drained < 50:
            evt = bus["event_q"].get_nowait()
            drained += 1
            st.session_state["_worker_last_event"] = evt
            # Keep latest status snapshot
            if "busy" in evt or "pdf_added_chunks" in evt:
                st.session_state["_worker_status"] = {
                    "busy": bool(evt.get("busy", bus["status"].get("busy"))),
                    "last_ts": evt.get("last_ts", bus["status"].get("last_ts")),
                    "pdf_added_chunks": evt.get("pdf_added_chunks", bus["status"].get("pdf_added_chunks")),
                    "news_created_files": evt.get("news_created_files", bus["status"].get("news_created_files")),
                    "news_added_chunks": evt.get("news_added_chunks", bus["status"].get("news_added_chunks")),
                    "last_error": evt.get("last_error", bus["status"].get("last_error")),
                }
    except queue.Empty:
        pass
    except Exception:
        pass









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
    # Defensive import: prevents NameError if `re` is shadowed or missing in some environments
    import re as _re
    if not text:
        return False

    s = text.strip()
    hangul = len(_re.findall(r"[가-힣]", s))

    # Latin words (length >= 3) excluding allowed acronyms
    latin_words = _re.findall(r"[A-Za-z]{3,}", s)
    latin_words_filtered = [w for w in latin_words if w.upper() not in _ALLOWED_ASCII_ACRONYMS]

    # Japanese Kana + CJK ideographs + Cyrillic etc. (anything we don't want mixed in)
    jp_kana = _re.findall(r"[\u3040-\u30ff]", s)
    cjk_ideographs = _re.findall(r"[\u4e00-\u9fff]", s)
    cyrillic = _re.findall(r"[\u0400-\u04FF]", s)
    turkish = _re.findall(r"[ğĞşŞıİöÖüÜ]", s)

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


# --- English->Korean helper: output both English and Korean versions ---
from typing import Dict

def english_then_korean(llm_model: str, answer: str) -> Dict[str, str]:
    """Return both English and Korean versions.

    Strategy:
    - If answer already looks English, keep it as `en`.
    - Otherwise rewrite to English first.
    - Then translate EN -> natural Korean.
    - UI should normally show only Korean; English can be shown in an expander.
    """
    llm = get_llm(llm_model)

    def _rewrite_to_english(text: str) -> str:
        sys = (
            "You are a technical writer. Rewrite the given text into clear, natural English only. "
            "Do not include Korean, Japanese, Chinese, Turkish, Russian, or any other language. "
            "Keep technical acronyms/standards/product names as-is (e.g., SDV, LLM, AI, ISO 26262, UNECE R155)."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", sys), ("human", "Rewrite into English only:\n\n{text}")]
        )
        try:
            msg = prompt.invoke({"text": text})
            out = llm.invoke(msg)
            return getattr(out, "content", str(out)).strip()
        except Exception:
            return text

    def _translate_en_to_ko(text_en: str) -> str:
        sys = (
            "당신은 전문 번역가이자 한국어 편집자입니다. 입력된 영어 텍스트를 의미를 유지한 채 "
            "자연스럽고 매끄러운 한국어로 번역해 주세요. 외국어 문장을 섞지 말고, 존댓말을 사용하세요. "
            "기술 약어/표준명/제품명(예: SDV, LLM, AI, ISO 26262, UNECE R155)은 필요할 때만 그대로 유지할 수 있습니다."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", sys), ("human", "다음 영어 텍스트를 한국어로 번역해 주세요:\n\n{en}")]
        )
        try:
            msg = prompt.invoke({"en": text_en})
            out = llm.invoke(msg)
            return getattr(out, "content", str(out)).strip()
        except Exception:
            return text_en

    # Heuristic: if the text contains a lot of Hangul, rewrite to English first
    import re as _re
    hangul = len(_re.findall(r"[가-힣]", answer or ""))
    latin = len(_re.findall(r"[A-Za-z]", answer or ""))
    if hangul > 0 and hangul >= max(10, latin // 2):
        en = _rewrite_to_english(answer)
    else:
        en = (answer or "").strip()

    ko = _translate_en_to_ko(en)
    # Final safety: ensure output is Korean-only-ish
    ko = ensure_korean_output(llm_model, ko)
    return {"en": en, "ko": ko}



# =========================
# 7) RAG chain
# =========================


def build_rag_chain(vs, llm_model: str):
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
        "You are a question-answering assistant. Use the retrieved Context to answer the user.\n"
        "- Output MUST be English only. Do not include any Korean/Japanese/Chinese or other languages.\n"
        "- Keep technical acronyms/standards/product names as-is (e.g., SDV, ISO 26262, LLM, AI, UNECE R155).\n"
        "- If the Context is insufficient, say you don't know instead of guessing.\n\n"
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

    sys = (
        "You are a helpful assistant. Output MUST be English only. "
        "Do not include Korean/Japanese/Chinese or other languages. "
        "Keep technical acronyms/standards/product names as-is when needed."
    )
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

# Start background worker (Req 20)
_worker_bus = start_worker_if_needed()
# Drain worker->UI events (do not block)
drain_worker_events(_worker_bus)

# If config file missing, show guidance
if not os.path.exists(CONFIG_FILE):
    st.sidebar.warning("./config/config.py가 없습니다. 기본 템플릿을 생성합니다.")

# Sidebar chat list - use trag.storage module
chat_store_dir = get_chat_store_dir(CFG)
init_chat_registry_state(chat_store_dir, logger)

def get_chat_history(sid: str) -> StreamlitChatMessageHistory:
    # One independent StreamlitChatMessageHistory per chat session
    return StreamlitChatMessageHistory(key=f"chat_messages_{sid}")

sidebar_chat_list_ui(chat_store_dir, logger, get_chat_history)

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
    save_uploaded_pdf(up, CFG, logger)

if st.sidebar.button("지금 PDF/NEWS 동기화"):
    try:
        get_worker_bus()["cmd_q"].put({"type": "sync_now"})
        st.sidebar.info("동기화 요청을 백그라운드 worker에 전달했습니다.")
    except Exception as e:
        st.sidebar.error(f"동기화 요청 실패: {e}")

# Sidebar: show current config
st.sidebar.header("설정")
st.sidebar.caption(f"LLM={CFG['LLM_MODEL']} / EMB={CFG['EMBED_MODEL']}")
st.sidebar.caption(f"chunk={CFG['CHUNK_SIZE']} overlap={CFG['CHUNK_OVERLAP']}")
st.sidebar.caption(f"pdf_interval={CFG['PDF_SYNC_INTERVAL_SEC']}s news_interval={CFG['NEWS_CRAWL_INTERVAL_SEC']}s")
st.sidebar.caption("뉴스 키워드:")
for k in CFG["NEWS_KEYWORDS"]:
    st.sidebar.caption(f"- {k}")
st.sidebar.caption(f"config: {CONFIG_FILE}")

# Worker status
wstat = st.session_state.get("_worker_status", {}) or {}
if wstat.get("busy"):
    st.sidebar.warning("백그라운드 동기화 실행 중입니다.")
else:
    st.sidebar.success("백그라운드 동기화 대기 중입니다.")

if wstat.get("last_ts"):
    st.sidebar.caption(f"마지막 동기화: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(wstat['last_ts'])))}")
st.sidebar.caption(f"최근 추가 PDF chunks: {wstat.get('pdf_added_chunks', 0)}")
st.sidebar.caption(f"최근 생성 News files: {wstat.get('news_created_files', 0)}")
st.sidebar.caption(f"최근 추가 News chunks: {wstat.get('news_added_chunks', 0)}")
if wstat.get("last_error"):
    st.sidebar.error(f"worker 오류: {wstat['last_error']}")

# Optional autorefresh so periodic tasks can run without user input
if bool(CFG["AUTO_REFRESH_ENABLED"]):
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(CFG["AUTO_REFRESH_TICK_SEC"]) * 1000, key="auto_refresh")
    except Exception:
        pass

# Vectorstore (lazy open; created when needed for RAG)
vs = None


# Status
pdf_count = len(list_pdfs(CFG))
news_count = len(list_news_txts(CFG))

st.header("TRAG Chatbot v01 💬📚")
st.caption(
    f"PDF: {pdf_count}개 / News txt: {news_count}개 / VectorDB: {CFG['VECTOR_DB']} ({vectordb_dir(CFG, CFG['EMBED_MODEL'])})"
)
st.caption(f"LLM: {CFG['LLM_MODEL']} / Embedding: {CFG['EMBED_MODEL']} / Log: {LOG_PATH}")

# Choose chain: (9) no docs -> pure LLM
no_docs = (pdf_count == 0 and news_count == 0)

if no_docs:
    chain = build_pure_llm_chain(CFG["LLM_MODEL"])
else:
    # Open vectorstore lazily with db_lock to avoid chroma concurrent access
    with get_worker_bus()["db_lock"]:
        vs = get_vectorstore(CFG, CFG["EMBED_MODEL"], logger)
    chain = build_rag_chain(vs, CFG["LLM_MODEL"])

# Per-session history (16)(17)(18)(19)
session_id = st.session_state["active_chat_id"]
if st.session_state.get("_last_logged_session") != session_id:
    logger.info(f"[SESSION] active_chat_id={session_id}")
    st.session_state["_last_logged_session"] = session_id

chat_history = get_chat_history(session_id)
hydrate_history(chat_store_dir, session_id, chat_history, logger)

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
                    both = english_then_korean(CFG["LLM_MODEL"], answer)
                    answer_en = both["en"]
                    answer_ko = both["ko"]
                    st.write(answer_ko)
                    with st.expander("(원문) English 답변 보기"):
                        st.write(answer_en)
                    answer = answer_ko
                else:
                    conversational = RunnableWithMessageHistory(
                        chain,
                        get_chat_history,
                        input_messages_key="input",
                        history_messages_key="history",
                        output_messages_key="answer",
                    )
                    try:
                        with get_worker_bus()["db_lock"]:
                            resp = conversational.invoke(
                                {"input": prompt},
                                config={"configurable": {"session_id": session_id}},
                            )
                    except Exception as e:
                        logger.info(f"[CHAT] RAG invoke failed: {e}")
                        resp = {"answer": f"답변 생성 중 오류가 발생했습니다: {e}", "context": []}

                    answer = resp.get("answer", "")
                    both = english_then_korean(CFG["LLM_MODEL"], answer)
                    answer_en = both["en"]
                    answer_ko = both["ko"]
                    st.write(answer_ko)
                    with st.expander("(원문) English 답변 보기"):
                        st.write(answer_en)
                    answer = answer_ko

                    with st.expander("참고 문서 확인"):
                        for doc in resp.get("context", []) or []:
                            src = doc.metadata.get("source", "unknown source")
                            st.markdown(src, help=doc.page_content)

        # --- Auto-rename chat after first exchange using keywords ---
        reg: Dict[str, str] = st.session_state.get("chat_registry", {})
        current_title = reg.get(session_id, "")
        # Only update if current title is "새 채팅" or starts with "Chat" (backward compatibility)
        if current_title == "새 채팅" or (current_title or "").startswith("Chat"):
            new_title = make_chat_title(prompt, answer)
            st.session_state["chat_registry"][session_id] = new_title
            save_chat_registry(chat_store_dir, st.session_state["chat_registry"], logger)
            # Do NOT st.rerun() here; sidebar will reflect on next rerun

        # ✅ Persist messages to disk so browser refresh won't lose them
        try:
            save_messages(chat_store_dir, session_id, chat_history.messages, logger)
        except Exception:
            pass

    finally:
        # 항상 생성 플래그를 내려 UI가 고착되지 않게 합니다.
        st.session_state["_is_generating"] = False
        st.session_state["_gen_started_at"] = None
        st.session_state["_busy_chat_id"] = None

        # 답변 생성이 끝났으면 예약된 전환을 즉시 반영합니다.
        sid = st.session_state.get("_switch_to_chat_id")
        if sid and sid in st.session_state.get("chat_registry", {}):
            st.session_state["active_chat_id"] = sid
            st.session_state["_switch_to_chat_id"] = None
            st.session_state["_switch_requested_at"] = None
            st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.caption(f"로그 파일: {LOG_PATH}")
