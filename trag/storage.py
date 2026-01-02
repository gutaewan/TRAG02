import os
import json
import time
import uuid
import re
import logging
from typing import Dict, List, Any, Callable

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# -------------------------
# Disk paths
# -------------------------
def get_chat_store_dir(cfg: Dict[str, Any]) -> str:
    chat_store_dir = os.path.join(str(cfg["VECTOR_DB_ROOT"]), "chat_store")
    os.makedirs(chat_store_dir, exist_ok=True)
    return chat_store_dir


def _chat_registry_path(chat_store_dir: str) -> str:
    return os.path.join(chat_store_dir, "chat_registry.json")


def _chat_messages_path(chat_store_dir: str, session_id: str) -> str:
    return os.path.join(chat_store_dir, f"chat_{session_id}.json")


# -------------------------
# Registry load/save
# -------------------------
def load_chat_registry(chat_store_dir: str, logger: logging.Logger) -> Dict[str, str]:
    p = _chat_registry_path(chat_store_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
            return out
        return {}
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to load registry: {e}")
        return {}


def save_chat_registry(chat_store_dir: str, reg: Dict[str, str], logger: logging.Logger) -> None:
    try:
        os.makedirs(chat_store_dir, exist_ok=True)
        tmp = _chat_registry_path(chat_store_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _chat_registry_path(chat_store_dir))
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to save registry: {e}")


# -------------------------
# Messages load/save
# -------------------------
def _serialize_messages(msgs) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in msgs or []:
        try:
            t = getattr(m, "type", None) or m.get("type")
            c = getattr(m, "content", None) or m.get("content")
            if t and c is not None:
                out.append({"type": str(t), "content": str(c)})
        except Exception:
            continue
    return out


def load_messages(chat_store_dir: str, session_id: str, logger: logging.Logger) -> List[Dict[str, str]]:
    p = _chat_messages_path(chat_store_dir, session_id)
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


def save_messages(chat_store_dir: str, session_id: str, msgs, logger: logging.Logger) -> None:
    try:
        os.makedirs(chat_store_dir, exist_ok=True)
        tmp = _chat_messages_path(chat_store_dir, session_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_serialize_messages(msgs), f, ensure_ascii=False, indent=2)
        os.replace(tmp, _chat_messages_path(chat_store_dir, session_id))
    except Exception as e:
        logger.info(f"[CHAT_STORE] failed to save messages for {session_id}: {e}")


def hydrate_history(chat_store_dir: str, session_id: str, history: StreamlitChatMessageHistory, logger: logging.Logger) -> None:
    try:
        if history.messages:
            return
        disk_msgs = load_messages(chat_store_dir, session_id, logger)
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


# -------------------------
# Title 만들기(키워드 기반)
# -------------------------
_CHAT_TITLE_STOPWORDS = set([
    "the","and","or","of","in","on","at","to","for","with","a","an","is","are","was","were","be",
    "by","as","from","that","this","it","but","if","then","so","not","do","does","did",
    "i","you","he","she","we","they","my","your","our","their","me","him","her","them","us",
    "can","will","would","should","could","may","might","must",
    "how","what","when","where","who","which","why","about",
    "의","이","가","은","는","을","를","에","에서","에게","께","로","으로","와","과","도","만",
    "보다","처럼","까지","부터","하고","마다","라도","이나","나","든지","조차","마저","밖에",
    "및","등","또는","그리고","그러나","하지만","그래서","즉","혹은","때문에","그러므로","따라서",
    "저","나","너","우리","저희","너희","그","그녀","이것","저것","그것","누구","무엇","어디","언제","어떻게","왜",
])


def _extract_title_keywords(text: str, max_terms: int = 3) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z0-9]{2,}(?:[.-][A-Za-z0-9]+)*", text)

    seen = set()
    result: List[str] = []
    for tok in tokens:
        tok_norm = tok.lower()
        if tok_norm in _CHAT_TITLE_STOPWORDS:
            continue
        if tok_norm in seen:
            continue
        seen.add(tok_norm)
        result.append(tok)
        if len(result) >= max_terms:
            break
    return result


def make_chat_title(user_text: str, ai_text: str = "") -> str:
    keywords = _extract_title_keywords(user_text, max_terms=3)
    if len(keywords) < 2 and ai_text:
        ai_kw = _extract_title_keywords(ai_text, max_terms=3)
        for k in ai_kw:
            if k not in keywords:
                keywords.append(k)
            if len(keywords) >= 3:
                break
    if keywords:
        return " · ".join(keywords)
    return "새 채팅"


# -------------------------
# Streamlit session state helpers
# -------------------------
def init_chat_registry_state(chat_store_dir: str, logger: logging.Logger) -> None:
    if "chat_registry" not in st.session_state:
        st.session_state["chat_registry"] = load_chat_registry(chat_store_dir, logger)

    st.session_state.setdefault("chat_registry", {})

    if not st.session_state.get("chat_registry"):
        sid = str(uuid.uuid4())
        st.session_state["chat_registry"][sid] = "새 채팅"
        st.session_state["active_chat_id"] = sid
        save_chat_registry(chat_store_dir, st.session_state["chat_registry"], logger)
        return

    if "active_chat_id" not in st.session_state:
        first_sid = next(iter(st.session_state["chat_registry"].keys()))
        st.session_state["active_chat_id"] = first_sid


def new_chat_state(chat_store_dir: str, logger: logging.Logger) -> str:
    reg: Dict[str, str] = st.session_state["chat_registry"]
    sid = str(uuid.uuid4())
    reg[sid] = "새 채팅"
    st.session_state["active_chat_id"] = sid
    save_chat_registry(chat_store_dir, reg, logger)
    return sid


def sidebar_chat_list_ui(
    chat_store_dir: str,
    logger: logging.Logger,
    get_chat_history_fn: Callable[[str], StreamlitChatMessageHistory],
) -> None:
    st.sidebar.header("채팅 목록")

    busy = bool(st.session_state.get("_is_generating", False))
    active = st.session_state["active_chat_id"]

    if busy:
        st.sidebar.info("답변 생성 중입니다. 지금 전환/새 채팅을 누르면 '전환 예약'으로 처리되고, 답변이 끝나면 자동으로 전환됩니다.")

    if st.sidebar.button("+ 새 채팅", key="btn_new_chat"):
        if busy:
            sid = str(uuid.uuid4())
            st.session_state["chat_registry"][sid] = "새 채팅"
            save_chat_registry(chat_store_dir, st.session_state["chat_registry"], logger)

            st.session_state["_switch_to_chat_id"] = sid
            st.session_state["_switch_requested_at"] = time.time()
            st.sidebar.success("새 채팅을 만들었습니다. 답변 생성이 끝나면 자동으로 전환됩니다.")
        else:
            new_chat_state(chat_store_dir, logger)
            st.rerun()

    reg: Dict[str, str] = st.session_state["chat_registry"]

    for sid, title in reg.items():
        label = ("✅ " if sid == active else "") + title
        if st.sidebar.button(label, key=f"chat_{sid}"):
            if busy:
                st.session_state["_switch_to_chat_id"] = sid
                st.session_state["_switch_requested_at"] = time.time()
                st.sidebar.warning("전환을 예약했습니다. 답변 생성이 완료되면 자동 전환됩니다.")
                st.rerun()
            else:
                try:
                    cur = st.session_state.get("active_chat_id")
                    if cur:
                        save_messages(chat_store_dir, cur, get_chat_history_fn(cur).messages, logger)
                except Exception:
                    pass
                st.session_state["active_chat_id"] = sid
                st.rerun()

    st.sidebar.divider()


import os
import time
from contextlib import contextmanager

@contextmanager
def file_lock(lock_path: str, timeout_sec: int = 60, poll_sec: float = 0.2):
    """A simple cross-process lock using an exclusive lock file.
    Ensures only one process performs VectorDB write operations at a time.
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Lock timeout: {lock_path}")
            time.sleep(poll_sec)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass