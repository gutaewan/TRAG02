# BaseRAG.py
# ------------------------------------------------------------
# 이 파일(BaseRAG.py)의 역할 요약
# ------------------------------------------------------------
# 1) Streamlit 기반 채팅 UI
#    - 여러 채팅방(chats) 관리
#    - 현재 채팅(active_chat_id) 선택 및 메시지 렌더링
# 2) RAG 파이프라인 호출
#    - generate_answer(user_input, history) 형태로 LLM 답변 생성
# 3) 파일 기반 지식 소스 관리
#    - 사용자가 업로드한 PDF는 DATA_DIR(기본 ./data)에 저장
#    - 뉴스/논문 텍스트(또는 상태 파일)는 NEWS_TEXT_DIR/STATE_DIR에 저장
# 4) 백그라운드 워커(Threading)
#    - PdfWatcher: DATA_DIR 변경 감지 및 임베딩
#    - NewsCrawlerEmbedder: 뉴스 크롤링 및 텍스트 임베딩
#    - PaperFetcherWorker: 논문(예: Semantic Scholar 등) 수집 및 PDF 다운로드
#
# [정리된 원칙]
# - 워커 시작은 start_background_workers_once() 하나로 통일합니다.
# - Streamlit rerun 특성상 워커 스레드가 중복 생성되지 않도록 session_state에 thread 핸들을 저장합니다.
# - "처음 Streamlit 실행 시" 즉시 임베딩/크롤링이 수행되지 않도록, 각 워커의 첫 run_once 호출을 1회 스킵합니다.
# - import/config 중복을 제거하여 유지보수성을 높입니다.
# ------------------------------------------------------------
from __future__ import annotations

import os
import time
import uuid
from typing import Callable

import streamlit as st
import config

# ------------------------------------------------------------
# config.py에서 가져오는 설정값들
# - 폴더 경로, 주기(초), 앱 타이틀 등
# ------------------------------------------------------------
from config import (
    APP_TITLE,
    DATA_DIR,
    NEWS_TEXT_DIR,
    LOG_DIR,
    STATE_DIR,
    PDF_WATCH_INTERVAL_SECONDS,
    NEWS_CRAWL_INTERVAL_SECONDS,
)

from trag.logging_utils import get_logger
from trag.vectorstore_factory import get_vectorstore
from trag.rag_pipeline import generate_answer

from trag.workers.base_worker import BaseWorker
from trag.workers.pdf_watcher_worker import PdfWatcher
from trag.workers.news_crawl_worker import NewsCrawlerEmbedder
from trag.workers.paper_fetch_worker import PaperFetcherWorker
from trag.workers.scorpus_search_worker import ScopusSearchWorker

# ------------------------------------------------------------
# 로깅
# - get_logger("app", LOG_DIR)가 FileHandler를 붙이면 ./logs 아래에 기록됩니다.
# - 실제 파일 저장 여부는 trag.logging_utils.get_logger 구현에 따라 달라질 수 있습니다.
# ------------------------------------------------------------
logger = get_logger("app", LOG_DIR)


def ensure_dirs():
    # 실행에 필요한 폴더를 미리 생성합니다.
    # exist_ok=True 이므로 이미 있으면 그대로 둡니다.
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NEWS_TEXT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)


def init_app_state():
    # Streamlit은 스크립트를 rerun(재실행)하지만 st.session_state는 유지됩니다.
    # 따라서 "없을 때만" 초기화해 중복 생성/초기화를 방지합니다.

    if "chats" not in st.session_state:
        # 채팅방 목록: chat_id -> {title, messages, created_at}
        st.session_state.chats = {}

    if "active_chat_id" not in st.session_state:
        # 현재 활성 채팅방(화면에 표시되는 채팅방)
        cid = create_new_chat()
        st.session_state.active_chat_id = cid

    # 워커 인스턴스(세션 내 싱글톤처럼 사용)
    if "pdf_watcher" not in st.session_state:
        st.session_state.pdf_watcher = PdfWatcher()

    if "news_embedder" not in st.session_state:
        st.session_state.news_embedder = NewsCrawlerEmbedder()

    if "paper_fetcher" not in st.session_state:
        st.session_state.paper_fetcher = PaperFetcherWorker()

    if "scopus_searcher" not in st.session_state:
        # Scopus(Elsevier) 검색 워커: 키워드 기반 논문 메타데이터 수집
        st.session_state.scopus_searcher = ScopusSearchWorker()

    # 워커 스레드 핸들 저장용(중복실행 방지 목적)
    if "workers" not in st.session_state:
        st.session_state.workers = {}  # name -> BaseWorker(Thread)


def create_new_chat() -> str:
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],  # list of {"role": "user"/"assistant", "content": "..."}
        "created_at": time.time(),
    }
    return cid


def set_chat_title_if_needed(chat_id: str):
    chat = st.session_state.chats[chat_id]
    if chat["title"].startswith("Chat ") and chat["messages"]:
        first_user = next((m for m in chat["messages"] if m["role"] == "user"), None)
        if first_user:
            t = first_user["content"].strip()
            chat["title"] = (t[:24] + "...") if len(t) > 24 else t


def _skip_first_run(fn: Callable[[], None], label: str) -> Callable[[], None]:
    """BaseWorker는 thread 시작 직후 run_once_fn()을 1회 호출합니다.

    요구사항: "처음 Streamlit 실행될 때는 임베딩/크롤링을 하지 말아달라"를 만족하기 위해
    첫 호출을 1회 no-op으로 스킵하는 wrapper를 제공합니다.

    - 이 wrapper는 thread마다 독립적으로 동작합니다.
    - 사용자가 업로드 직후 즉시 임베딩을 원해 수동 run_once를 호출하는 경우는 그대로 허용됩니다.
    """
    first = {"v": True}

    def _wrapped():
        if first["v"]:
            first["v"] = False
            try:
                logger.info(f"[{label}] first run skipped (avoid startup embed/crawl)")
            except Exception:
                pass
            return
        fn()

    return _wrapped


def start_background_workers_once():
    """워커 시작 방식을 이 함수 하나로 통일합니다.

    - Streamlit rerun에도 중복 스레드 생성 방지: st.session_state.workers에 핸들 저장
    - 워커 인스턴스는 init_app_state()에서 만든 객체를 재사용
    - thread 시작 직후 1회 호출되는 run_once를 스킵하여 "앱 시작 즉시" 임베딩/크롤링 방지
    """
    workers = st.session_state.workers

    # -------------------------
    # Paper Fetch Worker
    # -------------------------
    if "paper_fetch" not in workers or not workers["paper_fetch"].is_alive():
        paper_worker = st.session_state.paper_fetcher
        t = BaseWorker(
            name="PaperFetchWorker",
            interval_seconds=int(getattr(config, "PAPER_FETCH_INTERVAL_SECONDS", 600)),
            run_once_fn=_skip_first_run(paper_worker.run_once, "PaperFetchWorker"),
            daemon=True,
        )
        t.start()
        workers["paper_fetch"] = t

    # -------------------------
    # PDF Watcher Worker
    # -------------------------
    if "pdf_watch" not in workers or not workers["pdf_watch"].is_alive():
        pdf_worker = st.session_state.pdf_watcher
        t = BaseWorker(
            name="PdfWatcherWorker",
            interval_seconds=int(getattr(config, "PDF_WATCH_INTERVAL_SECONDS", PDF_WATCH_INTERVAL_SECONDS)),
            run_once_fn=_skip_first_run(pdf_worker.run_once, "PdfWatcherWorker"),
            daemon=True,
        )
        t.start()
        workers["pdf_watch"] = t

    # -------------------------
    # News Crawl + Embed Worker
    # -------------------------
    if "news_crawl" not in workers or not workers["news_crawl"].is_alive():
        news_worker = st.session_state.news_embedder
        t = BaseWorker(
            name="NewsCrawlerWorker",
            interval_seconds=int(getattr(config, "NEWS_CRAWL_INTERVAL_SECONDS", NEWS_CRAWL_INTERVAL_SECONDS)),
            run_once_fn=_skip_first_run(news_worker.run_once, "NewsCrawlerWorker"),
            daemon=True,
        )
        t.start()
        workers["news_crawl"] = t

    # -------------------------
    # Scopus Search Worker (metadata search)
    # -------------------------
    if "scopus_search" not in workers or not workers["scopus_search"].is_alive():
        scopus_worker = st.session_state.scopus_searcher
        t = BaseWorker(
            name="ScopusSearchWorker",
            interval_seconds=int(getattr(config, "SCOPUS_SEARCH_INTERVAL_SECONDS", 600)),
            run_once_fn=_skip_first_run(scopus_worker.run_once, "ScopusSearchWorker"),
            daemon=True,
        )
        t.start()
        workers["scopus_search"] = t


def build_langchain_history(chat_messages):
    """(16) 현재 채팅창 종료 전까지 맥락 유지

    - (요구사항) LLM 답변을 벡터DB에 넣지 않음
    - Streamlit 세션 내에서는 chat_messages가 유지되므로 history로 전달하여 대화 일관성을 높입니다.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    history = []
    for m in chat_messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


# -------------------- UI --------------------
ensure_dirs()
st.set_page_config(page_title=APP_TITLE, layout="wide")
init_app_state()
start_background_workers_once()

st.title(APP_TITLE)

# Sidebar: chat list + new chat
with st.sidebar:
    st.header("Chats")
    if st.button("➕ New chat", use_container_width=True):
        new_id = create_new_chat()
        st.session_state.active_chat_id = new_id
        st.rerun()

    # 기존 채팅 목록 (18)
    for cid, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]["created_at"], reverse=True):
        label = chat["title"]
        if st.button(label, key=f"chat_btn_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.rerun()

    st.divider()
    st.subheader("VectorDB Status")
    # VectorStore 초기화는 stats 조회를 위해 수행될 수 있으나, 임베딩(add_documents)은 하지 않습니다.
    st.write(get_vectorstore().stats())

    st.subheader("Workers")
    st.write({k: (v.is_alive() if v else False) for k, v in st.session_state.workers.items()})

    # ------------------------------------------------------------
    # PDF 업로드
    # - 업로드된 파일을 DATA_DIR(기본 ./data)에 저장
    # - 저장 직후 pdf_watcher.run_once()를 호출하여 즉시 반영(현재 구현)
    #   -> 운영상 원한다면 체크박스 옵션으로 즉시 임베딩을 켜/끄도록 바꾸는 것이 일반적입니다.
    # ------------------------------------------------------------
    st.subheader("Upload PDF")
    up = st.file_uploader("Upload a PDF to ./data", type=["pdf"])
    if up is not None:
        save_path = os.path.join(DATA_DIR, up.name)
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
        st.success(f"Saved: {save_path}")

        # 업로드 직후 즉시 임베딩(현재 구현: 항상 실행)
        # PdfWatcher가 '변경 없음'으로 판단하면 내부에서 스킵될 수 있습니다.
        try:
            st.session_state.pdf_watcher.run_once()
        except Exception as e:
            logger.exception(f"Immediate PDF embed failed err={e}")


# Main chat area
active_id = st.session_state.active_chat_id
chat = st.session_state.chats[active_id]

# ------------------------------------------------------------
# 채팅 UI 렌더링
# - chat["messages"]에 누적된 내용을 화면에 다시 그립니다.
# - Streamlit은 rerun 방식이므로, 매 실행마다 전체 히스토리를 재출력합니다.
# ------------------------------------------------------------
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    # 사용자 메시지 저장
    chat["messages"].append({"role": "user", "content": user_input})
    set_chat_title_if_needed(active_id)

    with st.chat_message("user"):
        st.write(user_input)

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 대화 맥락(history)을 LangChain 메시지 타입으로 변환합니다.
                # (요구사항) LLM 답변은 벡터DB에 임베딩하지 않고, 세션 메모리로만 유지합니다.
                history = build_langchain_history(chat["messages"][:-1])  # 마지막 user 입력은 LLM에 별도 전달
                answer = generate_answer(user_input, history)
            except Exception as e:
                logger.exception(f"Answer generation failed err={e}")
                answer = f"Error: {e}"

            st.write(answer)

    chat["messages"].append({"role": "assistant", "content": answer})