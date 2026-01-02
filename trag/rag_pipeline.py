# trag/rag_pipeline.py
from __future__ import annotations

from typing import List, Optional, Tuple

from config import SYSTEM_PROMPT, TOP_K, MAX_CONTEXT_CHARS, OLLAMA_BASE_URL, LLM_MODEL
from trag.vectorstore_factory import get_vectorstore

def _load_chat_ollama():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

def build_context(query: str) -> Tuple[str, List[dict]]:
    """
    벡터DB에서 관련 문서 조각을 가져와 컨텍스트 문자열을 만든다.
    (7) 검색 결과로 질의를 업데이트하는 역할
    """
    vsm = get_vectorstore()
    st = vsm.stats()
    if not st.get("count"):
        return "", []

    docs = vsm.similarity_search(query, k=TOP_K)
    chunks = []
    meta_list = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        chunks.append(txt)
        meta_list.append(d.metadata or {})

    context = "\n\n---\n\n".join(chunks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...(truncated)"
    return context, meta_list

def generate_answer(user_query: str, chat_history_messages) -> str:
    """
    (8) 벡터검색으로 업데이트된 질의를 LLM에 넣어 답변 생성
    (9) 벡터DB가 비어있으면(=pdf/news 임베딩 없음) 질의만으로 답변
    (16) 채팅 종료 전까지는 맥락 유지 -> chat_history_messages를 그대로 넣음
    """
    llm = _load_chat_ollama()

    # LangChain 메시지 타입
    from langchain_core.messages import SystemMessage, HumanMessage

    context, _ = build_context(user_query)

    if context:
        augmented = (
            f"[CONTEXT]\n{context}\n\n"
            f"[USER_QUESTION]\n{user_query}\n\n"
            "Answer using the context when relevant. "
            "If the context is insufficient, say so and answer carefully."
            "All results must be in English at the first"
            "Then, translate the answer to Korean."
            "If you cannot answer, say you don't know."            
        )
    else:
        augmented = user_query

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history_messages)
    messages.append(HumanMessage(content=augmented))

    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))