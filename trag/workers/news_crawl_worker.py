# trag/workers/news_crawl_worker.py
from __future__ import annotations

import os
from typing import List

from config import NEWS_TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP, LOG_DIR, STATE_DIR
from trag.logging_utils import get_logger
from trag.vectorstore_factory import get_vectorstore
from trag.crawlers.news_crawler import sync_news_dir

logger = get_logger("news_worker", LOG_DIR)

def _load_text_docs(txt_path: str):
    try:
        from langchain_community.document_loaders import TextLoader
    except Exception:
        from langchain.document_loaders import TextLoader

    loader = TextLoader(txt_path, encoding="utf-8")
    return loader.load()

def _split_docs(docs):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

class NewsCrawlerEmbedder:
    def __init__(self):
        os.makedirs(NEWS_TEXT_DIR, exist_ok=True)

    def run_once(self) -> None:
        vsm = get_vectorstore()

        # (11) 주기적으로 크롤링 수행 -> 새 파일만 반환
        new_files = sync_news_dir()
        if not new_files:
            return

        updated_any = False
        for p in new_files:
            try:
                docs = _load_text_docs(p)
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata.update({
                        "source_type": "news",
                        "source_path": p,
                    })

                chunks = _split_docs(docs)
                logger.info(f"Embed news file={p} chunks={len(chunks)} (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

                vsm.add_documents(chunks)
                updated_any = True

            except Exception as e:
                logger.exception(f"Failed to embed news file={p} err={e}")

        if updated_any:
            st = vsm.stats()
            logger.info(f"[VectorDB Updated] stats={st}")