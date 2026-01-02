# trag/workers/pdf_watcher_worker.py
from __future__ import annotations

import os
import sys
import argparse
import time
from typing import List

# Allow running this module directly (or via Streamlit from an arbitrary CWD)
# by ensuring the project root (where `config.py` lives) is on sys.path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, LOG_DIR, STATE_DIR
from trag.logging_utils import get_logger
from trag.file_manifest import ManifestStore
from trag.vectorstore_factory import get_vectorstore

logger = get_logger("pdf_worker", LOG_DIR)


def _load_pdf_docs(pdf_path: str):
    """Load PDF pages as Documents."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception:
        from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    return loader.load()


def _split_docs(docs):
    """Split Documents into chunks."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


class PdfWatcher:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(STATE_DIR, exist_ok=True)
        self.manifest = ManifestStore(STATE_DIR)

    def scan_pdf_files(self) -> List[str]:
        """List PDFs under DATA_DIR."""
        try:
            names = os.listdir(DATA_DIR)
        except FileNotFoundError:
            return []

        pdfs: List[str] = []
        for fn in names:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(DATA_DIR, fn))
        return sorted(pdfs)

    def run_once(self) -> None:
        """Embed only changed/new PDFs into the vector DB."""
        logger.info(
            f"run_once start | DATA_DIR={os.path.abspath(DATA_DIR)} STATE_DIR={os.path.abspath(STATE_DIR)} "
            f"chunk={CHUNK_SIZE} overlap={CHUNK_OVERLAP}"
        )

        # Vector store init can fail if Ollama/Chroma deps are missing.
        try:
            vsm = get_vectorstore()
        except Exception as e:
            logger.exception(f"VectorStore init failed: {e}")
            return

        pdf_files = self.scan_pdf_files()
        if not pdf_files:
            logger.info(f"No PDFs found in {os.path.abspath(DATA_DIR)}")
            return

        updated_any = False
        for pdf_path in pdf_files:
            try:
                if not os.path.exists(pdf_path):
                    continue

                if not self.manifest.pdf_is_changed(pdf_path):
                    logger.info(f"Unchanged, skip: {pdf_path}")
                    continue

                logger.info(f"PDF changed/new detected: {pdf_path}")

                docs = _load_pdf_docs(pdf_path)
                if not docs:
                    logger.warning(f"PDF loader returned no docs: {pdf_path}")
                    # Still update manifest to avoid repeated attempts on broken files? -> NO.
                    continue

                # metadata 보강
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata.update({
                        "source_type": "pdf",
                        "source_path": pdf_path,
                    })

                chunks = _split_docs(docs)
                logger.info(
                    f"Split into chunks: {len(chunks)} (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
                )

                if not chunks:
                    logger.warning(f"Splitter returned 0 chunks: {pdf_path}")
                    continue

                vsm.add_documents(chunks)
                self.manifest.pdf_update(pdf_path)

                updated_any = True
            except Exception as e:
                logger.exception(f"Failed to embed PDF: {pdf_path} | err={e}")

        if updated_any:
            st = vsm.stats()
            logger.info(f"[VectorDB Updated] stats={st}")
        else:
            logger.info("No PDF updates detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot / loop test for PdfWatcher")
    parser.add_argument("--once", action="store_true", help="Run a single scan+embed")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="If >0 and not --once, loop with this interval (seconds)",
    )
    args = parser.parse_args()

    w = PdfWatcher()

    if args.once or args.interval <= 0:
        w.run_once()
        print("[pdf_watcher_worker] done")
    else:
        print(f"[pdf_watcher_worker] loop start interval={args.interval}s")
        while True:
            w.run_once()
            time.sleep(args.interval)