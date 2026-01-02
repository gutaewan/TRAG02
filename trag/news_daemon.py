import os
import time
import logging

from .vectorstore import get_vectorstore, sync_pdf_dir, sync_news_dir, log_vectordb_state
from .news_crawler import crawl_news_once
from .storage import file_lock


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker.log")
    logger = logging.getLogger("trag.worker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_config() -> dict:
    """Load ./config/config.py (same single source of truth as UI)."""
    import importlib.util

    CONFIG_DIR = "./config"
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.py")

    DEFAULTS = {
        "DATA_DIR": "./data",
        "NEWS_DIR": "./news_texts",
        "LOG_DIR": "./logs",
        "VECTOR_DB_ROOT": "./vector_db",
        "VECTOR_DB": "chroma",
        "LLM_MODEL": "llama3.2",
        "EMBED_MODEL": "qwen3-embedding",
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 200,
        "PDF_SYNC_INTERVAL_SEC": 600,
        "NEWS_CRAWL_INTERVAL_SEC": 600,
        "NEWS_KEYWORDS": ["소프트웨어 공학", "AI 안전", "자동차 기능안전", "SDV"],
        "NEWS_MAX_ITEMS_PER_KEYWORD": 10,
        "NEWS_TIMEOUT_SEC": 20,
    }

    cfg = dict(DEFAULTS)
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if os.path.exists(CONFIG_FILE):
            spec = importlib.util.spec_from_file_location("trag_user_config", CONFIG_FILE)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                for k in list(DEFAULTS.keys()):
                    if hasattr(mod, k):
                        cfg[k] = getattr(mod, k)
    except Exception as e:
        print(f"[WARN] Failed to load config.py: {e}")

    os.makedirs(str(cfg["DATA_DIR"]), exist_ok=True)
    os.makedirs(str(cfg["NEWS_DIR"]), exist_ok=True)
    os.makedirs(str(cfg["LOG_DIR"]), exist_ok=True)
    os.makedirs(str(cfg["VECTOR_DB_ROOT"]), exist_ok=True)

    if isinstance(cfg.get("NEWS_KEYWORDS"), tuple):
        cfg["NEWS_KEYWORDS"] = list(cfg["NEWS_KEYWORDS"])

    return cfg


def run_once(CFG: dict, logger: logging.Logger) -> None:
    lock_path = os.path.join(str(CFG.get("VECTOR_DB_ROOT", "./vector_db")), ".writer.lock")

    # ✅ 단일 writer 보장 (PDF/NEWS 임베딩/동기화는 모두 이 락 아래에서 수행)
    with file_lock(lock_path, timeout_sec=180):
        vs = get_vectorstore(CFG, CFG["EMBED_MODEL"], logger)

        added_pdf = 0
        added_news = 0
        created_news_files = []

        try:
            added_pdf = sync_pdf_dir(vs, CFG, CFG["EMBED_MODEL"], logger)
        except Exception as e:
            logger.info(f"[PDF] sync failed: {e}")

        try:
            created_news_files = crawl_news_once(CFG, logger)
        except Exception as e:
            logger.info(f"[NEWS] crawl failed: {e}")

        try:
            added_news = sync_news_dir(vs, CFG, CFG["EMBED_MODEL"], logger)
        except Exception as e:
            logger.info(f"[NEWS] sync failed: {e}")

        if added_pdf or added_news:
            logger.info(
                f"[WORKER] updated vdb: pdf_added_chunks={added_pdf} "
                f"news_added_chunks={added_news} news_created_files={len(created_news_files)}"
            )
            log_vectordb_state(vs, logger)
        else:
            logger.info(f"[WORKER] no changes: news_created_files={len(created_news_files)}")


def loop(CFG: dict) -> None:
    logger = setup_logger(str(CFG.get("LOG_DIR", "./logs")))

    pdf_interval = int(CFG.get("PDF_SYNC_INTERVAL_SEC", 600))
    news_interval = int(CFG.get("NEWS_CRAWL_INTERVAL_SEC", 600))
    tick = max(10, min(pdf_interval, news_interval))

    logger.info(
        f"[WORKER] start. tick_sec={tick} embed={CFG.get('EMBED_MODEL')} vdb={CFG.get('VECTOR_DB')}"
    )

    while True:
        try:
            run_once(CFG, logger)
        except Exception as e:
            logger.info(f"[WORKER] run_once crashed: {e}")
        time.sleep(tick)


if __name__ == "__main__":
    CFG = load_config()
    loop(CFG)