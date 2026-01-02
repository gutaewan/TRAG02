# trag/logging_utils.py
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_logger(name: str, log_dir: str) -> logging.Logger:
    ensure_dir(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"{name}.log")

    handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")