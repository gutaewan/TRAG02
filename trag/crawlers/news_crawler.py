# trag/crawlers/news_crawler.py

import os
import re
import sys
import requests
from typing import List, Dict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    NEWS_TEXT_DIR, NEWS_USER_AGENT,
    GOOGLE_NEWS_MAX_RESULTS, NAVER_NEWS_MAX_RESULTS,
    NEWS_KEYWORDS, STATE_DIR, LOG_DIR
)
from trag.logging_utils import get_logger, now_ts
from trag.file_manifest import ManifestStore
from trag.crawlers.google_news_crawler import fetch_google_news_rss
from trag.crawlers.naver_news_crawler import fetch_naver_news_search

logger = get_logger("news_crawler", LOG_DIR)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-가-힣 ]+", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:60] if len(s) > 60 else s

def _clean_html_to_text(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&quot;|&#39;", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _fetch_article_text(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    return _clean_html_to_text(resp.text)

def crawl_news_once(keywords: List[str] | None = None) -> List[Dict]:
    keywords = keywords or NEWS_KEYWORDS
    all_items: List[Dict] = []
    for kw in keywords:
        try:
            g = fetch_google_news_rss(kw, GOOGLE_NEWS_MAX_RESULTS, NEWS_USER_AGENT)
            logger.info(f"Google RSS kw='{kw}' items={len(g)}")
            all_items.extend(g)
        except Exception as e:
            logger.exception(f"Google crawl failed kw='{kw}' err={e}")

        try:
            n = fetch_naver_news_search(kw, NAVER_NEWS_MAX_RESULTS, NEWS_USER_AGENT)
            logger.info(f"Naver search kw='{kw}' items={len(n)}")
            all_items.extend(n)
        except Exception as e:
            logger.exception(f"Naver crawl failed kw='{kw}' err={e}")

    uniq: Dict[str, Dict] = {}
    for it in all_items:
        url = (it.get("url") or "").strip()
        if url and url not in uniq:
            uniq[url] = it

    logger.info(f"crawl_news_once total={len(all_items)} unique={len(uniq)}")
    return list(uniq.values())

def sync_news_dir(keywords: List[str] | None = None) -> List[str]:
    ensure_dir(NEWS_TEXT_DIR)
    ms = ManifestStore(STATE_DIR)

    items = crawl_news_once(keywords=keywords)
    if not items:
        logger.warning("No items returned. Check network/blocks/bs4/selector.")
        return []

    new_files: List[str] = []
    dup = 0

    for it in items:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        if not url:
            continue

        if ms.news_is_duplicate_url(url):
            dup += 1
            continue

        safe = _sanitize_filename(title if title else "news")
        path = os.path.join(NEWS_TEXT_DIR, f"{now_ts()}_{safe}.txt")

        body = ""
        try:
            body = _fetch_article_text(url)
        except Exception as e:
            logger.warning(f"Body fetch failed url={url} err={e}")

        body = (body or "").strip()
        is_minimal = len(body) < 400

        content = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"SOURCE: {it.get('source','')}\n"
            f"QUERY: {it.get('query','')}\n"
            f"EXTRACTION: {'minimal' if is_minimal else 'fullish'}\n\n"
        )
        content += (body + "\n") if body else "(본문 추출 실패/짧음. URL을 직접 확인해 주세요.)\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        ms.news_add(title=title, url=url, saved_path=path)
        new_files.append(path)

    logger.info(f"sync_news_dir done: new_files={len(new_files)} dup_skipped={dup}")
    return new_files