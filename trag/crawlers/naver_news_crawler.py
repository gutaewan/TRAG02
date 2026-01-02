# trag/crawlers/naver_news_crawler.py
from __future__ import annotations

import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import html as _html
import re

import os
import sys
import argparse

# Set NAVER_CRAWLER_DEBUG=1 to verify the module is being imported/executed.
_DEBUG = os.environ.get("NAVER_CRAWLER_DEBUG", "").strip() == "1"
if _DEBUG:
    print("[naver_news_crawler] module imported", file=sys.stderr, flush=True)
    print(f"[naver_news_crawler] __file__={__file__}", file=sys.stderr, flush=True)

# Allow running this module directly by ensuring the project root (where `config.py` may live)
# is on sys.path. This does not affect package imports when used normally.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _strip_html_tags(s: str) -> str:
    if not s:
        return ""
    # Naver OpenAPI title/description may contain HTML tags (e.g., <b>).</n    s = _html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_naver_news_openapi(query: str, max_results: int, user_agent: str) -> List[Dict]:
    """Fetch Naver News via Naver Search OpenAPI (JSON).

    Requires NAVER_CLIENT_ID / NAVER_CLIENT_SECRET.
    If not configured, this function raises ValueError.
    """
    # Lazy import to avoid hard dependency when not used.
    try:
        import config as _config  # type: ignore
    except Exception as e:
        raise ValueError("config module not available for OpenAPI mode") from e

    client_id = getattr(_config, "NAVER_CLIENT_ID", "")
    client_secret = getattr(_config, "NAVER_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise ValueError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set in config.py")

    display = max(1, min(int(max_results), 100))
    q = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={q}&display={display}&sort=date"

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "User-Agent": user_agent,
    }

    resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    status = int(getattr(resp, "status_code", 0))
    # Provide a clearer error if blocked or unauthorized.
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict] = []
    for it in data.get("items", []) or []:
        title = _strip_html_tags(str(it.get("title", "")))
        link = str(it.get("link", "")).strip()
        pub = str(it.get("pubDate", "")).strip()
        if not link:
            continue
        results.append({
            "title": title,
            "url": link,
            "published": pub,
            "source": "naver_openapi",
            "query": query,
            "status": status,
        })
        if len(results) >= display:
            break

    return results


def _fetch_raw_html(query: str, user_agent: str, timeout_seconds: int = 20) -> tuple[str, str, int]:
    q = urllib.parse.quote(query)
    url = f"https://search.naver.com/search.naver?where=news&query={q}"

    resp = requests.get(
        url,
        headers={"User-Agent": user_agent},
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    status = int(getattr(resp, "status_code", 0))
    final_url = str(getattr(resp, "url", url))
    resp.raise_for_status()
    return resp.text, final_url, status


def fetch_naver_news_search(query: str, max_results: int, user_agent: str) -> List[Dict]:
    """
    네이버 뉴스 검색 결과 페이지를 파싱합니다.
    (사이트 정책/구조 변경에 따라 selector 조정이 필요할 수 있음)
    """
    # OpenAPI mode: prefer Naver Search OpenAPI when enabled in config.
    # This avoids issues where Naver search HTML is rendered dynamically in the browser.
    use_openapi = False
    try:
        import config as _config  # type: ignore
        use_openapi = bool(getattr(_config, "NAVER_USE_OPENAPI", False))
    except Exception:
        use_openapi = False

    if use_openapi:
        try:
            return fetch_naver_news_openapi(query, max_results if max_results > 0 else 10, user_agent)
        except Exception as e:
            # Fallback to HTML parsing if OpenAPI is misconfigured or fails.
            if max_results is not None and max_results <= 0:
                print(f"[naver_news_crawler diagnostics] OpenAPI failed, falling back to HTML: {e}")

    # When max_results <= 0, enable diagnostics mode without changing the public API.
    diagnostics = max_results is not None and max_results <= 0
    if diagnostics:
        max_results = 10

    html, final_url, status = _fetch_raw_html(query, user_agent)
    soup = BeautifulSoup(html, "html.parser")

    if diagnostics:
        print("[naver_news_crawler diagnostics]")
        print(f"status={status}")
        print(f"final_url={final_url}")
        print(f"html_len={len(html)}")
        print(f"count a.news_tit={len(soup.select('a.news_tit'))}")
        print(f"count a.link_tit={len(soup.select('a.link_tit'))}")
        print(f"count a.info={len(soup.select('a.info'))}")

    results: List[Dict] = []
    # 뉴스 결과 영역: a.info / a.news_tit 등 다양한 패턴 존재
    for a in soup.select("a.news_tit"):
        title = (a.get("title") or a.get_text() or "").strip()
        link = (a.get("href") or "").strip()
        if not title or not link:
            continue

        results.append({
            "title": title,
            "url": link,
            "published": "",
            "source": "naver_news_search",
            "query": query,
        })
        if len(results) >= max_results:
            break

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for Naver news crawler")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max", type=int, default=5, help="Max results. Use -1 for diagnostics mode.")
    parser.add_argument("--dump", default="", help="Optional path to dump fetched HTML")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    args = parser.parse_args()

    print("[naver_news_crawler] __main__ entered", file=sys.stderr, flush=True)

    print("[naver_news_crawler] started", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    print(f"query={args.query} max={args.max} timeout={args.timeout}", flush=True)


    # Try to load UA from config if available.
    ua = _DEFAULT_UA
    try:
        import config as _config  # type: ignore
        ua = getattr(_config, "NEWS_USER_AGENT", ua)
    except Exception:
        pass

    # Show OpenAPI mode status
    try:
        import config as _config  # type: ignore
        print(f"NAVER_USE_OPENAPI={getattr(_config, 'NAVER_USE_OPENAPI', False)}", flush=True)
    except Exception:
        print("NAVER_USE_OPENAPI=False (config not loaded)", flush=True)

    html_dump = args.dump.strip()
    if html_dump:
        try:
            html, final_url, status = _fetch_raw_html(args.query, ua, timeout_seconds=args.timeout)
            os.makedirs(os.path.dirname(os.path.abspath(html_dump)), exist_ok=True)
            with open(html_dump, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[dumped] {html_dump}")
            print(f"status={status} final_url={final_url} html_len={len(html)}")
        except Exception as e:
            print(f"[dump failed] err={e}")

    try:
        results = fetch_naver_news_search(args.query, args.max, ua)
        print(f"found={len(results)}", flush=True)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.get('title','').strip()}\n   {r.get('url','').strip()}", flush=True)
    except Exception as e:
        print(f"[naver_news_crawler] failed: {e}", flush=True)
        raise