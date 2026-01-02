# trag/crawlers/google_news_crawler.py
from __future__ import annotations

import urllib.parse
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

def fetch_google_news_rss(query: str, max_results: int, user_agent: str, hl: str = "ko", gl: str = "KR") -> List[Dict]:
    """
    Google News RSS:
    https://news.google.com/rss/search?q=...
    """
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={gl}:{hl}"

    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pubDate = (item.findtext("pubDate") or "").strip()

        if not title or not link:
            continue

        items.append({
            "title": title,
            "url": link,
            "published": pubDate,
            "source": "google_news_rss",
            "query": query,
        })
        if len(items) >= max_results:
            break
    return items