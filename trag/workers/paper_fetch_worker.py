# trag/workers/paper_fetch_worker.py
from __future__ import annotations

import os
import sys
import json
import time
import re
import hashlib
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests

# Ensure project root (where config.py lives) is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from trag.logging_utils import get_logger, now_ts

logger = get_logger("paper_worker", config.LOG_DIR)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def _has_hangul(s: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in s)


def _safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-가-힣\. ]+", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] if len(s) > max_len else s


class _JsonKVStore:
    """Simple JSON-backed dict store under STATE_DIR."""
    def __init__(self, path: str):
        self.path = path
        self.data: Dict = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def save(self) -> None:
        _ensure_dir(os.path.dirname(self.path))
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


def _ollama_translate_to_en(text: str) -> str:
    """
    Translate Korean keyword to English using Ollama (local).
    Returns a short English query phrase.
    """
    url = config.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
    prompt = (
        "Translate the following Korean search keyword into concise English search keywords. "
        "Return ONLY the English keywords (no quotes, no explanation).\n\n"
        f"Korean: {text}\nEnglish:"
    )
    payload = {
        "model": getattr(config, "PAPER_TRANSLATION_MODEL", "llama3.2"),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    j = r.json()
    out = (j.get("response") or "").strip()
    # sanitize: one line
    out = re.sub(r"[\r\n]+", " ", out).strip()
    # remove wrapping quotes if any
    out = out.strip("\"'").strip()
    return out


class PaperFetcherWorker:
    """
    Worker:
    - Uses NEWS_KEYWORDS as shared keywords
    - Auto-translates KO keywords to EN (cached)
    - Searches Semantic Scholar
    - Downloads Open Access PDFs into DATA_DIR
    - Leaves embedding to PdfWatcher (change detection)
    """

    def __init__(self):
        _ensure_dir(config.DATA_DIR)
        _ensure_dir(config.STATE_DIR)
        self.translation_cache = _JsonKVStore(
            os.path.join(config.STATE_DIR, getattr(config, "PAPER_TRANSLATION_CACHE_FILE", "paper_translation_cache.json"))
        )
        self.paper_manifest = _JsonKVStore(
            os.path.join(config.STATE_DIR, getattr(config, "PAPER_MANIFEST_FILE", "paper_manifest.json"))
        )
        # manifest schema:
        # {
        #   "downloaded": { "<paper_key>": {"pdf_path": "...", "ts": "...", "title": "...", "pdf_url": "..."} }
        # }
        if "downloaded" not in self.paper_manifest.data:
            self.paper_manifest.data["downloaded"] = {}

    def _translate_keyword(self, kw: str) -> str:
        kw = (kw or "").strip()
        if not kw:
            return kw
        if _is_ascii(kw) and not _has_hangul(kw):
            return kw

        if not getattr(config, "PAPER_AUTO_TRANSLATE_TO_EN", True):
            return kw

        cache = self.translation_cache.data
        if kw in cache and isinstance(cache[kw], str) and cache[kw].strip():
            return cache[kw].strip()

        translated = kw
        try:
            if getattr(config, "PAPER_TRANSLATE_WITH_OLLAMA", True) and _has_hangul(kw):
                translated = _ollama_translate_to_en(kw)
        except Exception as e:
            logger.warning(f"translate failed kw='{kw}' err={e}")

        # very last fallback: keep original
        if not translated:
            translated = kw

        cache[kw] = translated
        self.translation_cache.save()
        logger.info(f"keyword translated: '{kw}' -> '{translated}'")
        return translated

    def _paper_key(self, paper: Dict) -> str:
        """
        Stable unique key for dedup:
        Prefer DOI, else Semantic Scholar paperId, else hash(pdf_url/title).
        """
        ext = paper.get("externalIds") or {}
        doi = (ext.get("DOI") or "").strip()
        if doi:
            return "doi:" + doi.lower()

        pid = (paper.get("paperId") or "").strip()
        if pid:
            return "ssid:" + pid

        pdf_url = ""
        oapdf = paper.get("openAccessPdf") or {}
        if isinstance(oapdf, dict):
            pdf_url = (oapdf.get("url") or "").strip()

        title = (paper.get("title") or "").strip()
        h = hashlib.sha1((pdf_url + "|" + title).encode("utf-8", errors="ignore")).hexdigest()
        return "hash:" + h

    def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        params = {
            "query": query,
            "limit": max(1, min(int(limit), 100)),
            "fields": getattr(config, "SEMANTIC_SCHOLAR_FIELDS", "title,year,authors,externalIds,isOpenAccess,openAccessPdf,url"),
        }
        url = getattr(config, "SEMANTIC_SCHOLAR_SEARCH_URL", "https://api.semanticscholar.org/graph/v1/paper/search")
        headers = {"User-Agent": getattr(config, "PAPER_USER_AGENT", getattr(config, "NEWS_USER_AGENT", "TRAG02/1.0"))}

        r = requests.get(url, params=params, headers=headers, timeout=getattr(config, "PAPER_REQUEST_TIMEOUT_SECONDS", 20))
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if not isinstance(data, list):
            return []
        return data

    def _download_pdf(self, pdf_url: str, title: str, paper_key: str) -> Optional[str]:
        if not pdf_url:
            return None

        # file name: timestamp + title + hash
        base = _safe_filename(title if title else "paper")
        suffix = hashlib.sha1(paper_key.encode("utf-8")).hexdigest()[:10]
        filename = f"{now_ts()}_{base}_{suffix}.pdf"
        dst = os.path.join(config.DATA_DIR, filename)

        headers = {"User-Agent": getattr(config, "PAPER_USER_AGENT", getattr(config, "NEWS_USER_AGENT", "TRAG02/1.0"))}
        timeout = getattr(config, "PAPER_REQUEST_TIMEOUT_SECONDS", 20)

        r = requests.get(pdf_url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not pdf_url.lower().endswith(".pdf"):
            logger.warning(f"downloaded content-type not pdf: url={pdf_url} content-type={ctype}")

        with open(dst, "wb") as f:
            f.write(r.content)

        # sanity check: small file guard
        if os.path.getsize(dst) < 10_000:
            logger.warning(f"pdf too small, maybe blocked: {dst} size={os.path.getsize(dst)} url={pdf_url}")

        return dst

    def run_once(self) -> None:
        """
        One iteration:
        - translate shared keywords
        - search OA pdf
        - download new pdfs into DATA_DIR
        - update manifest
        """
        kws = getattr(config, "NEWS_KEYWORDS", [])
        if not kws:
            logger.info("No NEWS_KEYWORDS configured. Paper fetch skipped.")
            return

        logger.info(
            f"run_once start | DATA_DIR={os.path.abspath(config.DATA_DIR)} "
            f"keywords={len(kws)} interval={getattr(config, 'PAPER_FETCH_INTERVAL_SECONDS', 600)}"
        )

        max_results = getattr(config, "PAPER_MAX_RESULTS_PER_KEYWORD", 5)
        max_dl = getattr(config, "PAPER_MAX_DOWNLOADS_PER_RUN", 5)

        downloaded_any = 0
        downloaded_map: Dict[str, Dict] = self.paper_manifest.data.get("downloaded", {})

        for kw in kws:
            if downloaded_any >= max_dl:
                break

            kw_en = self._translate_keyword(kw)

            try:
                papers = self._search_semantic_scholar(kw_en, max_results)
                logger.info(f"semantic_scholar query='{kw_en}' results={len(papers)}")
            except Exception as e:
                logger.exception(f"semantic_scholar search failed query='{kw_en}' err={e}")
                continue

            for p in papers:
                if downloaded_any >= max_dl:
                    break

                # Only Open Access PDF
                oapdf = p.get("openAccessPdf") or {}
                pdf_url = ""
                if isinstance(oapdf, dict):
                    pdf_url = (oapdf.get("url") or "").strip()
                if not pdf_url:
                    continue

                key = self._paper_key(p)
                if key in downloaded_map:
                    continue

                title = (p.get("title") or "").strip()
                try:
                    path = self._download_pdf(pdf_url, title, key)
                    if not path:
                        continue

                    downloaded_map[key] = {
                        "pdf_path": path,
                        "ts": now_ts(),
                        "title": title,
                        "pdf_url": pdf_url,
                        "query": kw_en,
                        "query_orig": kw,
                        "source": "semantic_scholar",
                    }
                    self.paper_manifest.data["downloaded"] = downloaded_map
                    self.paper_manifest.save()

                    downloaded_any += 1
                    logger.info(f"[PDF Saved] {path} | title='{title[:80]}'")

                    # polite delay to avoid hammering
                    time.sleep(0.5)

                except Exception as e:
                    logger.exception(f"pdf download failed url={pdf_url} err={e}")

        if downloaded_any > 0:
            logger.info(f"[Paper Fetch Done] downloaded={downloaded_any} (PDFs saved into DATA_DIR)")
            logger.info("NOTE: PdfWatcher will detect changes and embed PDFs into VectorDB.")
        else:
            logger.info("[Paper Fetch Done] downloaded=0 (no new OA PDFs or all duplicates)")