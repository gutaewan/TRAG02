# trag/workers/scorpus_search_worker.py
from __future__ import annotations

import json
import os
import random
import re
import time
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ReadTimeout, ConnectTimeout, Timeout, HTTPError, ConnectionError as RequestsConnectionError

# config import 안정화 (직접 실행/모듈 실행 모두 대응)
try:
    import config  # repo root의 config.py
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # .../TRAG02
    sys.path.insert(0, str(ROOT))
    import config  # type: ignore

from trag.logging_utils import get_logger
from trag.vectorstore_factory import get_vectorstore

# LangChain Document / TextSplitter (버전 차이를 고려해 fallback import)
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

# VectorStore에 add_documents를 호출할 때(임베딩/추가) 충돌을 줄이기 위한 간단한 락
_VSTORE_LOCK = threading.Lock()


@dataclass
class ScopusItem:
    keyword: str
    title: str
    doi: str
    eid: str
    url: str
    cover_date: str
    source_title: str
    raw: Dict[str, Any]


class ScopusSearchWorker:
    """
    Scopus Search API로 키워드 기반 검색을 수행하고,
    신규 결과를 state 파일로 저장한 뒤, VectorDB에 임베딩까지 수행하는 worker.
    """

    def __init__(self):
        self.state_dir = getattr(config, "STATE_DIR", "./state")
        self.log_dir = getattr(config, "LOG_DIR", "./logs")
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = get_logger("scopus_worker", self.log_dir)

        # Ensure file logging (some environments may create a logger without a FileHandler)
        self.log_path = os.path.join(self.log_dir, "scopus_worker.log")
        self._ensure_file_handler()

        self.seen_path = os.path.join(self.state_dir, "scopus_seen.json")
        self.results_dir = os.path.join(self.state_dir, "scopus_results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.enabled = bool(getattr(config, "SCOPUS_ENABLED", True))

    def _ensure_file_handler(self) -> None:
        """Attach a FileHandler if none exists so execution logs are persisted under ./logs."""
        try:
            for h in getattr(self.logger, "handlers", []):
                if isinstance(h, logging.FileHandler):
                    return

            os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            fh.setFormatter(fmt)

            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
        except Exception:
            pass

    # -------------------------
    # public entrypoint for BaseWorker
    # -------------------------
    def run_once(self) -> None:
        start_ts = time.time()
        self.logger.info("[Scopus] run_once start")

        if not self.enabled:
            self.logger.info("SCOPUS_ENABLED=False. skip.")
            return

        api_key = getattr(config, "SCOPUS_API_KEY", "").strip()
        if not api_key:
            self.logger.warning("SCOPUS_API_KEY is empty. skip Scopus search.")
            return

        keywords = getattr(config, "SCOPUS_KEYWORDS", None) or getattr(config, "NEWS_KEYWORDS", [])
        if not keywords:
            self.logger.info("No keywords configured. skip.")
            return

        try:
            seen = self._load_seen()

            total_new = 0
            for kw in keywords:
                kw = str(kw).strip()
                if not kw:
                    continue

                new_items = self._search_keyword_and_collect_new(kw, seen)
                if new_items:
                    total_new += len(new_items)
                    self._append_results_jsonl(kw, new_items)
                    self.logger.info(f"[Scopus] keyword='{kw}' new_items={len(new_items)}")
                else:
                    self.logger.info(f"[Scopus] keyword='{kw}' new_items=0")

                # 키워드 간 약간의 간격(429/timeout 예방)
                time.sleep(float(getattr(config, "SCOPUS_BETWEEN_KEYWORDS_SLEEP_SECONDS", 0.8)) + random.uniform(0.0, 0.6))

            # ------------------------------------------------------------
            # (추가) Scopus 검색 결과를 VectorDB에 임베딩
            # - 뉴스/PDF와 동일한 chunk_size / chunk_overlap 사용
            # - 이미 seen에 등록된 항목이라도 embedded=False이면 재시도 가능
            # ------------------------------------------------------------
            embed_added = 0
            if bool(getattr(config, "SCOPUS_EMBED_ENABLED", True)):
                try:
                    embed_added = self._embed_pending_seen(seen)
                except Exception as e:
                    self.logger.exception(f"[Scopus] embedding failed err={e}")

                # 임베딩이 실제로 발생했으면 VectorDB 상태 로그
                if embed_added > 0:
                    try:
                        stats = get_vectorstore().stats()
                        self.logger.info(f"[Scopus] VectorStore stats after embed: {stats}")
                    except Exception as e:
                        self.logger.warning(f"[Scopus] failed to read VectorStore stats err={e}")
            else:
                self.logger.info("[Scopus] SCOPUS_EMBED_ENABLED=False. skip embedding")

            # seen 저장
            self._save_seen(seen)

            elapsed = time.time() - start_ts
            self.logger.info(
                f"[Scopus] run_once done. total_new={total_new} embed_chunks={embed_added} elapsed={elapsed:.1f}s"
            )
        except Exception as e:
            self.logger.exception(f"[Scopus] run_once failed err={e}")
            raise

    # -------------------------
    # internal: search
    # -------------------------
    def _search_keyword_and_collect_new(self, keyword: str, seen: Dict[str, Any]) -> List[ScopusItem]:
        max_results = int(getattr(config, "SCOPUS_MAX_RESULTS_PER_KEYWORD", 50))
        count = int(getattr(config, "SCOPUS_COUNT_PER_CALL", 25))
        count = max(1, min(count, 200))

        query_field = getattr(config, "SCOPUS_QUERY_FIELD", "TITLE-ABS-KEY")
        scopus_query = self._build_scopus_query(query_field, keyword)

        view = getattr(config, "SCOPUS_VIEW", "STANDARD")
        sort = getattr(config, "SCOPUS_SORT", "relevancy")

        collected: List[ScopusItem] = []
        start = 0

        # Scopus Search API는 결과 페이징에 실질적인 상한(통상 5000)을 둡니다.
        # 중요한 점: start는 0-based offset이며, start=5000 요청이 400을 유발할 수 있습니다.
        # 따라서 "exclusive" 상한(기본 5000)을 두고 start는 항상 < 상한이 되도록 강제합니다.
        max_start_cfg = int(getattr(config, "SCOPUS_MAX_START", 5000))
        api_max_start_exclusive = int(getattr(config, "SCOPUS_API_MAX_START_EXCLUSIVE", 5000))
        max_start_exclusive = min(max_start_cfg, api_max_start_exclusive)
        max_no_new_pages = int(getattr(config, "SCOPUS_MAX_NO_NEW_PAGES", 5))
        no_new_pages = 0

        while len(collected) < max_results and start < max_start_exclusive and (start + count) <= max_start_exclusive:
            # 안전장치: start가 상한에 도달하면 호출하지 않고 종료
            if start >= max_start_exclusive:
                break
            batch = self._call_scopus_search(scopus_query, start=start, count=count, view=view, sort=sort)
            entries = batch.get("entries", [])
            if not entries:
                break

            before_count = len(collected)

            for e in entries:
                item = self._parse_entry(keyword, e)
                if not item:
                    continue

                dedup_key = self._dedup_key(item)
                if dedup_key in seen:
                    continue

                # 검색 단계에서 중복 방지 상태(seen)에 기록
                # embedded=False로 시작 → 이후 임베딩 성공 시 True로 갱신
                seen[dedup_key] = {
                    "keyword": keyword,
                    "title": item.title,
                    "doi": item.doi,
                    "eid": item.eid,
                    "url": item.url,
                    "cover_date": item.cover_date,
                    "source_title": item.source_title,
                    "ts": time.time(),
                    "embedded": False,
                    "embedded_ts": 0,
                }
                collected.append(item)

                if len(collected) >= max_results:
                    break

            if len(collected) == before_count:
                no_new_pages += 1
                if no_new_pages >= max_no_new_pages:
                    self.logger.info(
                        f"[Scopus] stop paging: {max_no_new_pages} consecutive pages had no new items (keyword={keyword!r})"
                    )
                    break
            else:
                no_new_pages = 0

            start += count
            time.sleep(0.3 + random.uniform(0.0, 0.4))

        if start >= max_start_exclusive and len(collected) < max_results:
            self.logger.info(
                f"[Scopus] stop paging: reached max_start_exclusive={max_start_exclusive} without collecting enough new items (keyword={keyword!r})"
            )

        return collected

    def _call_scopus_search(self, query: str, start: int, count: int, view: str, sort: str) -> Dict[str, Any]:
        url = getattr(config, "SCOPUS_SEARCH_URL", "https://api.elsevier.com/content/search/scopus")
        # timeout은 (connect, read) 튜플 형태도 허용합니다.
        # - SCOPUS_CONNECT_TIMEOUT_SECONDS / SCOPUS_READ_TIMEOUT_SECONDS가 설정되어 있으면 이를 우선 사용
        # - 아니면 기존 SCOPUS_REQUEST_TIMEOUT_SECONDS(단일 read+connect) 사용
        connect_timeout = getattr(config, "SCOPUS_CONNECT_TIMEOUT_SECONDS", None)
        read_timeout = getattr(config, "SCOPUS_READ_TIMEOUT_SECONDS", None)
        if connect_timeout is not None and read_timeout is not None:
            timeout = (float(connect_timeout), float(read_timeout))
        else:
            timeout = float(getattr(config, "SCOPUS_REQUEST_TIMEOUT_SECONDS", 20))

        headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": getattr(config, "SCOPUS_API_KEY", "").strip(),
            "User-Agent": getattr(config, "PAPER_USER_AGENT", getattr(config, "NEWS_USER_AGENT", "TRAG02/1.0")),
        }

        insttoken = getattr(config, "SCOPUS_INSTTOKEN", "").strip()
        if insttoken:
            headers["X-ELS-Insttoken"] = insttoken

        params = {
            "query": query,
            "start": max(0, int(start)),
            "count": max(1, min(int(count), 200)),
            "view": view,
            "sort": sort,
        }

        # 최후의 안전장치: Scopus는 start>=5000 요청에서 400을 반환할 수 있습니다.
        # 어떤 경로로든 start가 상한 이상이면 API 호출을 건너뜁니다.
        api_max_start_exclusive = int(getattr(config, "SCOPUS_API_MAX_START_EXCLUSIVE", 5000))
        if int(params.get("start", 0)) >= api_max_start_exclusive:
            self.logger.warning(
                f"[Scopus] skip API call: start={params.get('start')} >= api_max_start_exclusive={api_max_start_exclusive}"
            )
            return {"raw": {}, "entries": []}

        max_retries = int(getattr(config, "SCOPUS_MAX_RETRIES", 5))
        backoff_base = float(getattr(config, "SCOPUS_BACKOFF_BASE_SECONDS", 2.0))
        backoff_cap = float(getattr(config, "SCOPUS_BACKOFF_CAP_SECONDS", 60.0))

        last_status: Optional[int] = None
        last_text: str = ""

        for attempt in range(max_retries + 1):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
            except (ReadTimeout, ConnectTimeout, Timeout, RequestsConnectionError) as e:
                # 네트워크/타임아웃 계열 오류는 일시적일 수 있으므로 backoff 후 재시도
                wait_s = min(backoff_base * (2 ** attempt), backoff_cap) + random.uniform(0.0, 0.7)
                self.logger.warning(
                    f"[Scopus] request error={type(e).__name__} wait={wait_s:.1f}s attempt={attempt+1}/{max_retries+1} "
                    f"query={params.get('query')!r} start={params.get('start')} count={params.get('count')}"
                )
                time.sleep(wait_s)
                continue
            last_status = r.status_code
            last_text = (r.text or "")[:500]

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_s: Optional[float] = None
                if retry_after:
                    try:
                        wait_s = float(str(retry_after).split(",")[0].strip())
                    except Exception:
                        wait_s = None
                if wait_s is None:
                    wait_s = min(backoff_base * (2 ** attempt), backoff_cap) + random.uniform(0.0, 0.7)

                self.logger.warning(
                    f"[Scopus] 429 throttled. wait={wait_s:.1f}s attempt={attempt+1}/{max_retries+1} query={params['query'][:60]!r}"
                )
                time.sleep(wait_s)
                continue

            if 500 <= r.status_code < 600:
                wait_s = min(backoff_base * (2 ** attempt), backoff_cap) + random.uniform(0.0, 0.7)
                self.logger.warning(
                    f"[Scopus] {r.status_code} server error. wait={wait_s:.1f}s attempt={attempt+1}/{max_retries+1}"
                )
                time.sleep(wait_s)
                continue

            # 4xx는 잘못된 요청/권한/페이징 범위 등의 이유일 수 있습니다.
            # 워커 전체가 죽지 않도록 4xx는 로그만 남기고 '빈 결과'로 처리합니다.
            if 400 <= r.status_code < 500:
                body_head = (r.text or "")[:300]
                self.logger.warning(
                    f"[Scopus] {r.status_code} Client Error. query={params.get('query')!r} "
                    f"start={params.get('start')} count={params.get('count')} body_head={body_head!r}"
                )
                return {"raw": {}, "entries": []}

            r.raise_for_status()
            j = r.json()

            sr = j.get("search-results") or {}
            entries = sr.get("entry") or []
            if not isinstance(entries, list):
                entries = []

            return {"raw": j, "entries": entries}

        # 여기에 도달하면 HTTP 응답을 받았으나 재시도 후에도 성공하지 못했거나,
        # 네트워크 오류가 반복되어 끝까지 재시도한 경우입니다.
        raise RuntimeError(f"Scopus API failed after retries. last_status={last_status} body_head={last_text!r}")

    def _build_scopus_query(self, field: str, keyword: str) -> str:
        kw = keyword.strip()
        kw = re.sub(r"\s+", " ", kw)
        kw = kw.replace('"', "")
        if " " in kw:
            kw = f"\"{kw}\""
        return f"{field}({kw})"

    def _parse_entry(self, keyword: str, e: Dict[str, Any]) -> Optional[ScopusItem]:
        title = (e.get("dc:title") or e.get("title") or "").strip()
        doi = (e.get("prism:doi") or "").strip()
        eid = (e.get("eid") or "").strip()
        url = (e.get("prism:url") or "").strip()
        cover_date = (e.get("prism:coverDate") or e.get("coverDate") or "").strip()
        source_title = (e.get("prism:publicationName") or e.get("publicationName") or "").strip()

        if not title and not doi and not eid:
            return None

        return ScopusItem(
            keyword=keyword,
            title=title,
            doi=doi,
            eid=eid,
            url=url,
            cover_date=cover_date,
            source_title=source_title,
            raw=e,
        )

    def _dedup_key(self, item: ScopusItem) -> str:
        if item.doi:
            return f"doi:{item.doi.lower()}"
        if item.eid:
            return f"eid:{item.eid}"
        return f"title:{item.title.lower()}|date:{item.cover_date}"

    def _load_seen(self) -> Dict[str, Any]:
        if not os.path.exists(self.seen_path):
            return {}
        try:
            with open(self.seen_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}

    def _save_seen(self, seen: Dict[str, Any]) -> None:
        tmp = self.seen_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(seen, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.seen_path)

    def _append_results_jsonl(self, keyword: str, items: List[ScopusItem]) -> None:
        safe_kw = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "_", keyword).strip("_")
        path = os.path.join(self.results_dir, f"scopus_{safe_kw}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for it in items:
                row = {
                    "keyword": it.keyword,
                    "title": it.title,
                    "doi": it.doi,
                    "eid": it.eid,
                    "url": it.url,
                    "cover_date": it.cover_date,
                    "source_title": it.source_title,
                    "raw": it.raw,
                    "ts": time.time(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -------------------------
    # Embedding helpers
    # -------------------------
    def _compose_embed_text(self, meta: Dict[str, Any]) -> str:
        """Scopus 메타데이터를 검색 가능한 텍스트로 구성합니다."""
        keyword = str(meta.get("keyword", "")).strip()
        title = str(meta.get("title", "")).strip()
        source_title = str(meta.get("source_title", "")).strip()
        cover_date = str(meta.get("cover_date", "")).strip()
        doi = str(meta.get("doi", "")).strip()
        eid = str(meta.get("eid", "")).strip()
        url = str(meta.get("url", "")).strip()
        abstract = str(meta.get("abstract", "")).strip()

        parts = []
        if keyword:
            parts.append(f"Keyword: {keyword}")
        if title:
            parts.append(f"Title: {title}")
        if source_title:
            parts.append(f"Venue: {source_title}")
        if cover_date:
            parts.append(f"Date: {cover_date}")
        if doi:
            parts.append(f"DOI: {doi}")
        if eid:
            parts.append(f"EID: {eid}")
        if url:
            parts.append(f"URL: {url}")
        if abstract:
            parts.append("Abstract:")
            parts.append(abstract)

        return "\n".join(parts).strip()

    def _embed_pending_seen(self, seen: Dict[str, Any]) -> int:
        """seen 중 embedded=False 인 항목을 VectorDB에 임베딩합니다."""
        max_embed = int(getattr(config, "SCOPUS_MAX_EMBED_PER_RUN", 200))
        chunk_size = int(getattr(config, "CHUNK_SIZE", 1000))
        chunk_overlap = int(getattr(config, "CHUNK_OVERLAP", 200))

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vs = get_vectorstore()

        docs: List[Document] = []
        embedded_keys: List[str] = []

        for key, meta in seen.items():
            if len(docs) >= max_embed:
                break
            if not isinstance(meta, dict):
                continue
            if bool(meta.get("embedded", False)):
                continue

            text_meta = {
                "source": "scopus",
                "source_type": "paper_metadata",
                "keyword": meta.get("keyword", ""),
                "title": meta.get("title", ""),
                "doi": meta.get("doi", ""),
                "eid": meta.get("eid", ""),
                "url": meta.get("url", ""),
                "cover_date": meta.get("cover_date", ""),
                "source_title": meta.get("source_title", ""),
                "abstract": meta.get("abstract", ""),
            }
            embed_text = self._compose_embed_text(text_meta)
            if not embed_text:
                continue

            chunks = splitter.split_text(embed_text)
            for idx, ch in enumerate(chunks):
                if len(docs) >= max_embed:
                    break
                md = dict(text_meta)
                md["scopus_key"] = key
                md["chunk_index"] = idx
                docs.append(Document(page_content=ch, metadata=md))

            embedded_keys.append(key)

        if not docs:
            self.logger.info("[Scopus] no pending items to embed")
            return 0

        with _VSTORE_LOCK:
            vs.add_documents(docs)

        now = time.time()
        for k in embedded_keys:
            if k in seen and isinstance(seen[k], dict):
                seen[k]["embedded"] = True
                seen[k]["embedded_ts"] = now

        self.logger.info(f"[Scopus] embedded chunks added={len(docs)} items={len(embedded_keys)}")
        return len(docs)