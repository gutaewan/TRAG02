import os
import re
import json
import uuid
import hashlib
import logging
from typing import Any, Dict, List
from urllib.parse import urlparse, unquote, quote_plus

import requests
import feedparser

# Optional: full article extraction
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None  # type: ignore

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from .vectorstore import (
    load_fingerprints,
    save_fingerprints,
    file_fingerprint,
    delete_by_source,
    add_documents,
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

# PDF crawling (keyword -> web search -> pdf urls)
DDG_HTML_SEARCH = "https://duckduckgo.com/html/"

PDF_DEFAULTS = {
    "PDF_DIR_FALLBACK": "./data",
    "PDF_MAX_ITEMS_PER_KEYWORD": 3,
    "PDF_TIMEOUT_SEC": 15,
    "PDF_MAX_BYTES": 25 * 1024 * 1024,  # 25MB
}
def pdf_dir(cfg: Dict[str, Any]) -> str:
    # Prefer explicit config keys; fall back to ./data
    return str(cfg.get("DATA_DIR") or cfg.get("PDF_DIR") or PDF_DEFAULTS["PDF_DIR_FALLBACK"])


def ensure_pdf_dir(cfg: Dict[str, Any]) -> None:
    os.makedirs(pdf_dir(cfg), exist_ok=True)


def pdf_index_path(cfg: Dict[str, Any]) -> str:
    # Store an index file in the same data directory
    return os.path.join(pdf_dir(cfg), "pdf_index.json")


def load_pdf_index(cfg: Dict[str, Any]) -> Dict[str, str]:
    p = pdf_index_path(cfg)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_pdf_index(cfg: Dict[str, Any], idx: Dict[str, str]) -> None:
    with open(pdf_index_path(cfg), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def _is_probably_pdf_url(u: str) -> bool:
    u = (u or "").strip()
    if not u:
        return False
    # Common direct pdf patterns
    if u.lower().endswith(".pdf"):
        return True
    # Sometimes pdf is behind query params
    return ".pdf" in u.lower()


def search_pdf_urls_ddg(cfg: Dict[str, Any], keyword: str, logger: logging.Logger) -> List[str]:
    """Best-effort web search for PDF URLs using DuckDuckGo HTML endpoint."""
    max_items = int(cfg.get("PDF_MAX_ITEMS_PER_KEYWORD", PDF_DEFAULTS["PDF_MAX_ITEMS_PER_KEYWORD"]))
    timeout = int(cfg.get("PDF_TIMEOUT_SEC", PDF_DEFAULTS["PDF_TIMEOUT_SEC"]))

    q = f"{keyword} filetype:pdf"
    try:
        r = requests.post(
            DDG_HTML_SEARCH,
            data={"q": q},
            headers=DEFAULT_HEADERS,
            timeout=timeout,
        )
        r.raise_for_status()
        html = r.text or ""
    except Exception as e:
        logger.info(f"[PDF] ddg search failed for '{keyword}': {e}")
        return []

    # DDG HTML results contain links like: <a rel="nofollow" class="result__a" href="...">
    # Extract hrefs and keep only probable pdf urls.
    urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
    cleaned: List[str] = []
    for u in urls:
        u = normalize_url(u)
        if not u:
            continue
        if is_google_url(u):
            # DDG sometimes returns google wrappers; we still try to resolve if possible
            u = resolve_publisher_url(u, timeout_sec=timeout, logger=logger)
        if not _is_probably_pdf_url(u):
            continue
        if u not in cleaned:
            cleaned.append(u)
        if len(cleaned) >= max_items:
            break

    if cleaned:
        logger.info(f"[PDF] found {len(cleaned)} candidates for '{keyword}'")
    return cleaned


def _safe_pdf_filename(url: str) -> str:
    # Create a stable-ish name from url hash
    h = hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}.pdf"


def download_pdf(cfg: Dict[str, Any], url: str, logger: logging.Logger) -> str:
    """Download a PDF to ./data (or cfg override). Returns local path or empty string."""
    url = normalize_url(url)
    if not url:
        return ""

    ensure_pdf_dir(cfg)

    timeout = int(cfg.get("PDF_TIMEOUT_SEC", PDF_DEFAULTS["PDF_TIMEOUT_SEC"]))
    max_bytes = int(cfg.get("PDF_MAX_BYTES", PDF_DEFAULTS["PDF_MAX_BYTES"]))

    try:
        # Follow redirects to get the final url
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        r.raise_for_status()
        final_url = normalize_url(getattr(r, "url", "") or url)

        # Basic content-type guard
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not _is_probably_pdf_url(final_url):
            logger.info(f"[PDF] skipped non-pdf content: {final_url} (content-type={ctype})")
            return ""

        fname = _safe_pdf_filename(final_url)
        path = os.path.join(pdf_dir(cfg), fname)

        # If already exists, do not re-download
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

        # Stream download with size cap
        total = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    try:
                        f.close()
                    except Exception:
                        pass
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    logger.info(f"[PDF] download exceeded limit ({max_bytes} bytes): {final_url}")
                    return ""
                f.write(chunk)

        logger.info(f"[PDF] downloaded: {os.path.basename(path)} bytes={total} url={final_url}")
        return path

    except Exception as e:
        logger.info(f"[PDF] download failed: {url} ({e})")
        return ""


def fetch_pdfs_for_keywords(cfg: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for kw in cfg.get("NEWS_KEYWORDS", []):
        try:
            urls = search_pdf_urls_ddg(cfg, kw, logger)
            for u in urls:
                items.append({"keyword": kw, "url": u})
        except Exception as e:
            logger.info(f"[PDF] fetch urls failed for '{kw}': {e}")
    return items


def save_pdf_items(cfg: Dict[str, Any], logger: logging.Logger, items: List[Dict[str, Any]]) -> List[str]:
    ensure_pdf_dir(cfg)
    idx = load_pdf_index(cfg)

    created: List[str] = []
    for it in items:
        url = (it.get("url") or "").strip()
        kw = (it.get("keyword") or "").strip()
        if not url:
            continue

        pid = hashlib.sha256(url.encode("utf-8")).hexdigest()
        if pid in idx:
            continue

        path = download_pdf(cfg, url, logger)
        if not path:
            continue

        # record in index (dedupe)
        idx[pid] = os.path.basename(path)
        created.append(path)
        logger.info(f"[PDF] saved index keyword='{kw}' file='{os.path.basename(path)}'")

    if created:
        save_pdf_index(cfg, idx)
        logger.info(f"[PDF] created {len(created)} new pdf files")

    return created


def crawl_pdfs_once(cfg: Dict[str, Any], logger: logging.Logger) -> List[str]:
    items = fetch_pdfs_for_keywords(cfg, logger)
    return save_pdf_items(cfg, logger, items)


def list_pdf_files(cfg: Dict[str, Any]) -> List[str]:
    ensure_pdf_dir(cfg)
    d = pdf_dir(cfg)
    return sorted(
        os.path.join(d, n)
        for n in os.listdir(d)
        if n.lower().endswith(".pdf")
    )


def sync_pdf_dir(vs: Chroma, cfg: Dict[str, Any], embed_model: str, logger: logging.Logger) -> int:
    """Sync ./data (PDF) into vector DB: only new/changed PDFs are embedded."""
    fps = load_fingerprints(cfg, embed_model)
    before = dict(fps.get("pdf", {}))

    total_added = 0
    for p in list_pdf_files(cfg):
        fp = file_fingerprint(p)
        if before.get(p) == fp:
            continue

        delete_by_source(vs, p, logger)

        loader = PyPDFLoader(p)
        docs = loader.load()  # page-level docs
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["source_type"] = "pdf"

        total_added += add_documents(
            vs, docs, cfg, logger,
            prefix="PDF",
            reason=f"changed_or_new: {os.path.basename(p)}"
        )
        fps["pdf"][p] = fp

    save_fingerprints(cfg, embed_model, fps)
    return total_added


def crawl_news_and_pdfs_once(cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, List[str]]:
    """Convenience: crawl news text files and keyword-based PDFs (download only)."""
    news_files = crawl_news_once(cfg, logger)
    pdf_files = crawl_pdfs_once(cfg, logger)
    return {"news": news_files, "pdf": pdf_files}


def google_news_rss_url(query: str) -> str:
    from urllib.parse import quote_plus
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def strip_html(text: str) -> str:
    s = text or ""
    if BeautifulSoup is not None:
        try:
            s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
        except Exception:
            s = re.sub(r"<[^>]+>", " ", s)
    else:
        s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_url(u: str) -> str:
    return (u or "").strip()


def _is_google_host(netloc: str) -> bool:
    n = (netloc or "").lower()
    return (
        n.endswith(".google.com")
        or n == "google.com"
        or n.endswith(".news.google.com")
        or n == "news.google.com"
        or n.endswith(".googleusercontent.com")
    )


def is_google_url(u: str) -> bool:
    try:
        return _is_google_host(urlparse(u).netloc)
    except Exception:
        return False


def _pick_first_non_google_url(urls: List[str]) -> str:
    for u in urls:
        u = normalize_url(u)
        if not u:
            continue
        if u.startswith("http://") or u.startswith("https://"):
            if not is_google_url(u):
                return u
    return ""


def extract_publisher_url_from_google_html(html: str, logger: logging.Logger) -> str:
    """Try to extract the original publisher URL from Google News article HTML."""
    s = html or ""
    if not s:
        return ""

    # 1) canonical link
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', s, flags=re.IGNORECASE)
    if m:
        cand = normalize_url(m.group(1))
        if cand and not is_google_url(cand):
            return cand

    # 2) og:url meta
    m = re.search(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']', s, flags=re.IGNORECASE)
    if m:
        cand = normalize_url(m.group(1))
        if cand and not is_google_url(cand):
            return cand

    # 3) Look for google redirect patterns containing url=... (percent-encoded)
    #    e.g. https://www.google.com/url?...&url=https%3A%2F%2Fpublisher...
    url_params = re.findall(r"[?&]url=([^&\"'<> ]+)", s, flags=re.IGNORECASE)
    decoded = []
    for p in url_params:
        try:
            decoded.append(unquote(p))
        except Exception:
            pass
    picked = _pick_first_non_google_url(decoded)
    if picked:
        return picked

    # 4) Last resort: pick any visible https://... that is not a google domain
    raw_urls = re.findall(r"https?://[^\s\"'<>]+", s)
    picked = _pick_first_non_google_url(raw_urls)
    if picked:
        return picked

    logger.info("[NEWS] could not extract publisher url from google html")
    return ""


def resolve_publisher_url(google_or_any_url: str, timeout_sec: int, logger: logging.Logger) -> str:
    """Resolve Google News RSS link to the original publisher URL when possible."""
    u = normalize_url(google_or_any_url)
    if not u:
        return ""

    # If it's already a non-google URL, just return it.
    if not is_google_url(u):
        return u

    try:
        r = requests.get(u, headers=DEFAULT_HEADERS, timeout=timeout_sec, allow_redirects=True)
        r.raise_for_status()
        final_url = normalize_url(getattr(r, "url", "") or "")

        # Redirect already landed on publisher
        if final_url and not is_google_url(final_url):
            return final_url

        html = r.text or ""
        pub = extract_publisher_url_from_google_html(html, logger)
        if pub:
            return pub

        # Fallback: keep google url (may still be usable for summary)
        return final_url or u
    except Exception as e:
        logger.info(f"[NEWS] resolve_publisher_url failed: {u} ({e})")
        return u


def representative_sentence(title: str, summary_html: str) -> str:
    summary = strip_html(summary_html)
    if not summary:
        return (title or "").strip()

    m = re.search(r"[.!?](?:\s+|$)|다\.(?:\s+|$)", summary)
    first = summary[: m.end()].strip() if m else summary.strip()

    if len(first) < 25:
        return f"{(title or '').strip()} - {first}".strip(" -")
    return first


def stable_news_id(title: str, link: str) -> str:
    base = ((title or "").strip().lower() + "|" + (link or "").strip())
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def fetch_article_fulltext(url: str, timeout_sec: int, logger: logging.Logger) -> str:
    url = normalize_url(url)
    if not url:
        return ""

    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_sec, allow_redirects=True)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logger.info(f"[NEWS] article fetch failed: {url} ({e})")
        return ""

    if trafilatura is not None:
        try:
            downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
            if downloaded:
                text = re.sub(r"\s+", " ", downloaded).strip()
                if len(text) >= 400:
                    return text
        except Exception as e:
            logger.info(f"[NEWS] trafilatura extract failed: {url} ({e})")

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            article = soup.find("article")
            if article is not None:
                text = article.get_text(" ", strip=True)
            else:
                candidates = []
                for sel in ["main", "#content", ".content", ".article", ".news", ".post", "body"]:
                    node = soup.select_one(sel)
                    if node is None:
                        continue
                    ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
                    joined = " ".join([p for p in ps if p])
                    if len(joined) > len(" ".join(candidates)):
                        candidates = ps
                text = " ".join([p for p in candidates if p])

            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 400:
                return text
        except Exception as e:
            logger.info(f"[NEWS] soup extract failed: {url} ({e})")

    return ""


def fetch_google_news(cfg: Dict[str, Any], keyword: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    url = google_news_rss_url(keyword)
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=int(cfg["NEWS_TIMEOUT_SEC"]))
    r.raise_for_status()
    feed = feedparser.parse(r.text)

    out: List[Dict[str, Any]] = []
    for e in feed.entries[: int(cfg["NEWS_MAX_ITEMS_PER_KEYWORD"])]:
        title = getattr(e, "title", "").strip()
        google_link = normalize_url(getattr(e, "link", "").strip())
        link = resolve_publisher_url(google_link, timeout_sec=int(cfg["NEWS_TIMEOUT_SEC"]), logger=logger)
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        summary = getattr(e, "summary", "") or getattr(e, "description", "")

        fulltext = ""
        if link:
            fulltext = fetch_article_fulltext(link, timeout_sec=int(cfg["NEWS_TIMEOUT_SEC"]), logger=logger)
        if (not fulltext) and google_link and (google_link != link):
            # fallback: sometimes publisher extraction fails; try the google page anyway
            fulltext = fetch_article_fulltext(google_link, timeout_sec=int(cfg["NEWS_TIMEOUT_SEC"]), logger=logger)

        out.append(
            {
                "keyword": keyword,
                "title": title,
                "link": link,
                "google_link": google_link,
                "published": published,
                "summary": summary,
                "fulltext": fulltext,
            }
        )
    return out


def ensure_news_dir(cfg: Dict[str, Any]) -> None:
    os.makedirs(str(cfg["NEWS_DIR"]), exist_ok=True)


def news_index_path(cfg: Dict[str, Any]) -> str:
    return os.path.join(str(cfg["NEWS_DIR"]), "news_index.json")


def load_news_index(cfg: Dict[str, Any]) -> Dict[str, str]:
    p = news_index_path(cfg)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_news_index(cfg: Dict[str, Any], idx: Dict[str, str]) -> None:
    with open(news_index_path(cfg), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def save_news_items(cfg: Dict[str, Any], logger: logging.Logger, items: List[Dict[str, Any]]) -> List[str]:
    ensure_news_dir(cfg)
    idx = load_news_index(cfg)

    created: List[str] = []
    for e in items:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        google_link = (e.get("google_link") or "").strip()
        published = (e.get("published") or "").strip()
        keyword = (e.get("keyword") or "").strip()
        summary = (e.get("summary") or "").strip()
        fulltext = (e.get("fulltext") or "").strip()

        nid = stable_news_id(title, link or google_link)
        if nid in idx:
            continue

        fname = f"news_{uuid.uuid4().hex}.txt"
        path = os.path.join(str(cfg["NEWS_DIR"]), fname)

        rep = representative_sentence(title, fulltext if fulltext else summary)
        body = strip_html(fulltext) if fulltext else strip_html(summary)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# title: {title}\n")
            f.write(f"# url: {link}\n")
            f.write(f"# google_url: {google_link}\n")
            f.write(f"# published: {published}\n")
            f.write(f"# keyword: {keyword}\n")
            f.write(f"# representative: {rep}\n")
            f.write(f"# has_fulltext: {bool(fulltext)}\n\n")
            f.write(f"# resolved_url: {link or google_link}\n\n")
            f.write(rep + "\n\n")
            f.write("===== FULL CONTENT =====\n")
            f.write(body + "\n")

        logger.info(f"[NEWS] saved: {fname} fulltext={bool(fulltext)} len={len(body)}")

        idx[nid] = fname
        created.append(path)

    if created:
        save_news_index(cfg, idx)
        logger.info(f"[NEWS] created {len(created)} new files")

    return created


def crawl_news_once(cfg: Dict[str, Any], logger: logging.Logger) -> List[str]:
    items: List[Dict[str, Any]] = []
    for kw in cfg["NEWS_KEYWORDS"]:
        try:
            items.extend(fetch_google_news(cfg, kw, logger))
        except Exception as e:
            logger.info(f"[NEWS] fetch failed for '{kw}': {e}")
    return save_news_items(cfg, logger, items)


def list_news_txts(cfg: Dict[str, Any]) -> List[str]:
    ensure_news_dir(cfg)
    news_dir = str(cfg["NEWS_DIR"])
    return sorted(
        os.path.join(news_dir, n)
        for n in os.listdir(news_dir)
        if n.lower().endswith(".txt")
    )


def sync_news_dir(vs: Chroma, cfg: Dict[str, Any], embed_model: str, logger: logging.Logger) -> int:
    fps = load_fingerprints(cfg, embed_model)
    before = dict(fps.get("news", {}))

    total_added = 0
    for p in list_news_txts(cfg):
        fp = file_fingerprint(p)
        if before.get(p) == fp:
            continue

        delete_by_source(vs, p, logger)

        loader = TextLoader(p, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["source_type"] = "news_text"

        total_added += add_documents(
            vs, docs, cfg, logger,
            prefix="NEWS",
            reason=f"changed_or_new: {os.path.basename(p)}"
        )
        fps["news"][p] = fp

    save_fingerprints(cfg, embed_model, fps)
    return total_added


# Helper: sync both news and pdf dirs into vector DB
def sync_news_and_pdf_dirs(vs: Chroma, cfg: Dict[str, Any], embed_model: str, logger: logging.Logger) -> int:
    """Sync both ./news_texts/*.txt and ./data/*.pdf into the vector DB."""
    added_news = sync_news_dir(vs, cfg, embed_model, logger)
    added_pdf = sync_pdf_dir(vs, cfg, embed_model, logger)
    total = int(added_news) + int(added_pdf)
    if total:
        logger.info(f"[SYNC] vector DB updated: +{total} (news={added_news}, pdf={added_pdf})")
    return total