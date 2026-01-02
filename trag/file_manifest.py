# trag/file_manifest.py
from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class PdfEntry:
    path: str
    sha256: str
    mtime: float
    size: int

@dataclass
class NewsEntry:
    url_sha256: str
    title: str
    url: str
    saved_path: str
    content_sha256: str

class ManifestStore:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        ensure_dir(state_dir)

        self.pdf_manifest_path = os.path.join(state_dir, "pdf_manifest.json")
        self.news_manifest_path = os.path.join(state_dir, "news_manifest.json")

        self.pdf_data: Dict[str, Any] = self._load_json(self.pdf_manifest_path, default={"files": {}})
        self.news_data: Dict[str, Any] = self._load_json(self.news_manifest_path, default={"seen": {}})

    def _load_json(self, path: str, default: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, data: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ---------- PDF ----------
    def pdf_is_changed(self, file_path: str) -> bool:
        files = self.pdf_data.get("files", {})
        prev = files.get(file_path)
        if not os.path.exists(file_path):
            return False
        st = os.stat(file_path)
        cur_hash = sha256_file(file_path)

        if prev is None:
            return True
        if prev.get("sha256") != cur_hash:
            return True
        # 보조: mtime/size 비교
        if float(prev.get("mtime", 0)) != float(st.st_mtime):
            return True
        if int(prev.get("size", -1)) != int(st.st_size):
            return True
        return False

    def pdf_update(self, file_path: str) -> PdfEntry:
        st = os.stat(file_path)
        entry = PdfEntry(
            path=file_path,
            sha256=sha256_file(file_path),
            mtime=float(st.st_mtime),
            size=int(st.st_size),
        )
        self.pdf_data.setdefault("files", {})[file_path] = asdict(entry)
        self._save_json(self.pdf_manifest_path, self.pdf_data)
        return entry

    # ---------- NEWS ----------
    def news_is_duplicate_url(self, url: str) -> bool:
        url_hash = sha256_bytes(url.encode("utf-8"))
        seen = self.news_data.get("seen", {})
        return url_hash in seen

    def news_add(self, title: str, url: str, saved_path: str) -> NewsEntry:
        url_hash = sha256_bytes(url.encode("utf-8"))
        content_hash = sha256_file(saved_path) if os.path.exists(saved_path) else ""
        entry = NewsEntry(
            url_sha256=url_hash,
            title=title,
            url=url,
            saved_path=saved_path,
            content_sha256=content_hash,
        )
        self.news_data.setdefault("seen", {})[url_hash] = asdict(entry)
        self._save_json(self.news_manifest_path, self.news_data)
        return entry

    def news_count(self) -> int:
        return len(self.news_data.get("seen", {}))