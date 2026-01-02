# config.py
from __future__ import annotations


NAVER_CLIENT_ID = "GXfsvgyrE8K9tQMntP4R"
NAVER_CLIENT_SECRET = "sOu_94iEYh"
NAVER_USE_OPENAPI = True


# ========== Directory ==========
# config.py (디렉토리 섹션만 교체)

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NEWS_TEXT_DIR = os.path.join(PROJECT_ROOT, "news_texts")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
STATE_DIR = os.path.join(PROJECT_ROOT, "state")

CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# ========== Chunking ==========
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ========== Periodic Jobs ==========
# (4) PDF 변경사항 확인 주기
PDF_WATCH_INTERVAL_SECONDS = 10 * 60   # 10분
# (11) 뉴스 크롤링 주기
NEWS_CRAWL_INTERVAL_SECONDS = 10 * 60  # 10분

# ========== VectorDB ==========
VECTORDB_PROVIDER = "chroma"     # 기본값 ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"

# ========== Models (Ollama) ==========
# (6) LLM / Embedding 모델
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.2"                 # 기본값 Ollama3.2
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "qwen3-embedding"    # 기본값 qwen3-embedding

# ========== Retrieval ==========
TOP_K = 4
MAX_CONTEXT_CHARS = 4000

# ========== News Crawling ==========
NEWS_KEYWORDS = [
    "소프트웨어 공학", 
    "AI 안전", 
    "자동차 기능안전", 
    "SDV",
    "Ontology"
]  # (10) 기본값
GOOGLE_NEWS_MAX_RESULTS = 10
NAVER_NEWS_MAX_RESULTS = 10
NEWS_LANGUAGE = "ko"
NEWS_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

# ========== Streamlit ==========
APP_TITLE = "TRAG Chatbot ver 0.1"
SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. "
    "If context is provided, ground your answer in it. "
    "If context is not provided, answer based on general knowledge. "
    "Be consistent with the ongoing chat history."
    "All answers must be in English at the first"
    "Then, translate the answer to Korean."
)

# =========================
# Paper fetch (Semantic Scholar)
# =========================
PAPER_FETCH_INTERVAL_SECONDS = 60000        # 기본 10분
PAPER_MAX_RESULTS_PER_KEYWORD = 5         # 키워드당 최대 검색 결과
PAPER_MAX_DOWNLOADS_PER_RUN = 5           # 한 번 실행에서 최대 다운로드 수(과도한 요청 방지)
PAPER_REQUEST_TIMEOUT_SECONDS = 20
PAPER_USER_AGENT = "Mozilla/5.0 (compatible; TRAG02/1.0; +https://localhost)"  # 원하면 수정

# Semantic Scholar Graph API(무료, 키 없이도 일정 범위 사용 가능)
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,year,authors,externalIds,isOpenAccess,openAccessPdf,url"

# =========================
# Auto translate KO -> EN for paper queries
# =========================
PAPER_AUTO_TRANSLATE_TO_EN = True

# 번역을 Ollama로 수행(권장: 로컬이라 안정적). 실패 시 단순 fallback.
PAPER_TRANSLATE_WITH_OLLAMA = True
PAPER_TRANSLATION_MODEL = LLM_MODEL   # 기본 LLM과 동일.

# 번역 캐시 파일(STATE_DIR 아래 저장)
PAPER_TRANSLATION_CACHE_FILE = "paper_translation_cache.json"

# 논문 다운로드 결과 기록(중복 다운로드 방지)
PAPER_MANIFEST_FILE = "paper_manifest.json"


# --- Scopus API (Elsevier) ---
SCOPUS_ENABLED = True

# scopus embedding on/off
SCOPUS_EMBED_ENABLED = True

# 필수: Elsevier Developer Portal에서 발급받은 API Key
SCOPUS_API_KEY = "5aedef11e458bfa8c7cbc4aa02e9a7e8"  # 예: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 선택: insttoken (기관 IP 인증이 불가능한 환경에서 필요할 수 있음)
SCOPUS_INSTTOKEN = ""  # 예: "xxxxxxxxxxxxxxxx"

# Scopus Search API endpoint
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"  #  [oai_citation:2‡Elsevier Developer Portal](https://dev.elsevier.com/guides/Scopus%20API%20Guide_V1_20230907.pdf)

# 검색 키워드(뉴스 키워드와 동일하게 쓰고 싶다면 NEWS_KEYWORDS를 그대로 사용)
SCOPUS_KEYWORDS = NEWS_KEYWORDS

# Scopus query 필드: 기본 TITLE-ABS-KEY 사용 (고급검색 문법은 Scopus Advanced Search와 동일 계열)
SCOPUS_QUERY_FIELD = "TITLE-ABS-KEY"

# view: STANDARD 또는 COMPLETE
#SCOPUS_VIEW = "STANDARD"  # 또는 "COMPLETE"  [oai_citation:3‡Elsevier Developer Portal](https://dev.elsevier.com/guides/Scopus%20API%20Guide_V1_20230907.pdf)
SCOPUS_VIEW = "COMPLETE"  # 또는 "COMPLETE"  [oai_citation:3‡Elsevier Developer Portal](https://dev.elsevier.com/guides/Scopus%20API%20Guide_V1_20230907.pdf)

# 페이징
SCOPUS_COUNT_PER_CALL = 25  # 한 번에 받을 개수(count). Scopus API는 start/count로 배치 조회 가능  [oai_citation:4‡Elsevier Developer Portal](https://dev.elsevier.com/guides/Scopus%20API%20Guide_V1_20230907.pdf)
SCOPUS_MAX_RESULTS_PER_KEYWORD = 50  # 키워드당 최대 몇 개까지 가져올지

# 정렬(옵션): relevancy, pubyear 등 (가이드에 sort 옵션 존재)  [oai_citation:5‡Elsevier Developer Portal](https://dev.elsevier.com/guides/Scopus%20API%20Guide_V1_20230907.pdf)
SCOPUS_SORT = "relevancy"  # 또는 "pubyear"

# 한 run_once에서 너무 많이 넣지 않도록 제한(선택)
SCOPUS_MAX_EMBED_PER_RUN = 200

# Scopus 검색 주기(초)
# - 하루 1회 실행하려면 24*60*60(=86400)으로 설정
# - BaseRAG.py는 SCOPUS_SEARCH_INTERVAL_SECONDS를 읽어 워커 주기를 결정합니다.
SCOPUS_SEARCH_INTERVAL_SECONDS = 24 * 60 * 60  # 1 day

# 요청/재시도
SCOPUS_REQUEST_TIMEOUT_SECONDS = 20
SCOPUS_MAX_RETRIES = 5
SCOPUS_BACKOFF_BASE_SECONDS = 2.0
SCOPUS_BACKOFF_CAP_SECONDS = 60.0

SCOPUS_MAX_START = 5000
SCOPUS_API_MAX_START = 5000   # 안전 상한(기본 5000)

# 저장 경로(이미 프로젝트에 STATE_DIR / NEWS_TEXT_DIR 같은 것이 있으면 그걸 사용)
# STATE_DIR = "./state"
# LOG_DIR = "./logs"