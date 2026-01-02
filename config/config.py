# ./config/config.py

# Directories
DATA_DIR = "./data"
NEWS_DIR = "./news_texts"
LOG_DIR = "./logs"
CONFIG_DIR = "./config"
CONFIG_FILE = "./config/config.py"
VECTOR_DB_ROOT = "./vector_db"

# Vector DB
VECTOR_DB = "chroma"

# Models
LLM_MODEL = "llama3.2"
EMBED_MODEL = "qwen3-embedding"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Periodic intervals (seconds)
PDF_SYNC_INTERVAL_SEC = 600
NEWS_CRAWL_INTERVAL_SEC = 600

# News
NEWS_KEYWORDS = ("소프트웨어 공학", "AI 안전", "자동차 기능안전", "SDV")
NEWS_MAX_ITEMS_PER_KEYWORD = 10
NEWS_TIMEOUT_SEC = 20

# Streamlit auto refresh (so periodic checks run)
AUTO_REFRESH_ENABLED = True
AUTO_REFRESH_TICK_SEC = 30

GENERATION_STALE_SEC = 180
