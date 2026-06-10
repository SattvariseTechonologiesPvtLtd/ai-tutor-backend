"""
config.py — centralised paths, model config, constants
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent

# Book DB
QDRANT_BOOK_PATH   = BACKEND_DIR / "qdrant_storage"
BM25_BOOK_PATH     = BACKEND_DIR / "bm25_index.pkl"

# Playlist DB
QDRANT_PL_PATH     = BACKEND_DIR / "qdrant_storage_playlists"
BM25_PL_PATH       = BACKEND_DIR / "playlist_bm25_index.pkl"

# ── AnatomyRAG compat aliases (read by AnatomyRAG.__init__ via `import config`) ──
# Set QDRANT_PATH so AnatomyRAG opens the right local storage.
# Set QDRANT_HOST / QDRANT_PORT to None — not used in local mode.
QDRANT_PATH        = QDRANT_BOOK_PATH
QDRANT_HOST        = None
QDRANT_PORT        = None

# ── Qdrant collection names ────────────────────────────────────────────
# Book
COL_BOOK_CHUNKS    = "anatomy_physiology_chunks"
COL_BOOK_SECTIONS  = "anatomy_physiology_sections"
COL_BOOK_CHAPTERS  = "anatomy_physiology_chapters"

# Playlists
COL_PL_CHUNKS      = "anatomy_transcripts_chunks"
COL_PL_PLAYLISTS   = "anatomy_transcripts_playlists"
COL_PL_TOPICS      = "anatomy_transcripts_topics"

# ── Embedding model ────────────────────────────────────────────────────
EMBED_MODEL_ID     = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM          = 768
MODELS_DIR         = BACKEND_DIR / "models"

# ── DeepSeek API ───────────────────────────────────────────────────────
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL  = "https://api.deepseek.com"
DEEPSEEK_MODEL     = "deepseek-chat"

# ── Retrieval config ───────────────────────────────────────────────────
TOP_K_VECTOR       = 5
TOP_K_BM25         = 5
TOP_K_FINAL        = 6
VECTOR_WEIGHT      = 0.6
BM25_WEIGHT        = 0.4

# ── CORS ───────────────────────────────────────────────────────────────
ALLOWED_ORIGINS    = ["http://localhost:5173", "http://localhost:3000"]