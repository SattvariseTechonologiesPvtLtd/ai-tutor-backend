"""
build_textbook_bm25s_index.py — One-time index builder for textbook BM25s

Run this once (or whenever the textbook Qdrant collection changes):
    python build_textbook_bm25s_index.py

Mirrors exactly what retriever.py does for the playlist index.
Saves:
  bm25s_index_book/           ← bm25s native index files (corpus.jsonl, vocab, etc.)
  bm25s_index_book/chunk_metadata.jsonl  ← one JSON object per line, aligned with corpus
"""

import json
import logging
from pathlib import Path

import bm25s
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_textbook_bm25s_index")

# ── Paths ──────────────────────────────────────────────────────────────
BASE            = Path(__file__).parent
QDRANT_PATH     = BASE / "qdrant_storage"
INDEX_DIR       = BASE / "bm25s_index_book"
COL_CHUNKS      = "anatomy_physiology_chunks"


def build():
    log.info("Opening Qdrant at %s ...", QDRANT_PATH)
    client = QdrantClient(path=str(QDRANT_PATH))

    chunk_texts: list[str]  = []
    chunk_meta:  list[dict] = []
    offset = None

    log.info("Scrolling all chunks from '%s' ...", COL_CHUNKS)
    while True:
        res = client.scroll(
            collection_name=COL_CHUNKS,
            scroll_filter=None,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        pts = res[0]
        if not pts:
            break
        for pt in pts:
            p = pt.payload
            chunk_texts.append(p.get("text", ""))
            chunk_meta.append({
                "chunk_id"     : p.get("chunk_id", ""),
                "chapter_title": p.get("chapter_title", ""),
                "chapter_num"  : p.get("chapter_number", ""),
                "section_title": p.get("section_title", ""),
                "body_system"  : p.get("body_system", ""),
                "content_type" : p.get("content_type", ""),
            })
        offset = res[1]
        if offset is None:
            break

    client.close()
    log.info("Loaded %d chunks from Qdrant", len(chunk_texts))

    # Build bm25s index (same as retriever.py)
    corpus_tokens = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever     = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)

    INDEX_DIR.mkdir(exist_ok=True)
    retriever.save(INDEX_DIR, corpus=chunk_texts)   # writes corpus.jsonl + vocab etc.
    log.info("bm25s index saved → %s", INDEX_DIR)

    # Save metadata as .jsonl — one line per chunk, aligned with corpus order
    meta_path = INDEX_DIR / "chunk_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log.info("Metadata saved → %s  (%d lines)", meta_path, len(chunk_meta))


if __name__ == "__main__":
    build()
