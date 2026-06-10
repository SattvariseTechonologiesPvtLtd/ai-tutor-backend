"""
retriever.py — Vector search + BM25s retrieval for DEFTXR Anatomy AI

Responsibilities:
  - Load / build the bm25s index from Qdrant chunks
  - Run vector search against Qdrant
  - Run BM25s keyword search
  - Expose a single retrieve(query) entry point used by server.py
"""

import logging
import threading
import bm25s
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

log = logging.getLogger("retriever")

# ── Constants ──────────────────────────────────────────────────────────
COL_CHUNKS      = "anatomy_transcripts_chunks"
EMBED_MODEL     = "nomic-ai/nomic-embed-text-v1.5"
TOP_K           = 4
VECTOR_WEIGHT   = 0.6
BM25_WEIGHT     = 0.4

# ── Module-level state (initialised once via init()) ───────────────────
_embedder:    SentenceTransformer | None = None
_qdrant:      QdrantClient        | None = None
_bm25:        bm25s.BM25          | None = None
_chunk_texts: list[str]                  = []
_chunk_meta:  list[dict]                 = []
_embed_lock:  threading.Lock             = threading.Lock()


# ── Initialisation ─────────────────────────────────────────────────────
def init(
    qdrant_client:  QdrantClient,
    embedder:       SentenceTransformer,
    bm25_index_dir: Path,
) -> None:
    """
    Called once at server startup.
    Receives already-loaded Qdrant client and embedder from server.py,
    then builds the BM25s index in-process.
    """
    global _embedder, _qdrant, _bm25, _chunk_texts, _chunk_meta

    _embedder = embedder
    _qdrant   = qdrant_client

    log.info("Building bm25s index from Qdrant chunks...")
    _bm25, _chunk_texts, _chunk_meta = _build_bm25s_index(bm25_index_dir)
    log.info(f"  bm25s ready — {len(_chunk_texts)} chunks indexed")


# ── BM25s index builder ────────────────────────────────────────────────
def _build_bm25s_index(index_dir: Path):
    """Scroll all chunks from Qdrant, build and save bm25s index."""
    chunk_texts, chunk_metadata = [], []
    offset = None

    while True:
        res = _qdrant.scroll(
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
            chunk_texts.append(p["text"])
            chunk_metadata.append({
                "chunk_id"   : p["chunk_id"],
                "video_title": p["video_title"],
                "playlist"   : p["playlist"],
                "topic_area" : p["topic_area"],
            })
        offset = res[1]
        if offset is None:
            break

    log.info(f"  Loaded {len(chunk_texts)} chunks from Qdrant")

    corpus_tokens = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever     = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)
    retriever.save(index_dir, corpus=chunk_texts)
    log.info(f"  bm25s index saved → {index_dir}")

    return retriever, chunk_texts, chunk_metadata


# ── Search helpers ─────────────────────────────────────────────────────
def _embed(text: str) -> list:
    with _embed_lock:
        return _embedder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


def _vector_search(vec: list, top_k: int) -> list[dict]:
    hits = _qdrant.query_points(
        collection_name=COL_CHUNKS,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    return [{"_score": h.score, "_source": "playlist", **h.payload} for h in hits]


def _bm25s_search(query: str, top_k: int) -> list[dict]:
    """Deduplicated BM25s retrieval — one result per unique video."""
    tokens = bm25s.tokenize([query])
    results, scores = _bm25.retrieve(tokens, k=top_k * 3, return_as="tuple")
    indices, score_vals = results[0], scores[0]

    seen, out = set(), []
    for idx, score in zip(indices, score_vals):
        if len(out) >= top_k:
            break
        meta = _chunk_meta[int(idx)]
        if meta["video_title"] in seen:
            continue
        seen.add(meta["video_title"])
        out.append({
            "_score"     : float(score),
            "_source"    : "playlist",
            "chunk_id"   : meta["chunk_id"],
            "video_title": meta["video_title"],
            "playlist"   : meta["playlist"],
            "topic_area" : meta["topic_area"],
            "text"       : _chunk_texts[int(idx)],
        })
    return out


# ── Public API ─────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    """
    Run hybrid search (vector + BM25s) and return fused, ranked results.
    Delegated to reranker.fuse() for score combination.
    """
    from reranker import fuse   # local import to avoid circular deps

    vec       = _embed(query)
    vec_hits  = _vector_search(vec, TOP_K)
    bm25_hits = _bm25s_search(query, TOP_K)
    return fuse(vec_hits, bm25_hits, TOP_K, VECTOR_WEIGHT, BM25_WEIGHT)


def build_context(ranked: list[dict]) -> str:
    """Format ranked chunks into a numbered context string for the LLM."""
    parts = []
    for i, r in enumerate(ranked, 1):
        if r.get("_source") == "book":
            ch    = r.get("chapter_num", "")
            sec   = r.get("section_title", "")
            label = f"[{i}] 📖 {r.get('chapter_title', 'Textbook')} — {sec} (Ch {ch})"
        else:
            label = f"[{i}] 🎬 {r['video_title']} | {r['playlist']}"
        snippet = " ".join(r.get("text", "").split()[:500])
        parts.append(f"{label}\n{snippet}")
    return "\n\n".join(parts)