"""
textbook_retriever.py — Textbook retrieval for DEFTXR Anatomy AI

Mirrors retriever.py exactly — no cross-encoder, no separate corpus.json.
BM25s index is built from Qdrant at startup (or loaded if already on disk).
Metadata stored as chunk_metadata.jsonl aligned with corpus order.
"""

import json
import logging
from pathlib import Path

import bm25s
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

log = logging.getLogger("textbook_retriever")

# ── Constants ──────────────────────────────────────────────────────────
COL_CHUNKS   = "anatomy_physiology_chunks"
TOP_K        = 4
VECTOR_WEIGHT = 0.6
BM25_WEIGHT   = 0.4

# ── Module-level state ─────────────────────────────────────────────────
_embedder:    SentenceTransformer | None = None
_qdrant:      QdrantClient        | None = None
_bm25:        bm25s.BM25          | None = None
_chunk_texts: list[str]                  = []
_chunk_meta:  list[dict]                 = []


# ── Initialisation ─────────────────────────────────────────────────────
def init(
    qdrant_client:  QdrantClient,
    embedder:       SentenceTransformer,
    bm25_index_dir: Path,
) -> None:
    """
    Called once at server startup.
    Receives already-loaded Qdrant client and embedder from server.py,
    then builds (or loads) the BM25s index.
    """
    global _embedder, _qdrant, _bm25, _chunk_texts, _chunk_meta

    _embedder = embedder
    _qdrant   = qdrant_client

    log.info("Building/loading bm25s index for textbook...")
    _bm25, _chunk_texts, _chunk_meta = _load_or_build_bm25s_index(bm25_index_dir)
    log.info("  bm25s ready — %d chunks indexed", len(_chunk_texts))


# ── BM25s index (load if exists, build otherwise) ─────────────────────
def _load_or_build_bm25s_index(index_dir: Path):
    meta_path = index_dir / "chunk_metadata.jsonl"

    if index_dir.exists() and (index_dir / "vocab.index.json").exists() and meta_path.exists():
        log.info("  Loading existing bm25s index from %s", index_dir)
        # load_corpus=False — we manage corpus texts ourselves via chunk_metadata.jsonl
        # to avoid bm25s wrapping texts in {"id":..,"text":..} dicts
        retriever = bm25s.BM25.load(index_dir, load_corpus=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = [json.loads(line) for line in f if line.strip()]
        # Re-read raw texts from corpus.jsonl written by bm25s.save()
        corpus_jsonl = index_dir / "corpus.jsonl"
        corpus_texts = []
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                corpus_texts.append(obj["text"] if isinstance(obj, dict) else obj)
        log.info("  Loaded %d chunks from disk", len(corpus_texts))
        return retriever, corpus_texts, metadata

    log.info("  No existing index — building from Qdrant...")
    return _build_bm25s_index(index_dir)


def _build_bm25s_index(index_dir: Path):
    """Scroll all chunks from Qdrant, build and save bm25s index."""
    chunk_texts: list[str]  = []
    chunk_meta:  list[dict] = []
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

    log.info("  Loaded %d chunks from Qdrant", len(chunk_texts))

    corpus_tokens = bm25s.tokenize(chunk_texts, stopwords="en")
    retriever     = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)

    index_dir.mkdir(exist_ok=True)
    retriever.save(index_dir, corpus=chunk_texts)
    log.info("  bm25s index saved → %s", index_dir)

    # Save metadata as .jsonl — one line per chunk, aligned with corpus
    meta_path = index_dir / "chunk_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in chunk_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log.info("  Metadata saved → %s", meta_path)

    return retriever, chunk_texts, chunk_meta


# ── Search helpers ─────────────────────────────────────────────────────
def _embed(text: str) -> list:
    # Import the shared lock from retriever — same embedder instance, not thread-safe
    from retriever import _embed_lock
    with _embed_lock:
        return _embedder.encode(
            f"search_query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def _vector_search(vec: list, top_k: int) -> list[dict]:
    hits = _qdrant.query_points(
        collection_name=COL_CHUNKS,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    return [{"_score": h.score, "_source": "book", **h.payload} for h in hits]


def _bm25s_search(query: str, top_k: int) -> list[dict]:
    """Deduplicated BM25s retrieval — one result per unique chunk_id."""
    tokens = bm25s.tokenize([query])
    results, scores = _bm25.retrieve(tokens, k=top_k * 3, return_as="tuple")
    indices, score_vals = results[0], scores[0]

    seen, out = set(), []
    for idx, score in zip(indices, score_vals):
        if len(out) >= top_k:
            break
        meta = _chunk_meta[int(idx)]
        cid  = meta["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append({
            "_score"       : float(score),
            "_source"      : "book",
            "chunk_id"     : cid,
            "chapter_title": meta["chapter_title"],
            "chapter_num"  : meta["chapter_num"],
            "section_title": meta["section_title"],
            "body_system"  : meta["body_system"],
            "content_type" : meta["content_type"],
            "text"         : _chunk_texts[int(idx)],
        })
    return out


# ── Public API ─────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    """
    Run hybrid search (vector + BM25s) and return fused, ranked results.
    Shape matches reranker.merge() expectations: chunk_id, _fused, _source="book".
    """
    if _qdrant is None or _embedder is None:
        return []

    from reranker import fuse   # local import to avoid circular deps

    vec       = _embed(query)
    vec_hits  = _vector_search(vec, TOP_K)
    bm25_hits = _bm25s_search(query, TOP_K)
    fused     = fuse(vec_hits, bm25_hits, TOP_K, VECTOR_WEIGHT, BM25_WEIGHT)

    # Normalise field names for reranker.merge() and build_context()
    out = []
    for r in fused:
        out.append({
            "chunk_id"     : r.get("chunk_id", ""),
            "text"         : r.get("text", ""),
            "_fused"       : r.get("_fused", 0.0),
            "_source"      : "book",
            "section_title": r.get("section_title", ""),
            "chapter_title": r.get("chapter_title", ""),
            "chapter_num"  : r.get("chapter_num", r.get("chapter_number", "")),
            "body_system"  : r.get("body_system", ""),
            "figure_refs"  : r.get("figure_refs", []),
            "has_figures"  : r.get("has_figures", False),
            # aliases for build_context()
            "video_title"  : r.get("chapter_title", ""),
            "playlist"     : r.get("body_system", "Textbook"),
        })
    return out


def close() -> None:
    global _qdrant
    if _qdrant is not None:
        try:
            _qdrant.close()
        except Exception:
            pass
        _qdrant = None