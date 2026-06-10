"""
reranker.py — Score fusion and reranking for DEFTXR Anatomy AI

Responsibilities:
  - Normalise scores from heterogeneous retrievers (vector, BM25)
  - Fuse scores with configurable weights
  - Return a deduplicated, sorted list of top-K results

No external dependencies beyond the Python standard library.
"""

import logging

log = logging.getLogger("reranker")


def _normalise(hits: list[dict], key: str = "_score") -> list[float]:
    """Min-max normalise raw scores to [0, 1]."""
    vals = [h.get(key, 0.0) for h in hits]
    mn, mx = min(vals, default=0.0), max(vals, default=1.0)
    if mx == mn:
        return [1.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


def fuse(
    vec_hits:      list[dict],
    bm25_hits:     list[dict],
    top_k:         int,
    vector_weight: float = 0.6,
    bm25_weight:   float = 0.4,
) -> list[dict]:
    """
    Weighted score fusion of vector and BM25 results.
    Adds a '_fused' key to each result dict.
    """
    combined: dict[str, dict] = {}

    for hit, ns in zip(vec_hits, _normalise(vec_hits, "_score")):
        cid = hit["chunk_id"]
        combined[cid] = {**hit, "_fused": vector_weight * ns}

    for hit, ns in zip(bm25_hits, _normalise(bm25_hits, "_score")):
        cid = hit["chunk_id"]
        if cid in combined:
            combined[cid]["_fused"] += bm25_weight * ns
        else:
            combined[cid] = {**hit, "_fused": bm25_weight * ns}

    ranked = sorted(combined.values(), key=lambda x: x["_fused"], reverse=True)

    log.debug(
        "Fused %d vector + %d BM25 hits → top-%d  (vec=%.2f bm25=%.2f)",
        len(vec_hits), len(bm25_hits), top_k, vector_weight, bm25_weight,
    )
    return ranked[:top_k]


def merge(
    playlist_hits:   list[dict],
    book_hits:       list[dict],
    top_k:           int,
    playlist_weight: float = 0.7,
    book_weight:     float = 0.3,
    merge_threshold: float = 0.12,
) -> list[dict]:
    """
    Merge playlist and textbook results into a single top-K list.
    Both sources now carry '_fused' (set by reranker.fuse() inside each retriever).
    """
    playlist_hits = playlist_hits or []
    book_hits     = book_hits     or []

    combined: dict[str, dict] = {}

    for hit, ns in zip(playlist_hits, _normalise(playlist_hits, "_fused")):
        cid = hit["chunk_id"]
        combined[cid] = {**hit, "_merged": playlist_weight * ns}

    for hit, ns in zip(book_hits, _normalise(book_hits, "_fused")):
        cid = hit["chunk_id"]
        if cid in combined:
            combined[cid]["_merged"] = max(
                combined[cid]["_merged"], book_weight * ns
            )
        else:
            combined[cid] = {**hit, "_merged": book_weight * ns}

    ranked = sorted(combined.values(), key=lambda x: x["_merged"], reverse=True)

    log.debug(
        "Merged %d playlist + %d book hits → top-%d",
        len(playlist_hits), len(book_hits), top_k,
    )
    before = len(ranked)
    ranked = [r for r in ranked if r["_merged"] >= merge_threshold]
    log.debug(
        "Threshold %.0f%% dropped %d/%d results",
        merge_threshold * 100, before - len(ranked), before,
    )
    return ranked[:top_k]
