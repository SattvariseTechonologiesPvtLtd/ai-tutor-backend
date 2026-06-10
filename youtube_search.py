"""
youtube_search.py
- run_background_sync()  → called once at server startup, re-runs youtube_sync
- search(query)          → finds best matching video in missing_data.json
- fetch_transcript(url)  → pulls raw transcript via yt-dlp (no timestamps)
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from openai import OpenAI

log = logging.getLogger("youtube_search")

MISSING_DATA = Path(__file__).parent / "missing_data.json"
YTDLP        = [sys.executable, "-m", "yt_dlp"]
MIN_SCORE    = 0.25   # minimum word-overlap score to accept a match

_client: OpenAI | None = None

def init(api_key: str, base_url: str) -> None:
    global _client
    _client = OpenAI(api_key=api_key, base_url=base_url)
    log.info("youtube_search LLM client ready")


def pick_videos_llm(question: str, video_list: list[dict]) -> list[dict]:
    """
    Pass question + all video titles/descriptions to LLM.
    Returns up to 2 best-matching video dicts.
    """
    if not _client:
        return []
    # Build compact list: just id, title, hashtags to save tokens
    compact = [
        {"i": i, "title": v["title"], "tags": v.get("hashtags", [])[:5]}
        for i, v in enumerate(video_list)
    ]
    prompt = (
        f"You are helping find VB Anatomy YouTube videos relevant to a student's question.\n"
        f"Question: {question}\n\n"
        f"Video list (JSON):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        f"Return ONLY a JSON array of at most 2 index numbers (the 'i' field) of the most relevant videos. "
        f"Example: [3, 17]. If none are relevant return []."
    )
    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=30,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        indices = json.loads(text)
        return [video_list[i] for i in indices if 0 <= i < len(video_list)]
    except Exception as e:
        log.warning("LLM video picker error: %s", e)
        return []


# ── Background sync ───────────────────────────────────────────────────

def run_background_sync():
    """Import and call youtube_sync.run_sync() directly. Called in background thread."""
    try:
        import youtube_sync
        youtube_sync.run_sync(dry_run=False, enrich=False)
        log.info("✅ YouTube sync complete")
    except Exception as e:
        log.warning("YouTube sync failed: %s", e)


# ── Search ────────────────────────────────────────────────────────────

def _normalise(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return set(text.split())


def _score(query_words: set, video: dict) -> float:
    """Simple word-overlap score across title + description + hashtags."""
    candidate_text = " ".join([
        video.get("title", ""),
        video.get("description", ""),
        " ".join(video.get("hashtags", [])),
    ])
    candidate_words = _normalise(candidate_text)
    if not candidate_words:
        return 0.0
    overlap = query_words & candidate_words
    return len(overlap) / max(len(query_words), 1)


def search(query: str) -> dict | None:
    """
    LLM picks the best video(s) from missing_data.json.
    Falls back to keyword scoring if LLM unavailable.
    Returns the single best match or None.
    """
    if not MISSING_DATA.exists():
        log.warning("missing_data.json not found")
        return None
    try:
        data = json.loads(MISSING_DATA.read_text(encoding="utf-8"))
        videos = data.get("videos", [])
    except Exception as e:
        log.error("Failed to load missing_data.json: %s", e)
        return None
    if not videos:
        return None

    # Try LLM picker first
    if _client:
        picks = pick_videos_llm(query, videos)
        if picks:
            log.info("YT LLM pick: '%s'", picks[0]["title"])
            return picks[0]

    # Fallback: keyword scoring
    query_words = _normalise(query)
    best, best_score = None, 0.0
    for v in videos:
        s = _score(query_words, v)
        if s > best_score:
            best_score = s
            best = v
    if best_score < MIN_SCORE:
        log.info("YT search: no keyword match (best=%.2f) for '%s'", best_score, query)
        return None
    log.info("YT keyword match: '%s' (%.2f)", best["title"], best_score)
    return best


# ── Transcript fetch ──────────────────────────────────────────────────

def fetch_transcript(url: str) -> str | None:
    """
    Extract raw transcript text (no timestamps) via yt-dlp auto-sub.
    Returns plain text or None on failure.
    """
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = YTDLP + [
            "--write-auto-sub", "--sub-lang", "en",
            "--sub-format", "vtt", "--skip-download",
            "--no-warnings", "-o", f"{tmpdir}/%(id)s",
            url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            log.warning("yt-dlp transcript error: %s", r.stderr[:200])
            return None

        # Find the downloaded .vtt file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            log.warning("No VTT subtitle file found for %s", url)
            return None

        raw = vtt_files[0].read_text(encoding="utf-8", errors="ignore")
        return _clean_vtt(raw)


def _clean_vtt(vtt: str) -> str:
    """Strip VTT timestamps and tags, deduplicate lines, return plain text."""
    lines = []
    seen  = set()
    for line in vtt.splitlines():
        line = line.strip()
        # Skip headers, timestamps, empty lines
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        # Strip HTML tags like <00:00:01.000><c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return " ".join(lines)