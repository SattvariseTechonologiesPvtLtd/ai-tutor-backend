"""
youtube_sync.py
Dry run:  python youtube_sync.py --dry-run
Full run: python youtube_sync.py
Enrich (slow, fetches description/tags): python youtube_sync.py --enrich

Reads clean_metadata.json (your 142 ingested videos) and compares against
the full channel to produce missing_videos.json.
"""

import json
import re
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Works whether yt-dlp is installed as a script or a module inside the venv
YTDLP = [sys.executable, "-m", "yt_dlp"]

CHANNEL_URL        = "https://www.youtube.com/@VBAnatomy/videos"
KNOWN_METADATA     = Path(__file__).parent / "clean_metadata.json"
OUTPUT_JSON        = Path(__file__).parent / "missing_data.json"
MIN_DURATION_SECS  = 120   # filter out reels / shorts < 2 minutes


# ── helpers ──────────────────────────────────────────────────────────────────

def normalise(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def load_known_titles() -> set[str]:
    data = json.loads(KNOWN_METADATA.read_text())
    return {normalise(v["title"]) for v in data["videos"]}


def fetch_channel_videos() -> list[dict]:
    cmd = YTDLP + [
        "--flat-playlist", "--dump-json",
        "--yes-playlist", "--ignore-errors", "--no-warnings",
        CHANNEL_URL,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr[:300]}")

    videos = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        dur = e.get("duration") or 0
        if dur < MIN_DURATION_SECS:   # skip reels / shorts
            continue
        videos.append({
            "id":           e.get("id", ""),
            "title":        e.get("title", ""),
            "url":          e.get("webpage_url") or f"https://www.youtube.com/watch?v={e.get('id', '')}",
            "duration":     dur,
            "duration_str": e.get("duration_string", ""),
            "description":  e.get("description", ""),
            "hashtags":     e.get("tags", []),
        })
    return videos


def enrich_video(video_id: str) -> dict:
    """Fetch full description + tags for one video."""
    cmd = YTDLP + ["--dump-json", "--no-warnings", "--skip-download",
                   f"https://www.youtube.com/watch?v={video_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    try:
        d = json.loads(r.stdout.strip())
        return {
            "duration":    d.get("duration"),
            "description": (d.get("description") or "")[:500],  # cap size
            "hashtags":    d.get("tags", []),
        }
    except Exception:
        return {}


# ── main ─────────────────────────────────────────────────────────────────────

def run_sync(dry_run: bool = False, enrich: bool = False):
    print(f"[youtube_sync] channel : {CHANNEL_URL}")
    print(f"[youtube_sync] dry_run={dry_run}  enrich={enrich}\n")

    known = load_known_titles()
    print(f"[youtube_sync] Ingested titles loaded : {len(known)}")

    print("[youtube_sync] Fetching channel listing via yt-dlp …")
    all_videos = fetch_channel_videos()
    print(f"[youtube_sync] Channel total (≥2min)  : {len(all_videos)}")

    missing = [v for v in all_videos if normalise(v["title"]) not in known]
    print(f"[youtube_sync] Missing (not ingested) : {len(missing)}")

    if dry_run:
        print("\n── DRY RUN — missing videos ──")
        for v in missing:
            print(f"  • {v['title']}")
            print(f"    {v['url']}")
        print(f"\nTotal missing: {len(missing)}  |  No file written.")
        return

    if enrich:
        print("[youtube_sync] Enriching metadata …")
        for i, v in enumerate(missing, 1):
            details = enrich_video(v["id"])
            v.update(details)
            print(f"  [{i}/{len(missing)}] {v['title'][:70]}")

    output = {
        "updated_at":                  datetime.now(timezone.utc).isoformat(),
        "channel":                     CHANNEL_URL,
        "min_duration_secs":           MIN_DURATION_SECS,
        "total_channel_videos_ge2min": len(all_videos),
        "missing_count":               len(missing),
        "videos":                      missing,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[youtube_sync] ✅ Written → {OUTPUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enrich",  action="store_true",
                        help="Fetch full description/tags per video (slow)")
    args = parser.parse_args()
    run_sync(dry_run=args.dry_run, enrich=args.enrich)