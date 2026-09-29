"""
tts.py — Sarvam AI Text-to-Speech wrapper for DEFTXR Anatomy AI

No extra pip install needed — uses httpx (already a project dependency).

Sarvam TTS docs: https://docs.sarvam.ai/api-reference-docs/endpoints/text-to-speech
"""

import base64
import logging
import os
import re

import httpx

log = logging.getLogger("tts")

# ── Config ─────────────────────────────────────────────────────────────
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# bulbul:v3 — latest model, better quality, supports longer text in one shot
DEFAULT_MODEL    = "bulbul:v3"
DEFAULT_SPEAKER  = "priya"     # clear female en-IN voice; change to e.g. "aditya" for male
DEFAULT_LANGUAGE = "en-IN"
DEFAULT_PACE     = 1.0         # 0.5 – 2.0 for v3

# Resolved lazily inside init() so that load_dotenv() in server.py has already
# run by the time we read the value.  .strip() handles dotenv files that use
# "key = value" syntax (spaces around =) and AWS/EC2 env vars with trailing
# whitespace injected by shell expansions or Parameter Store.
SARVAM_API_KEY: str = ""

_client: httpx.AsyncClient | None = None


def init() -> None:
    """Call once at server startup — validates config and creates the shared HTTP client."""
    global _client, SARVAM_API_KEY

    # Read here, after load_dotenv() has populated os.environ
    SARVAM_API_KEY = (
        (os.getenv("SARVAM_API_KEY") or os.getenv("sarvam_api_key") or "")
        .strip()
    )

    _client = httpx.AsyncClient(timeout=30.0)

    if not SARVAM_API_KEY:
        log.warning(
            "⚠️  SARVAM_API_KEY / sarvam_api_key not found in environment — "
            "TTS endpoint (/tts) will return 503 until the key is set."
        )
    else:
        log.info(
            "✅ Sarvam TTS API ready  model=%s  speaker=%s  lang=%s",
            DEFAULT_MODEL, DEFAULT_SPEAKER, DEFAULT_LANGUAGE,
        )


def _strip_markdown(text: str) -> str:
    """Remove Markdown syntax so the TTS doesn't read out '**', '#', backticks, etc."""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)  # bold/italic
    text = re.sub(r'#{1,6}\s+', '', text)                                   # headings
    text = re.sub(r'`{3}[\s\S]*?`{3}', '', text)                           # code blocks
    text = re.sub(r'`[^`]+`', '', text)                                     # inline code
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)                             # images
    text = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', text)                        # links → label
    text = re.sub(r'^\s*[-*>]\s+', '', text, flags=re.MULTILINE)           # bullets/quotes
    text = re.sub(r'\n{2,}', ' ', text)                                     # collapse newlines
    text = re.sub(r'\s{2,}', ' ', text)                                     # collapse spaces
    return text.strip()


async def synthesize(text: str) -> bytes:
    """
    Convert *text* to WAV audio bytes via the Sarvam TTS API.

    Returns raw WAV bytes.
    Raises RuntimeError on config / API errors, ValueError for empty input.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "Sarvam API key not configured. "
            "Set SARVAM_API_KEY (or sarvam_api_key) in your .env file."
        )
    if _client is None:
        raise RuntimeError("tts.init() has not been called.")

    clean = _strip_markdown(text)
    if not clean:
        raise ValueError("Text is empty after stripping markdown.")

    # bulbul:v3 supports long text natively — no chunking needed for typical messages
    log.info("🔊 TTS synthesize  chars=%d  speaker=%s  model=%s",
             len(clean), DEFAULT_SPEAKER, DEFAULT_MODEL)

    payload = {
        "text":                  clean,
        "target_language_code":  DEFAULT_LANGUAGE,
        "speaker":               DEFAULT_SPEAKER,
        "pace":                  DEFAULT_PACE,
        "model":                 DEFAULT_MODEL,
        "speech_sample_rate":    22050,
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }

    resp = await _client.post(SARVAM_TTS_URL, json=payload, headers=headers)

    if resp.status_code != 200:
        log.error("Sarvam TTS HTTP %s: %s", resp.status_code, resp.text[:400])
        raise RuntimeError(
            f"Sarvam TTS returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    audios = data.get("audios", [])
    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio in response.")

    audio_bytes = base64.b64decode(audios[0])
    log.info("✅ TTS audio ready  bytes=%d", len(audio_bytes))
    return audio_bytes