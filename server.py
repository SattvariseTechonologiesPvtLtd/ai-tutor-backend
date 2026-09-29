"""
server.py — DEFTXR Anatomy AI  (FastAPI entrypoint)

Run:
    uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import retriever
import textbook_retriever
import chat
import summarizer
import classifier
import tts

# ── Env ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
load_dotenv(BASE / ".env")
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
STORAGE_PL_DIR   = BASE / "qdrant_storage_playlists"
STORAGE_BOOK_DIR = BASE / "qdrant_storage"
BM25S_PL_DIR     = BASE / "bm25s_index_playlists"
BM25S_BOOK_DIR   = BASE / "bm25s_index_book"        # ← textbook index
MODELS_DIR       = BASE / "models"
FIGURES_DIR      = BASE / "extracted_figures" / "images"
FIGURE_INDEX     = BASE / "extracted_figures" / "figure_index.json"

BM25S_PL_DIR.mkdir(exist_ok=True)
BM25S_BOOK_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Figure index (label → metadata) ───────────────────────────────────
_figure_index: dict = {}
if FIGURE_INDEX.exists():
    with open(FIGURE_INDEX, "r", encoding="utf-8") as _f:
        for _entry in json.load(_f):
            # key: "1.2" derived from "FIGURE 1.2"
            label = _entry.get("fig_label", "").replace("FIGURE", "").strip()
            _figure_index[label] = _entry

# ── Config ─────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DS_KEY      = os.getenv("DEEPSEEK_API_KEY", os.getenv("deepseek_api_key", ""))
DS_URL      = "https://api.deepseek.com"

ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://localhost"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")


# ── Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading embedding model…")
    embedder = SentenceTransformer(
        EMBED_MODEL, trust_remote_code=True, cache_folder=str(MODELS_DIR)
    )

    log.info("Opening playlist Qdrant client…")
    qdrant_pl   = QdrantClient(path=str(STORAGE_PL_DIR))

    log.info("Opening textbook Qdrant client…")
    qdrant_book = QdrantClient(path=str(STORAGE_BOOK_DIR))

    log.info("Initialising playlist retriever…")
    await asyncio.to_thread(retriever.init, qdrant_pl, embedder, BM25S_PL_DIR)

    log.info("Initialising textbook retriever…")
    await asyncio.to_thread(textbook_retriever.init, qdrant_book, embedder, BM25S_BOOK_DIR)

    log.info("Initialising chat client…")
    chat.init(api_key=DS_KEY, base_url=DS_URL)

    log.info("Initialising summariser client…")
    summarizer.init(api_key=DS_KEY, base_url=DS_URL)

    log.info("Initialising classifier…")
    classifier.init(api_key=DS_KEY, base_url=DS_URL)

    log.info("Initialising Sarvam TTS client…")
    tts.init()

    log.info("Server ready ✓")
    yield

    qdrant_pl.close()
    textbook_retriever.close()


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="DEFTXR Anatomy AI", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


from openai import AsyncOpenAI

# ── Figure relevance checker ───────────────────────────────────────────
async def _is_figure_relevant(client: AsyncOpenAI, question: str, fig: dict) -> bool:
    """Ask DeepSeek if a figure is relevant to the question. Returns True/False."""
    try:
        resp = await client.chat.completions.create(
            model       = "deepseek-chat",
            max_tokens  = 10,
            temperature = 0.0,
            messages    = [{
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Figure: {fig['fig_label']}\n"
                    f"Caption: {fig['caption'][:300]}\n\n"
                    "Is this figure directly relevant to the question? "
                    'Reply with {"relevant":true} or {"relevant":false} only.'
                ),
            }],
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text).get("relevant", False)
    except Exception:
        return True   # default to showing if check fails


async def _filter_figures(client: AsyncOpenAI, question: str, images: list[dict]) -> list[dict]:
    """Run relevance checks in parallel, return only relevant figures."""
    if not images:
        return []
    results = await asyncio.gather(*[_is_figure_relevant(client, question, img) for img in images])
    filtered = []
    for img, keep in zip(images, results):
        if keep:
            filtered.append(img)
            log.info("✅ FIGURE ACCEPTED  %s — %s", img["fig_label"], img["caption"].split("\n")[0][:80])
        else:
            log.info("🚫 FIGURE REJECTED  %s — %s", img["fig_label"], img["caption"].split("\n")[0][:80])
    return filtered



class AskRequest(BaseModel):
    question: str
    thinking: bool       = False
    history:  list[dict] = []
    refined:  bool       = False   # True when sent after MCQ clarification


class SummarizeRequest(BaseModel):
    history: list[dict] = []


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    SSE stream:
        data: <token>           — streamed answer tokens
        data: [DONE]            — signals end of answer
        data: [SOURCES] <json>  — source cards for the frontend
    """
    if not req.question.strip():
        raise HTTPException(400, "Empty question.")

    # ── Guardrail (skip if already refined by MCQ answers) ────────────
    if not req.refined:
        label = await classifier.classify(req.question)

        if label == "irrelevant":
            async def _irrelevant():
                msg = "I'm your anatomy tutor — I can only help with biology, anatomy, physiology, and related medical topics. Ask me anything in that space!"
                yield f"data: {json.dumps(msg)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_irrelevant(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        if label == "vague":
            mcq = await classifier.gather_context(req.question)
            async def _clarify():
                yield f"data: [CLARIFY] {json.dumps(mcq)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_clarify(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    # ── End guardrail ─────────────────────────────────────────────────

    # 1. Retrieve from both sources sequentially — embedder is not thread-safe
    # (nomic-embed rotary cache corrupts under concurrent encode() calls)
    pl_ranked   = await asyncio.to_thread(retriever.retrieve, req.question)
    book_ranked = await asyncio.to_thread(textbook_retriever.retrieve, req.question)
    pl_ranked   = pl_ranked   or []
    book_ranked = book_ranked or []

    # 2. Merge + re-rank across both sources (threshold=0.15 drops noise)
    from reranker import merge
    ranked = merge(pl_ranked, book_ranked, top_k=6, merge_threshold=0.12)

    if not ranked:
        raise HTTPException(422, "No relevant context found for that question.")

    # 3. Build context string for the LLM
    context = retriever.build_context(ranked)

    # 4. Build source cards — resolve figures for book chunks
    from urllib.parse import quote
    ds_client = AsyncOpenAI(api_key=DS_KEY, base_url=DS_URL)

    # Collect raw figures per ranked result
    raw_images_per = []
    for r in ranked:
        if r.get("_source") == "book":
            imgs = []
            for ref in r.get("figure_refs", []):
                fig = _figure_index.get(ref.strip())
                if fig:
                    imgs.append({
                        "fig_label": fig["fig_label"],
                        "caption"  : fig["caption"],
                        "url"      : f"/figures/{quote(fig['filename'])}",
                    })
            raw_images_per.append(imgs)
        else:
            raw_images_per.append([])

    # Filter all figures in parallel
    filtered_images_per = await asyncio.gather(*[
        _filter_figures(ds_client, req.question, imgs)
        for imgs in raw_images_per
    ])

    sources = []
    for i, (r, images) in enumerate(zip(ranked, filtered_images_per)):
        if r.get("_source") == "book":
            sources.append({
                "index"  : i + 1,
                "title"  : f"Ch {r.get('chapter_num', '')} — {r.get('chapter_title', 'Textbook')}",
                "section": r.get("section_title", ""),
                "source" : "book",
                "score"  : round(r.get("_merged", r.get("_fused", 0.0)), 4),
                "images" : images,
            })
        else:
            sources.append({
                "index"  : i + 1,
                "title"  : r.get("video_title", ""),
                "section": r.get("playlist", ""),
                "source" : "playlist",
                "score"  : round(r.get("_merged", r.get("_fused", 0.0)), 4),
            })

    # 5. Stream answer from LLM
    return StreamingResponse(
        chat.stream_answer(
            question = req.question,
            context  = context,
            sources  = sources,
            history  = req.history,
            thinking = req.thinking,
        ),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert a message to speech using the Sarvam TTS API.
    Returns raw WAV audio with Content-Type audio/wav.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text.")
    try:
        audio_bytes = await tts.synthesize(req.text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    from fastapi.responses import Response
    return Response(content=audio_bytes, media_type="audio/wav")


@app.websocket("/ws/stt")
async def stt_proxy(websocket: WebSocket):
    """
    WebSocket proxy: browser → backend → Sarvam STT.
    Browser can't set custom headers on WebSocket, so the backend
    attaches the api-subscription-key header when connecting to Sarvam.
    """
    import asyncio
    import websockets

    api_key = os.getenv("SARVAM_API_KEY") or os.getenv("sarvam_api_key", "")
    log.info("[STT-PROXY] browser connected, key present=%s", bool(api_key))
    if not api_key:
        await websocket.close(code=1011, reason="Sarvam API key not configured")
        return

    await websocket.accept()

    # Forward query params from the browser WS URL to Sarvam
    qs = websocket.url.query or ""
    sarvam_url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{qs}"
    log.info("[STT-PROXY] opening Sarvam socket: %s", sarvam_url)

    try:
        async with websockets.connect(
            sarvam_url,
            additional_headers={"api-subscription-key": api_key},
        ) as sarvam_ws:
            log.info("[STT-PROXY] ✅ Sarvam socket open")

            async def forward_to_sarvam():
                n = 0
                try:
                    async for msg in websocket.iter_text():
                        n += 1
                        if n <= 3 or n % 20 == 0:
                            log.info("[STT-PROXY] → Sarvam msg #%d (%d bytes)", n, len(msg))
                        await sarvam_ws.send(msg)
                except Exception as e:
                    log.info("[STT-PROXY] browser→Sarvam loop ended: %s", e)
                finally:
                    log.info("[STT-PROXY] browser→Sarvam total msgs: %d", n)

            async def forward_to_browser():
                try:
                    async for msg in sarvam_ws:
                        text = msg if isinstance(msg, str) else msg.decode("utf-8", "replace")
                        log.info("[STT-PROXY] ← Sarvam: %s", text[:160])
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception as e:
                    log.info("[STT-PROXY] Sarvam→browser loop ended: %s", e)

            await asyncio.gather(forward_to_sarvam(), forward_to_browser())
            log.info("[STT-PROXY] both loops finished")

    except Exception as e:
        log.error("[STT-PROXY] ❌ error: %s", e)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


@app.post("/summarize")
async def summarize_chat(req: SummarizeRequest):
    if not req.history:
        raise HTTPException(400, "No chat history provided.")

    clean_history = [
        m for m in req.history
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]

    if not clean_history:
        raise HTTPException(400, "Chat history contains no readable messages.")

    return StreamingResponse(
        summarizer.stream_summary(clean_history),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )