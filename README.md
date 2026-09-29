# DEFTXR AI Tutor — Backend

FastAPI backend for the DEFTXR anatomy & physiology AI tutor.

Retrieval runs over two corpora — YouTube lecture transcripts and the textbook — using
**Qdrant** (vector) + **BM25s** (keyword), fuses the scores, then streams a **DeepSeek**
answer over SSE. Speech is handled by **Sarvam AI** (TTS + real-time STT).

---

## Requirements

- **Python 3.11 or newer** (3.13 tested)
- ~2 GB free disk — the embedding model (`nomic-ai/nomic-embed-text-v1.5`, ~550 MB)
  downloads on first run

---

## 1. Get the code

```bash
git clone https://github.com/SattvariseTechonologiesPvtLtd/ai-tutor-backend.git
cd ai-tutor-backend
```

---

## 2. Create and activate a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows — PowerShell

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, allow it for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

### Windows — Command Prompt (cmd)

```cmd
py -3.13 -m venv venv
venv\Scripts\activate.bat
```

Your prompt should now start with `(venv)`. Everything below runs inside it.

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch:** `requirements.txt` deliberately omits `torch` because the correct wheel is
> platform-specific. Install it **before** the rest:
>
> - **NVIDIA GPU (CUDA 12.4)** — Windows / Linux:
>   ```bash
>   pip install torch --index-url https://download.pytorch.org/whl/cu124
>   ```
> - **CPU only** — macOS or a machine without an NVIDIA GPU:
>   ```bash
>   pip install torch
>   ```
>
> `sentence-transformers` pulls in a CPU build automatically if you skip this step, but
> installing it explicitly first avoids a large download being replaced later.

---

## 4. Create the `.env` file

Create a file named **`.env`** in the project root (next to `server.py`):

```ini
deepseek_api_key = your_deepseek_key_here
sarvam_api_key   = your_sarvam_key_here
```

| Key | Used for | Get it from |
|---|---|---|
| `deepseek_api_key` | Chat answers, query classification, chat summaries | https://platform.deepseek.com |
| `sarvam_api_key` | Text-to-speech (`/tts`) **and** speech-to-text (`/ws/stt`) | https://dashboard.sarvam.ai |

`.env` is git-ignored. **Never commit it.**

---

## 5. Run the server

```bash
uvicorn server:app --reload --port 8000
```

The first start takes **20–30 seconds** — it loads the embedding model and builds the
BM25s indices before serving. Wait for:

```
INFO:     Server ready ✓
INFO:     Application startup complete.
```

Verify it is up:

```bash
curl http://localhost:8000/health      # -> {"status":"ok"}
```

Open **http://localhost:8000/docs** for the interactive API docs.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ask` | Main chat — SSE stream (answer tokens, `[SOURCES]`, `[THINKING]`, `[CLARIFY]`) |
| `POST` | `/summarize` | Summarise a chat session (SSE) |
| `POST` | `/tts` | Text → speech (Sarvam `bulbul:v3`), returns WAV |
| `WS` | `/ws/stt` | Real-time speech-to-text proxy → Sarvam `saaras:v3-realtime` |
| `GET` | `/figures/*` | Static textbook figures |

### Why `/ws/stt` is a proxy

Sarvam authenticates its STT WebSocket with the `api-subscription-key` **HTTP header**,
and a browser `WebSocket` cannot set custom headers. The backend opens the upstream socket
with the header, then relays frames in both directions. The API key never reaches the
browser.

---

## Project layout

| File | Role |
|---|---|
| `server.py` | FastAPI entrypoint — lifespan, routes, `/ws/stt` proxy |
| `classifier.py` | Classifies a query as `ok` / `vague` / `irrelevant` |
| `retriever.py` | Playlist transcript retrieval (Qdrant + BM25s) |
| `textbook_retriever.py` | Textbook retrieval (Qdrant + BM25s) |
| `reranker.py` | Score fusion / merge across sources |
| `chat.py` | DeepSeek streaming answer with RAG context |
| `summarizer.py` | Session summary generation |
| `tts.py` | Sarvam TTS wrapper |
| `config.py` | Paths, model config, collection names |
| `build_textbook_bm25s_index.py` | Rebuild the textbook BM25s index |

**Data directories (tracked in the repo):** `qdrant_storage/`, `qdrant_storage_playlists/`,
`bm25s_index*/`, `extracted_figures/`. The `models/` folder (downloaded embedding model) is
git-ignored.

---

## Troubleshooting

**`ModuleNotFoundError` right after install** — check the venv is actually active and that
`pip` points at it: `which pip` (macOS/Linux) or `where pip` (Windows) should show a path
inside `venv/`.

**`AttributeError: 'NomicBertModel' object has no attribute 'get_extended_attention_mask'`**
or `/ask` returning HTTP 500 — `transformers` is too new. The nomic embedding model's custom
code needs `transformers<5`, which `requirements.txt` pins. Verify with:

```bash
python -c "import transformers; print(transformers.__version__)"   # must be < 5
```

**`ModuleNotFoundError: No module named 'websockets'`** — required by the `/ws/stt` proxy:

```bash
pip install websockets
```

**Port 8000 already in use** — find and stop the old process, or run on another port:
`uvicorn server:app --reload --port 8001`.

**Frontend shows "Offline"** — the backend isn't answering `/health` yet. It needs the full
20–30 s startup.
