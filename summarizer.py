"""
summarizer.py — DEFTXR Anatomy AI  (Chat Summariser)

Responsibilities:
  - Accept the full chat history from the frontend
  - Call the LLM directly — NO retrieval, NO RAG, NO Qdrant
  - Return a structured summary as plain text (streamed SSE)
  - The /summarize endpoint (registered in server.py) calls this module

Usage:
    import summarizer
    summarizer.init(api_key=DS_KEY, base_url=DS_URL)
    # then use via FastAPI endpoint
"""

import json
import logging
from openai import OpenAI

log = logging.getLogger("summarizer")

_client: OpenAI | None = None

SYSTEM_PROMPT = """You are a precise academic summariser for an anatomy tutoring system called DEFTXR.

Your task: summarise the provided chat conversation between the student and the AI tutor.

Structure your summary as follows:

## Session Summary

### Topics Covered
List every anatomy topic discussed in the session.

### Key Concepts Explained
For each topic, bullet-point the core facts, definitions, or mechanisms the tutor explained.

### Student Questions
List the student's main questions or areas of confusion.

### Important Takeaways
3–5 concise bullet points of the most important things the student should remember from this session.

### Recommended Review
Suggest 2–3 specific anatomy areas the student should revisit based on what was asked.

---
Be accurate, concise, and academically appropriate. Do not fabricate information not present in the conversation.
"""


def init(api_key: str, base_url: str) -> None:
    global _client
    _client = OpenAI(api_key=api_key, base_url=base_url)
    log.info("Summariser client initialised ✓")


def _format_history_for_prompt(history: list[dict]) -> str:
    """Convert message list to a readable transcript."""
    lines = []
    for msg in history:
        role = "Student" if msg.get("role") == "user" else "Tutor"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


async def stream_summary(history: list[dict]):
    """
    SSE generator — yields:
        data: <token>
        data: [DONE]
    No [SOURCES] frame — summaries have no retrieval sources.
    """
    if _client is None:
        yield "data: [ERROR] Summariser not initialised.\n\n"
        yield "data: [DONE]\n\n"
        return

    if not history:
        yield "data: [ERROR] No chat history provided to summarise.\n\n"
        yield "data: [DONE]\n\n"
        return

    transcript = _format_history_for_prompt(history)

    user_prompt = f"""Please summarise the following anatomy tutoring session:\n\n{transcript}"""

    try:
        stream = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
            max_tokens=5096,
            temperature=0.3,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                # Escape newlines so they survive SSE transport
                token = delta.content.replace("\n", "\n")
                yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        log.error("Summariser LLM error: %s", exc)
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"