"""
classifier.py — Query classification and context gathering for DEFTXR Anatomy AI

Three outcomes:
  "irrelevant" — not biology/anatomy/medicine related
  "vague"      — medically relevant but too broad to retrieve well
  "ok"         — specific enough to retrieve directly
"""

import json
import logging
from openai import AsyncOpenAI

log = logging.getLogger("classifier")

_client: AsyncOpenAI | None = None

def init(api_key: str, base_url: str) -> None:
    global _client
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    log.info("Classifier client ready")


CLASSIFY_PROMPT = """You are a query classifier for an anatomy and physiology AI tutor.

Classify the student's question into exactly one of three categories:

"irrelevant" — The question has nothing to do with biology, anatomy, physiology, medicine, or health sciences.
  Examples: "who won the world cup", "write me a poem", "what is python", "best pokemon"

"vague" — The question IS medically/biologically relevant but is too broad or general to give a focused answer.
  A question is vague if it:
  - Names a large body region or system without specifying what aspect (e.g. "tell me about the hand", "explain the brain")
  - Asks to "explain" or "tell me about" an entire organ, system, or broad topic without a specific angle
  - Could reasonably be answered from 10+ different directions (anatomy, blood supply, nerve supply, clinical, embryology, etc.)
  Examples: "tell me about the knee", "explain the heart", "what about the lungs", "describe the shoulder"

"ok" — The question is medically relevant AND specific enough to retrieve focused content.
  A question is ok if it:
  - Asks about a specific structure, pathway, mechanism, syndrome, or clinical condition
  - Has a clear single angle even if the topic is broad (e.g. "what is the blood supply of the spinal cord")
  - Is a follow-up that references prior context
  Examples: "what are the rotator cuff muscles", "how does negative feedback work", "what is Brown-Sequard syndrome",
            "what is the blood supply of the hand", "difference between UMN and LMN lesions"

Reply with ONLY a JSON object, nothing else:
{"classification": "irrelevant"} or {"classification": "vague"} or {"classification": "ok"}"""


GATHER_PROMPT = """You are a context-gathering assistant for an anatomy and physiology AI tutor.
A student asked a broad question. Generate 3 focused questions to understand exactly what they want to learn.

Important — the UI lets students:
- Select MULTIPLE options per question (so options should be genuinely combinable)
- Type their own custom answer
- Skip a question entirely

Rules:
- Questions must be directly relevant to what the student asked about
- Each question must have exactly 4 options that are COMBINABLE (not mutually exclusive)
- Options should represent different angles that can logically co-exist
- Tailor question themes to the topic:
  * Limbs/joints: focus on aspect (bones, muscles, nerves, vessels), clinical conditions, movement types, surgical relevance
  * Organs: structure, blood supply, nerve supply, function, pathology, embryology
  * Systems: components, regulation mechanism, common pathologies, clinical tests, pharmacology
  * Regions: boundaries/contents, neurovascular supply, clinical conditions, surgical anatomy
- Make questions feel natural, not like a form — like a tutor asking to understand the student
- Keep option labels SHORT (2-5 words max)
- Question 1: what aspect/focus, Question 2: what depth/context, Question 3: specific angle (clinical, surgical, exam-style etc.)

Reply with ONLY a JSON object, nothing else:
{
  "questions": [
    {
      "id": "q1",
      "question": "What aspects do you want covered?",
      "options": [
        {"id": "a", "label": "Bones & joints"},
        {"id": "b", "label": "Muscles & movements"},
        {"id": "c", "label": "Nerve supply"},
        {"id": "d", "label": "Blood supply"}
      ]
    },
    {
      "id": "q2",
      "question": "What level of detail?",
      "options": [
        {"id": "a", "label": "Quick overview"},
        {"id": "b", "label": "Detailed anatomy"},
        {"id": "c", "label": "Clinical relevance"},
        {"id": "d", "label": "Surgical anatomy"}
      ]
    },
    {
      "id": "q3",
      "question": "Any specific focus?",
      "options": [
        {"id": "a", "label": "Common pathologies"},
        {"id": "b", "label": "Exam high-yields"},
        {"id": "c", "label": "Embryology"},
        {"id": "d", "label": "No preference"}
      ]
    }
  ]
}"""


async def classify(question: str) -> str:
    """Returns 'irrelevant', 'vague', or 'ok'."""
    try:
        resp = await _client.chat.completions.create(
            model       = "deepseek-chat",
            max_tokens  = 20,
            temperature = 0.0,
            messages    = [
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user",   "content": f"Question: {question}"},
            ],
        )
        text = resp.choices[0].message.content.strip()
        result = json.loads(text).get("classification", "ok")
        log.info("🔍 CLASSIFY  '%s'  →  %s", question[:60], result)
        return result
    except Exception as e:
        log.warning("Classifier error: %s — defaulting to ok", e)
        return "ok"


async def gather_context(question: str) -> dict:
    """Returns MCQ JSON for vague questions."""
    try:
        resp = await _client.chat.completions.create(
            model       = "deepseek-chat",
            max_tokens  = 400,
            temperature = 0.3,
            messages    = [
                {"role": "system", "content": GATHER_PROMPT},
                {"role": "user",   "content": f"Student's broad question: {question}"},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        log.warning("Context gatherer error: %s", e)
        return {"questions": []}