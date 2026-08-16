"""
Scores hawkish/dovish tone shift from the retrieved chunks. Structured JSON
output (not free text) because this score needs to be machine-readable for
the synthesis agent (Day 3) and for the regime-comparison logic - a prose
answer here would need to be re-parsed downstream, which is fragile.

Groq/Llama doesn't have a strict JSON mode guarantee, so this parses
defensively and retries once with a stricter instruction if the first
response isn't valid JSON, rather than crashing the graph run.
"""
import json
import os
import re

from groq import Groq

from graph.state import GraphState, ToneScore

GROQ_MODEL = "llama-3.3-70b-versatile"

SCORING_SYSTEM_PROMPT = """You are a monetary policy tone analyst. Given excerpts from Federal \
Reserve communications, score the hawkish/dovish tone shift.

Respond with ONLY a JSON object, no other text, in this exact shape:
{
  "direction": "hawkish" | "dovish" | "neutral" | "mixed",
  "score": <float between -1.0 (max dovish) and 1.0 (max hawkish)>,
  "rationale": "<2-3 sentence explanation citing specific dates/documents>",
  "based_on": ["<doc_type date>", ...]
}

"mixed" means different documents in the excerpts point in different directions.
Base the score ONLY on the provided excerpts - do not use outside knowledge of
what the Fed did after the excerpts' dates."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown code fences if the model added them despite instructions
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def scoring_node(state: GraphState) -> GraphState:
    if not state.get("needs_scoring"):
        return {}

    chunks = state.get("retrieved_chunks", [])
    errors = list(state.get("errors", []))

    if not chunks:
        errors.append("scoring_node: no retrieved chunks to score")
        return {"tone_score": None, "errors": errors}

    context = "\n\n---\n\n".join(
        f"[{c['doc_type'].upper()} | {c['date']} | {c['title']}]\n{c['text']}" for c in chunks
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query context: {state['query']}\n\nExcerpts:\n\n{context}"},
    ]

    parsed = None
    for attempt in range(2):
        resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.1)
        raw = resp.choices[0].message.content
        try:
            parsed = _extract_json(raw)
            break
        except (json.JSONDecodeError, AttributeError):
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "That wasn't valid JSON. Respond with ONLY the JSON object, nothing else."})
            else:
                errors.append(f"scoring_node: failed to parse JSON after retry, raw: {raw[:200]}")

    tone_score: ToneScore | None = None
    if parsed:
        tone_score = {
            "direction": parsed.get("direction", "neutral"),
            "score": float(parsed.get("score", 0.0)),
            "rationale": parsed.get("rationale", ""),
            "based_on": parsed.get("based_on", []),
        }

    return {"tone_score": tone_score, "errors": errors}
