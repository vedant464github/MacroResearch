"""
Synthesis: the last node in the graph. Takes whatever the upstream branches
produced (tone_score always, regime_context only if the query triggered it)
and writes a short, cited research note in prose - the actual "analyst
deliverable" this project is supposed to produce, not just raw structured
data.

Design choices:
- Explicitly told to paraphrase Fed language rather than quote at length -
  these are public-domain government documents so there's no legal issue,
  but a research note built from long verbatim pulls reads as scraped, not
  analyzed. Short attributed phrases are fine.
- If regime_context is missing (query didn't need it, or the export was
  stale/absent), the prompt is told to skip that section entirely rather
  than fabricate a market-positioning claim it has no data for.
- Runs even if tone_score is None (e.g. scoring failed) - it should still
  produce a partial note referencing what IS available and noting the gap,
  rather than crash the graph.
"""
import os

from groq import Groq

from graph.state import GraphState

GROQ_MODEL = "llama-3.3-70b-versatile"

SYNTHESIS_SYSTEM_PROMPT = """You are a macro research analyst writing a short internal research \
note. You'll be given a hawkish/dovish tone analysis and, if relevant, a market regime read. \
Write a concise, well-organized note (3-5 short paragraphs) that:

1. States the tone finding clearly (direction and magnitude) with its rationale.
2. If regime context is provided, explicitly addresses whether markets appear to have already \
priced in the tone shift - this comparison is the point of the note, not an afterthought.
3. Cites sources by document type and date, e.g. (FOMC Minutes, 2026-06-17).
4. Paraphrases source language in your own words rather than quoting at length - short phrases \
(under 10 words) in quotes are fine, longer verbatim reproduction is not.
5. If regime context is missing or stale, do not speculate about market positioning - say so \
explicitly instead of guessing.
6. Ends with a one-line "Bottom line" summary.

Write in a professional, analytical register - this reads like an internal desk note, not a \
blog post or a chat response."""


def _format_tone_score(tone_score: dict | None) -> str:
    if not tone_score:
        return "Tone scoring unavailable for this query."
    return (
        f"Direction: {tone_score['direction']}\n"
        f"Score: {tone_score['score']} (-1.0 max dovish to +1.0 max hawkish)\n"
        f"Rationale: {tone_score['rationale']}\n"
        f"Based on: {', '.join(tone_score.get('based_on', []))}"
    )


def _format_regime_context(regime_context: dict | None) -> str:
    if not regime_context:
        return "No regime context requested or available for this query."
    staleness_note = " (NOTE: this regime read may be stale, treat with caution)" if regime_context.get("stale") else ""
    return (
        f"Current regime: {regime_context['current_regime']} "
        f"(probability {regime_context['regime_probability']}){staleness_note}\n"
        f"Prior regime: {regime_context['prior_regime']}\n"
        f"Regime since: {regime_context['regime_since']}\n"
        f"As of: {regime_context['as_of_date']}"
    )


def synthesis_node(state: GraphState) -> GraphState:
    query = state["query"]
    tone_score = state.get("tone_score")
    regime_context = state.get("regime_context")

    user_content = (
        f"Research question: {query}\n\n"
        f"=== Tone Analysis ===\n{_format_tone_score(tone_score)}\n\n"
        f"=== Market Regime Context ===\n{_format_regime_context(regime_context)}"
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return {"final_answer": resp.choices[0].message.content}
