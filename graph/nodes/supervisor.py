"""
Supervisor: cheap, deterministic keyword-based routing rather than an LLM
classification call. This is a conscious tradeoff for the 3-day build -
an LLM router would be more flexible but adds latency + cost + another
failure mode to debug on Day 2. Keyword routing is auditable and instant;
swap it for an LLM classifier later if query variety grows beyond what
these patterns can catch.

Decisions made here:
- needs_scoring: almost every macro query benefits from a tone read, so
  this defaults True and only turns off for queries that are clearly pure
  lookups ("when is the next FOMC meeting", "who is on the committee").
- needs_regime: only True when the query explicitly asks about markets/
  positioning/pricing - this is the QuantRisk integration and should NOT
  fire on every query, or it dilutes the one thing that makes this project
  different from a plain RAG chatbot.
"""
import re

from graph.state import GraphState

PURE_LOOKUP_PATTERNS = re.compile(
    r"\b(when is|who is|who chairs|committee members|next meeting date|meeting schedule)\b",
    re.I,
)

REGIME_TRIGGER_PATTERNS = re.compile(
    r"\b(market|markets|priced in|pricing|regime|risk[- ]on|risk[- ]off|portfolio|positioning|volatility|cvar)\b",
    re.I,
)


def supervisor_node(state: GraphState) -> GraphState:
    query = state["query"]

    needs_scoring = not PURE_LOOKUP_PATTERNS.search(query)
    needs_regime = bool(REGIME_TRIGGER_PATTERNS.search(query))

    return {
        "needs_scoring": needs_scoring,
        "needs_regime": needs_regime,
    }
