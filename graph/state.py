"""
Shared state that flows through the LangGraph graph. Every node reads
from and writes to this single dict-like object - that's the core
LangGraph pattern (as opposed to nodes passing custom args to each other).

Nodes should return ONLY the keys they change, not the full state - when
nodes run in parallel (scoring + regime both fire after retrieval), each
key can only receive one write per step unless it has a reducer. `errors`
uses operator.add so parallel branches can each append to it safely; every
other key uses default "last write wins" semantics, which is fine since
only one branch ever writes to tone_score or regime_context.
"""
import operator
from typing import TypedDict, Literal, Optional, Annotated


class ToneScore(TypedDict):
    direction: Literal["hawkish", "dovish", "neutral", "mixed"]
    score: float          # -1.0 (max dovish) to +1.0 (max hawkish)
    rationale: str
    based_on: list[str]   # doc identifiers the score was derived from


class RegimeContext(TypedDict):
    as_of_date: str
    current_regime: str
    regime_probability: float
    prior_regime: str
    regime_since: str
    key_indicators: dict
    stale: bool            # True if regime_export.json is older than N days - lets
                            # synthesis flag "this regime read may be outdated" instead
                            # of silently trusting a stale file


class GraphState(TypedDict, total=False):
    query: str
    needs_scoring: bool
    needs_regime: bool
    date_cutoff: Optional[str]

    retrieved_chunks: list[dict]

    tone_score: Optional[ToneScore]
    regime_context: Optional[RegimeContext]

    final_answer: Optional[str]   # filled by synthesis node on Day 3
    errors: Annotated[list[str], operator.add]            # non-fatal node errors, surfaced rather than swallowed
