"""
MCP server exposing MacroPilot's pipeline as three tools, pluggable into
Claude Desktop or any other MCP host.

Tools:
  - get_macro_briefing(query): runs the full graph (retrieval + scoring +
    conditionally regime) and returns structured results. This is the main
    "analyst-in-a-box" entry point.
  - search_fed_speech(query, k): raw retrieval only, no LLM scoring - useful
    when the host just wants source material, not analysis.
  - regime_context(): returns QuantRisk's latest exported regime state
    directly, no query needed.

get_macro_briefing returns both the synthesized prose note (final_answer)
and the underlying structured data (tone_score, regime_context, sources) -
the prose is the primary deliverable, the structured fields let a caller
build its own UI on top without re-parsing the note.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from graph.build import build_graph
from graph.nodes.retrieval import _chain
from rag.chain import detect_recency_filter
mcp = FastMCP("macropilot")
_graph = build_graph()

@mcp.tool()
def get_macro_briefing(query: str) -> dict:
    """
    Run a full macro research query through MacroPilot: retrieves relevant
    Fed communications, scores hawkish/dovish tone shift, and - if the query
    concerns markets/positioning - cross-references QuantRisk's current
    regime read. Use this for questions like "has the Fed turned more
    hawkish recently" or "has the market priced in the latest Fed shift".
    """
    result = _graph.invoke({"query": query, "errors": []})

    sources = [
        {"doc_type": c["doc_type"], "date": c["date"], "title": c["title"], "url": c["url"]}
        for c in result.get("retrieved_chunks", [])
    ]
    # dedupe sources by document identity
    seen = set()
    deduped_sources = []
    for s in sources:
        key = (s["doc_type"], s["date"], s["title"])
        if key not in seen:
            seen.add(key)
            deduped_sources.append(s)

    return {
        "query": query,
        "final_answer": result.get("final_answer"),
        "date_cutoff_applied": result.get("date_cutoff"),
        "tone_score": result.get("tone_score"),
        "regime_context": result.get("regime_context"),
        "sources": deduped_sources,
        "warnings": result.get("errors", []),
    }


@mcp.tool()
def search_fed_speech(query: str, k: int = 6) -> dict:
    """
    Search Federal Reserve statements, minutes, and speeches for relevant
    excerpts, without running tone scoring. Use this when you just need
    source material or citations, not an analytical judgment.
    """
    date_cutoff = detect_recency_filter(query, _chain._meeting_dates())
    hits = _chain.retrieve(query, k=k, date_cutoff=date_cutoff)
    if not hits and date_cutoff:
        hits = _chain.retrieve(query, k=k)

    return {
        "query": query,
        "date_cutoff_applied": date_cutoff,
        "results": [
            {
                "doc_type": h["doc_type"],
                "date": h["date"],
                "title": h["title"],
                "url": h["url"],
                "excerpt": h["text"],
            }
            for h in hits
        ],
    }


@mcp.tool()
def regime_context() -> dict:
    """
    Return QuantRisk's most recently exported market regime read (risk-on/
    risk-off/transition, with probability and key indicators). No query
    needed - always returns the latest export.
    """
    from graph.nodes.regime import regime_node
    result = regime_node({"needs_regime": True, "errors": []})
    return {
        "regime_context": result.get("regime_context"),
        "warnings": result.get("errors", []),
    }


if __name__ == "__main__":
    mcp.run()
