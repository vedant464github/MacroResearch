"""
Thin REST wrapper around the same LangGraph pipeline the MCP server uses.
Both entry points (MCP tools, this API) call build_graph() and get identical
behavior - no logic duplicated here, this file is just transport.

Why both MCP and REST exist: MCP is the portfolio differentiator (pluggable
into Claude Desktop / other MCP hosts), REST is what makes this demoable to
someone who just wants to curl an endpoint or hit it from a browser/Postman
without any MCP client at all. Per the fallback cut order, this is explicitly
the first thing to drop if Day 3 runs short - build synthesis + README first,
come back to this only once those are solid.
"""
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.build import build_graph

app = FastAPI(
    title="MacroPilot API",
    description="Multi-agent macro research copilot over Federal Reserve communications.",
    version="0.1.0",
)

_graph = build_graph()


class BriefingRequest(BaseModel):
    query: str


class BriefingResponse(BaseModel):
    query: str
    final_answer: str | None
    date_cutoff_applied: str | None
    tone_score: dict | None
    regime_context: dict | None
    sources: list[dict]
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/briefing", response_model=BriefingResponse)
def briefing(req: BriefingRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = _graph.invoke({"query": req.query, "errors": []})

    sources = [
        {"doc_type": c["doc_type"], "date": c["date"], "title": c["title"], "url": c["url"]}
        for c in result.get("retrieved_chunks", [])
    ]
    seen = set()
    deduped_sources = []
    for s in sources:
        key = (s["doc_type"], s["date"], s["title"])
        if key not in seen:
            seen.add(key)
            deduped_sources.append(s)

    return BriefingResponse(
        query=req.query,
        final_answer=result.get("final_answer"),
        date_cutoff_applied=result.get("date_cutoff"),
        tone_score=result.get("tone_score"),
        regime_context=result.get("regime_context"),
        sources=deduped_sources,
        warnings=result.get("errors", []),
    )
