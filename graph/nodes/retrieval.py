"""
Thin wrapper around rag.chain's retrieval + recency-filter logic, so Day 1's
tested code is reused as-is inside the graph rather than duplicated. The
RAGChain instance is created once at module load (embeddings model + Chroma
client are expensive to reinit per call) and shared across graph runs.
"""
from graph.state import GraphState
from rag.chain import RAGChain, detect_recency_filter

_chain = RAGChain()


def retrieval_node(state: GraphState) -> GraphState:
    query = state["query"]
    errors = list(state.get("errors", []))

    try:
        date_cutoff = detect_recency_filter(query, _chain._meeting_dates())
        hits = _chain.retrieve(query, k=8, date_cutoff=date_cutoff)
        if not hits and date_cutoff:
            hits = _chain.retrieve(query, k=8)  # cutoff too aggressive, fall back
    except Exception as e:
        errors.append(f"retrieval_node failed: {e}")
        hits, date_cutoff = [], None

    return {
        "retrieved_chunks": hits,
        "date_cutoff": date_cutoff,
        "errors": errors,
    }
