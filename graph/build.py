"""
Graph shape (Day 3 - synthesis added as the fan-in join point):

    supervisor -> retrieval -> [scoring?, regime?] -> synthesis -> END

Both scoring and regime now point to "synthesis" instead of END. LangGraph's
underlying Pregel/BSP execution model runs scoring and regime in parallel
(since route_after_retrieval dispatches both in the same step when needed),
then runs synthesis exactly once after both have completed - no extra
join/barrier code needed, this is standard LangGraph fan-out/fan-in behavior.

If a query only needs one branch (e.g. needs_regime=False), only that branch
runs, and synthesis still fires once afterward with regime_context left as
whatever default state carries (None) - synthesis_node handles that case
explicitly rather than assuming both branches always ran.
"""
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.supervisor import supervisor_node
from graph.nodes.retrieval import retrieval_node
from graph.nodes.scoring import scoring_node
from graph.nodes.regime import regime_node
from graph.nodes.synthesis import synthesis_node


def route_after_retrieval(state: GraphState) -> list[str]:
    """Fan out to whichever of scoring/regime the supervisor flagged as needed.
    LangGraph runs all returned node names as parallel branches."""
    targets = []
    if state.get("needs_scoring"):
        targets.append("scoring")
    if state.get("needs_regime"):
        targets.append("regime")
    return targets or ["scoring"]  # never return an empty list - always score something


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("regime", regime_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "retrieval")
    graph.add_conditional_edges("retrieval", route_after_retrieval, ["scoring", "regime"])
    graph.add_edge("scoring", "synthesis")
    graph.add_edge("regime", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    test_queries = [
        "Has the Fed's tone on inflation shifted more hawkish in the last two meetings?",
        "Has the market already priced in a hawkish shift given the current regime?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 80}\nQUERY: {q}\n{'=' * 80}")
        result = app.invoke({"query": q, "errors": []})
        print("needs_scoring:", result.get("needs_scoring"))
        print("needs_regime:", result.get("needs_regime"))
        print("date_cutoff:", result.get("date_cutoff"))
        print("retrieved chunks:", len(result.get("retrieved_chunks", [])))
        print()
        print("FINAL ANSWER:")
        print(result.get("final_answer"))
        if result.get("errors"):
            print("\nerrors:", result["errors"])
