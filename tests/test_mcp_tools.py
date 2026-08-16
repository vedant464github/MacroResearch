"""
Tests the three MCP tool functions directly as Python calls, bypassing the
MCP protocol/transport layer entirely. This validates the actual business
logic (the thing that matters for the demo) without needing Node/npx for
the Inspector UI. Once Claude Desktop is wired up later, that exercises the
real protocol layer - this script is just for fast local iteration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.server import get_macro_briefing, search_fed_speech, regime_context

print("=" * 80)
print("TEST 1: get_macro_briefing")
print("=" * 80)
result = get_macro_briefing("Has the Fed's tone on inflation shifted more hawkish in the last two meetings?")
print("date_cutoff_applied:", result["date_cutoff_applied"])
print("tone_score:", result["tone_score"])
print("regime_context:", result["regime_context"])
print("num sources:", len(result["sources"]))
print("warnings:", result["warnings"])

print()
print("=" * 80)
print("TEST 2: search_fed_speech")
print("=" * 80)
result = search_fed_speech("labor market risks", k=3)
print("date_cutoff_applied:", result["date_cutoff_applied"])
for r in result["results"]:
    print(f"  - {r['doc_type']} {r['date']}: {r['title']}")

print()
print("=" * 80)
print("TEST 3: regime_context")
print("=" * 80)
result = regime_context()
print(result)
