# MacroPilot

A multi-agent research copilot that reads Federal Reserve communications (FOMC statements, meeting minutes, speeches) and answers the question a junior macro analyst spends hours on manually: **is monetary policy turning more hawkish or dovish, and has the market already priced it in?**

It retrieves relevant primary-source Fed documents (RAG), scores hawkish/dovish tone shift with an LLM agent, cross-references [QuantRisk](../quantrisk) — a separately-built regime-detection engine — to check whether markets have already repriced around that shift, and synthesizes a short cited research note.

This isn't a search-and-summarize chatbot. The differentiator is the QuantRisk cross-reference: tone analysis on its own tells you what the Fed said, but pairing it with an independent regime read is what turns it into an actual trading/positioning signal.

---

## Architecture

```
                    ┌─────────────┐
   query ─────────▶ │ supervisor  │  (keyword-based intent routing)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  retrieval  │  (RAG over Chroma, recency-aware)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌──────────────┐          ┌──────────────┐
      │   scoring    │          │    regime    │
      │ (hawkish/    │          │ (reads       │
      │  dovish, LLM)│          │  QuantRisk   │
      │              │          │  export)     │
      └──────┬───────┘          └──────┬───────┘
             │                         │
             └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │  synthesis  │  (writes cited research note)
                    └──────┬──────┘
                           ▼
                     final answer
```

- **Supervisor**: deterministic keyword routing (not an LLM call) — decides if a query needs tone scoring, regime context, or both. Cheap, fast, auditable.
- **Retrieval**: semantic search over Chroma, with recency-aware date filtering — queries like "last two meetings" resolve against the actual meeting-date index rather than relying on pure embedding similarity, which otherwise has no concept of "recent."
- **Scoring**: LLM agent (Groq/Llama 3.3 70B) that scores hawkish/dovish direction and magnitude from retrieved excerpts, with structured JSON output for downstream use.
- **Regime**: reads QuantRisk's exported market regime state (risk-on/risk-off/transition, HMM-based), with a staleness check so an old export doesn't get silently presented as current.
- **Synthesis**: fan-in node that combines tone score + regime context into a short, cited, professional research note — explicitly instructed to paraphrase rather than quote at length, and to state "no data" honestly rather than speculate.

Both **scoring** and **regime** run in parallel when a query needs both (LangGraph's native fan-out/fan-in), and **synthesis** runs exactly once after whichever branches were needed complete.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM | Groq API, Llama 3.3 70B | Free tier, fast inference |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Local, CPU, no API key |
| Vector DB | Chroma (local, persistent) | Simple, no infra |
| Data source | federalreserve.gov (scraped) + QuantRisk export | Primary sources, no API key needed |
| Orchestration | LangGraph + LangChain | Multi-agent state graph |
| Tool exposure | Official MCP Python SDK (`mcp[cli]`, pinned `<2`) | Pluggable into Claude Desktop |
| API | FastAPI | REST access without an MCP client |

---

## Repo structure

```
macropilot/
├── scraper/fed_scraper.py       # federalreserve.gov scraping
├── ingest/
│   ├── chunk.py                 # paragraph-aware chunking
│   └── embed_and_load.py        # embeddings -> Chroma
├── rag/chain.py                 # retrieval + recency filtering (Day 1 core, reused everywhere)
├── graph/
│   ├── state.py                 # shared LangGraph state schema
│   ├── build.py                 # graph wiring
│   └── nodes/
│       ├── supervisor.py
│       ├── retrieval.py
│       ├── scoring.py
│       ├── regime.py
│       └── synthesis.py
├── mcp_server/server.py         # MCP tools: get_macro_briefing, search_fed_speech, regime_context
├── app/main.py                  # FastAPI wrapper
├── data/
│   ├── raw/                     # scraped Fed documents (JSON)
│   ├── chunks/                  # chunked documents (JSONL)
│   ├── chroma/                  # vector index
│   └── regime/regime_export.json # QuantRisk's exported regime state
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Run the pipeline in order:
```bash
python scraper/fed_scraper.py --years 2024 2025 2026
python ingest/chunk.py
python ingest/embed_and_load.py
python -m graph.build          # sanity check the full graph
uvicorn app.main:app --reload --port 8000   # REST API
```

### QuantRisk integration

MacroPilot reads `data/regime/regime_export.json`, exported from a QuantRisk notebook cell:
```python
import json
from pathlib import Path
from datetime import datetime, timezone

regime_export = {
    "as_of_date": "...", "current_regime": "risk_off", "regime_probability": 0.78,
    "prior_regime": "transition", "regime_since": "...",
    "key_indicators": {"realized_vol": None, "cvar_95": None},
    "exported_at": datetime.now(timezone.utc).isoformat(),
}
Path("regime_export.json").write_text(json.dumps(regime_export, indent=2))
```
Copy the output into `macropilot/data/regime/regime_export.json`. If the export is more than 7 days old, MacroPilot flags it as stale rather than treating it as current.

### Connecting to Claude Desktop (optional)

1. Install [Claude Desktop](https://claude.ai/download) (free — no subscription needed for local MCP servers).
2. Find your config file via **Settings → Developer → Edit Config**. On Windows this is often *not* `%APPDATA%\Claude\` but a virtualized MSIX path like:
   ```
   C:\Users\<you>\AppData\Local\Packages\Claude_<hash>\LocalCache\Roaming\Claude\claude_desktop_config.json
   ```
3. Add (merge, don't overwrite):
   ```json
   {
     "mcpServers": {
       "macropilot": {
         "command": "<absolute path to venv>\\Scripts\\python.exe",
         "args": ["<absolute path to>\\mcp_server\\server.py"]
       }
     }
   }
   ```
4. Fully quit and reopen Claude Desktop. Check **Settings → Developer** for a connected `macropilot` server with 3 tools.

**Gotchas worth knowing if you hit "server disconnected":**
- Claude Desktop launches the server with its own working directory, not your project root — if you rely on relative imports or `.env` autodiscovery, they can silently fail. Fix: resolve paths from `Path(__file__).resolve()`, and point `load_dotenv()` at an explicit path.
- The client has a ~60s startup timeout. If your server does slow first-time work at import (e.g. downloading an embedding model from HuggingFace), it can miss that window. Fix: pin `HF_HUB_OFFLINE=1` and an explicit `HF_HOME` once the model is cached locally, and avoid re-instantiating the same expensive model multiple times across your module graph.

---

## Sample outputs

**Query**: *"Has the Fed's tone on inflation shifted more hawkish in the last two meetings?"*

> Our analysis indicates that the Federal Reserve's tone on inflation has shifted in a more hawkish direction over the last two meetings, with a score of 0.4 on our hawkish-dovish spectrum. This shift is primarily driven by the emphasis on upside inflation risks and the potential need for tighter monetary policy, as highlighted in recent speeches and meeting minutes...
>
> Given the lack of market regime context, we cannot assess whether markets have already priced in this tone shift.
>
> **Bottom line**: The Fed's tone has shifted more hawkish on inflation, with a score of 0.4, driven by concerns over upside inflation risks and potential monetary policy tightening.

*Sources: FOMC Minutes (2026-06-17), speeches from 2026-07-06, 07-13, 07-15*

**Query**: *"Has the market already priced in a hawkish shift given the current regime?"* (triggers the QuantRisk cross-reference)

> Our tone analysis indicates a hawkish shift, with a score of 0.6... In the context of the current market regime, which has been characterized as "risk_off" since 2026-07-20 with a probability of 0.78, the question arises as to whether this hawkish tone has already been priced in by markets... it is challenging to determine the extent to which this shift has been fully priced in.
>
> **Bottom line**: The market may have partially priced in the hawkish shift, but the current tone suggests potential for further adjustments within the existing risk-off regime.

This second example is the actual differentiator: the note doesn't just report Fed tone, it explicitly reasons about whether that tone is already reflected in market positioning — the comparison a real macro/risk desk would want.

---

## Known limitations

- **Supervisor routing is keyword-based**, not an LLM classifier — fast and auditable, but can miss queries phrased in ways the patterns don't anticipate.
- **Regime data freshness depends on manual QuantRisk export** — there's no live pipeline between the two projects, by design (avoids a fragile cross-import between a notebook and a production-ish service), but it means the regime read is only as current as the last manual export.
- **Scoring is single-pass**, not adversarially checked — a second "critic" pass could catch cases where the LLM overweights a single strongly-worded excerpt.

## What's next

- Feed dissent counts and SEP (Summary of Economic Projections) deltas into the scoring agent alongside pure text tone — a real analyst reads voting patterns as a distinct signal from statement language, and pure NLP tone scoring can understate a hawkish shift that shows up mainly in dissents rather than prose.
- Automate the QuantRisk export (scheduled job) instead of manual copy.
- Add a lightweight frontend for non-technical demo access (currently cut per the project's fallback-priority plan, in favor of the QuantRisk integration and MCP/Claude Desktop connection).
