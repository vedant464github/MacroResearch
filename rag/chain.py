"""
Day-1 goal: prove retrieval -> LLM answer works end to end, before any
LangGraph structure gets built on top of it on Day 2.

This is intentionally NOT the final agent architecture. It's a flat
retrieve-then-generate chain used to sanity check:
  1. Chroma retrieval quality (are the right chunks coming back?)
  2. Groq API connectivity + prompt shape
  3. That citations (date, title, url) survive the round trip

Day 2 replaces the "generate" step with the scoring/regime/synthesis agents,
but retrieval logic here gets reused as-is inside the LangGraph retrieval node.
"""
import os
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
import os
import re
import chromadb
from chromadb.config import Settings
from groq import Groq
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "fed_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.3-70b-versatile"

# bge models want this instruction prefix on the QUERY side only (asymmetric search)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are a macro research assistant. Answer the user's question about \
Federal Reserve monetary policy using ONLY the provided source excerpts. \
Every claim must cite the source date and document type in parentheses, e.g. (FOMC Statement, 2026-06-17). \
If the excerpts don't support an answer, say so explicitly instead of guessing."""

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

RECENCY_PATTERNS = [
    re.compile(r"\blast\s+(\d+)\s+meetings?\b", re.I),
    re.compile(r"\bpast\s+(\d+)\s+meetings?\b", re.I),
    re.compile(r"\blast\s+(" + "|".join(NUMBER_WORDS) + r")\s+meetings?\b", re.I),
    re.compile(r"\bpast\s+(" + "|".join(NUMBER_WORDS) + r")\s+meetings?\b", re.I),
]
GENERIC_RECENCY_WORDS = re.compile(r"\b(recent|recently|latest|current|now|these days)\b", re.I)


def detect_recency_filter(query: str, meeting_dates: list[str]) -> str | None:
    """
    meeting_dates: sorted descending list of distinct FOMC meeting dates (YYYY-MM-DD),
    from statements + minutes only (speeches happen off-cycle, don't count as 'meetings').

    Returns a date cutoff string (inclusive) to pass to Chroma's $gte filter, or None
    if the query has no recency signal - in which case retrieval stays unfiltered.
    """
    if not meeting_dates:
        return None

    for pattern in RECENCY_PATTERNS:
        m = pattern.search(query)
        if m:
            raw = m.group(1)
            n = NUMBER_WORDS.get(raw.lower(), None)
            if n is None:
                n = int(raw)
            n = min(n, len(meeting_dates))
            return meeting_dates[n - 1]

    if GENERIC_RECENCY_WORDS.search(query):
        n = min(2, len(meeting_dates))
        return meeting_dates[n - 1]

    return None

class RAGChain:
    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._meeting_dates_cache: list[str] | None = None

    def _meeting_dates(self) -> list[str]:
        """Distinct dates from statements + minutes (actual FOMC meetings, not speeches),
        sorted most-recent-first. Cached per instance since it barely changes within a run."""
        if self._meeting_dates_cache is not None:
            return self._meeting_dates_cache
        res = self.collection.get(where={"doc_type": {"$in": ["statement", "minutes"]}}, include=["metadatas"])
        dates = sorted({m["date"] for m in res["metadatas"]}, reverse=True)
        self._meeting_dates_cache = dates
        return dates

    def retrieve(self, query: str, k: int = 6, date_cutoff: str | None = None) -> list[dict]:
        q_emb = self.embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
        where = {"date_int": {"$gte": int(date_cutoff.replace("-", ""))}} if date_cutoff else None
        res = self.collection.query(query_embeddings=q_emb, n_results=k, where=where)
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({**meta, "text": doc, "score": 1 - dist})
        return out

    def answer(self, query: str, k: int = 6) -> dict:
        date_cutoff = detect_recency_filter(query, self._meeting_dates())
        hits = self.retrieve(query, k=k, date_cutoff=date_cutoff)

        if not hits and date_cutoff:
            # filter was too aggressive (e.g. sparse data near the cutoff) - fall back to unfiltered
            hits = self.retrieve(query, k=k)

        # dedupe by (doc_type, date, title) for the source list, keep all chunks in context
        context = "\n\n---\n\n".join(
            f"[{h['doc_type'].upper()} | {h['date']} | {h['title']}]\n{h['text']}" for h in hits
        )
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nSource excerpts:\n\n{context}"},
            ],
            temperature=0.1,
        )
        seen = set()
        sources = []
        for h in hits:
            key = (h["doc_type"], h["date"], h["title"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({"date": h["date"], "doc_type": h["doc_type"], "title": h["title"], "url": h["url"]})

        return {
            "query": query,
            "answer": resp.choices[0].message.content,
            "sources": sources,
            "date_cutoff_applied": date_cutoff,
        }


if __name__ == "__main__":
    chain = RAGChain()
    result = chain.answer("Has the Fed's tone on inflation shifted more hawkish in the last two meetings?")
    print("date_cutoff_applied:", result["date_cutoff_applied"])
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['doc_type']} {s['date']}: {s['title']}")
