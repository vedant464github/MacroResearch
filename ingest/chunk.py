"""
Chunking strategy for Fed documents.

Why not naive fixed-size chunking: FOMC statements are short (~800-1200 words)
and dense - each paragraph is a distinct policy signal (economic assessment,
rate decision, forward guidance, dissents). Splitting mid-paragraph loses the
unit of meaning the scoring agent needs. Minutes and speeches are long and
benefit from paragraph-aware chunking with overlap so we don't cut a hawkish/
dovish qualifier off from the sentence that gives it context.

Strategy:
  - Split on paragraph boundaries first.
  - Merge small paragraphs up to a target token budget (~350 tokens).
  - Overlap 1 paragraph between chunks so scoring context isn't lost at edges.
  - Keep doc-level metadata (date, type, speaker, url) on every chunk -
    the regime-context agent and citations both need this later.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOKENS = 350          # rough budget per chunk, using word-count as a cheap proxy
OVERLAP_PARAGRAPHS = 1


@dataclass
class Chunk:
    chunk_id: str
    doc_type: str
    date: str
    title: str
    speaker: str | None
    url: str
    text: str
    chunk_index: int

def _slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    return s[:60]

def _word_count(s: str) -> int:
    return len(s.split())


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # drop boilerplate lines the Fed sticks on every page (nav crumbs, "Last Update" footer)
    paras = [p for p in paras if not re.match(r"^(Last Update|Home|Skip to)", p)]
    return paras


def chunk_document(doc: dict) -> list[Chunk]:
    paras = _split_paragraphs(doc["text"])
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0  
    idx = 0

    def flush():
        nonlocal buf, buf_tokens, idx
        if not buf:
            return
        chunks.append(Chunk(
            chunk_id=f"{doc['doc_type']}_{doc['date']}_{_slug(doc['title'])}_{idx}",
            doc_type=doc["doc_type"],
            date=doc["date"],
            title=doc["title"],
            speaker=doc.get("speaker"),
            url=doc["url"],
            text="\n\n".join(buf),
            chunk_index=idx,
        ))
        idx += 1

    for para in paras:
        pt = _word_count(para)
        if buf and buf_tokens + pt > TARGET_TOKENS:
            flush()
            # carry overlap forward
            buf = buf[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS else []
            buf_tokens = sum(_word_count(p) for p in buf)
        buf.append(para)
        buf_tokens += pt
    flush()
    return chunks


def chunk_all() -> Path:
    out_path = CHUNKS_DIR / "chunks.jsonl"
    n_docs, n_chunks = 0, 0
    with out_path.open("w", encoding="utf-8") as out:
        for f in sorted(RAW_DIR.glob("*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            for c in chunk_document(doc):
                out.write(json.dumps(asdict(c)) + "\n")
                n_chunks += 1
            n_docs += 1
    print(f"[ok] chunked {n_docs} docs -> {n_chunks} chunks -> {out_path}")
    return out_path


if __name__ == "__main__":
    chunk_all()
