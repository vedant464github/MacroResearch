"""
Embeds chunks with a local sentence-transformers model (CPU, no API key)
and loads them into a persistent Chroma collection.

Model choice: BAAI/bge-small-en-v1.5 (384-dim, ~130MB) - good quality/speed
tradeoff for a 3-day build, runs fine on CPU. Swap to all-MiniLM-L6-v2 if
disk/RAM is tight - it's smaller and marginally lower quality.

Note on bge models: they expect a query-side instruction prefix for
asymmetric search (query != passage). We add it at query time in rag/chain.py,
NOT at ingest time - passages are embedded raw.
"""
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "fed_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index():
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks found at {CHUNKS_PATH} - run ingest/chunk.py first")

    print(f"[info] embedding {len(chunks)} chunks with {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    # fresh collection each full rebuild - simplest correctness story for a 3-day build
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "doc_type": c["doc_type"],
                "date": c["date"],
                "date_int": int(c["date"].replace("-", "")),
                "title": c["title"],
                "speaker": c["speaker"] or "",
                "url": c["url"],
            } for c in batch],
        )
        print(f"[ok] loaded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"[done] {collection.count()} chunks indexed in Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
