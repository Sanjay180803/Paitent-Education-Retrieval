"""
Embeds chunks.jsonl using FastEmbed (local ONNX embeddings, no GPU or API
key needed) and indexes them into Qdrant, with jurisdiction/audience/
source_type stored as payload fields so retrieval can filter DURING the
ANN search itself -- not post-filter after retrieval, which would leak
information and silently return fewer than k real results.

Two ways to run Qdrant:

  A) Local, on-disk, no server needed (default here) -- fine for
     development and for this portfolio project.

  B) A real Qdrant server via Docker, closer to how you'd actually deploy
     this (and matches the containerized-services pattern in your other
     projects):
       docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
     then set USE_SERVER = True below.

First run will download the embedding model (~67MB) from HuggingFace and
cache it locally -- only happens once.

Run:
  pip install fastembed qdrant-client
  python embed_and_index.py
"""

import json
import shutil
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path.cwd()
CHUNKS_FILE = PROJECT_ROOT / "data_corpus" / "chunks.jsonl"
QDRANT_STORAGE = PROJECT_ROOT / "qdrant_storage"

COLLECTION_NAME = "patient_education_policy"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ~67MB, good general-purpose model
EMBED_BATCH_SIZE = 32

USE_SERVER = False
SERVER_URL = "http://localhost:6333"


def get_client() -> QdrantClient:
    if USE_SERVER:
        return QdrantClient(url=SERVER_URL)
    return QdrantClient(path=str(QDRANT_STORAGE))


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found -- run chunk_corpus.py first")
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads once, then cached)...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    dim = embedder.embedding_size

    # Qdrant's local (on-disk) mode can leave stale segment files behind
    # even after delete_collection() + create_collection() with the same
    # name -- deleting the storage directory outright guarantees a truly
    # clean rebuild instead of silently mixing old and new points.
    if not USE_SERVER and QDRANT_STORAGE.exists():
        shutil.rmtree(QDRANT_STORAGE)

    client = get_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Index the payload fields you'll filter on -- this makes filtered
    # queries fast instead of a full scan for every jurisdiction/audience check.
    for field in ("jurisdiction", "audience", "source_type"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    print(f"Embedding {len(chunks)} chunks (batch size {EMBED_BATCH_SIZE})...")
    texts = [c["text"] for c in chunks]
    vectors = list(embedder.embed(texts, batch_size=EMBED_BATCH_SIZE))

    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "source_type": c["source_type"],
                "url": c["url"],
                "title": c["title"],
                "audience": c["audience"],
                "jurisdiction": c["jurisdiction"],
                "effective_date": c["effective_date"],
                "text": c["text"],  # kept in payload so retrieval doesn't need a second lookup
            },
        )
        for i, c in enumerate(chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    count = client.count(COLLECTION_NAME).count
    print(f"Done. Collection '{COLLECTION_NAME}' now has {count} points.")

    # --- Smoke test -------------------------------------------------------
    print("\n--- Smoke test: unfiltered query ---")
    query_vec = list(embedder.embed(["does Medicare cover insulin"]))[0].tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=query_vec, limit=3).points
    for r in results:
        print(f"  score={r.score:.3f} | {r.payload['source_type']} | {r.payload['title']}")

    print("\n--- Smoke test: filtered query (jurisdiction=federal only) ---")
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="jurisdiction", match=models.MatchValue(value="federal"))]
        ),
        limit=3,
    ).points
    for r in results:
        print(f"  score={r.score:.3f} | {r.payload['jurisdiction']} | {r.payload['title']}")


if __name__ == "__main__":
    main()
