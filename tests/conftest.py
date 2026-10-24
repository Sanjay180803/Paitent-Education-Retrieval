"""
Shared fixtures. Unit and API tests run without downloading any model:
the embedder and cross-encoder are replaced with small deterministic
stand-ins, while Qdrant (in-memory), BM25, RRF, the cache (fakeredis) and
the FastAPI endpoint are the real code.
"""
import hashlib
import math
import os
import re
import sys

import fakeredis
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qdrant_client import QdrantClient, models  # noqa: E402

from hybrid_search import BM25Index  # noqa: E402
from query_cache import QueryCache  # noqa: E402
from settings import COLLECTION_NAME  # noqa: E402

DIM = 256
WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Texts sharing
    words get high cosine similarity; unrelated texts get ~0."""

    def embed(self, texts, batch_size=None):
        for text in texts:
            v = np.zeros(DIM)
            for w in WORD.findall(text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = np.linalg.norm(v)
            yield v / n if n else v


class OverlapReranker:
    """Scores (question, chunk) by the fraction of question words found in
    the chunk, squashed to 0-1. Stands in for the cross-encoder."""

    def rerank(self, question, candidates, text_key="text"):
        q = set(WORD.findall(question.lower()))
        out = []
        for c in candidates:
            words = set(WORD.findall(c[text_key].lower()))
            frac = len(q & words) / max(len(q), 1)
            out.append({**c, "cross_encoder_score": 1 / (1 + math.exp(-12 * (frac - 0.5)))})
        out.sort(key=lambda c: c["cross_encoder_score"], reverse=True)
        return out


def _chunk(cid, text, jurisdiction="federal", audience="patient", title=None):
    return {
        "chunk_id": cid,
        "doc_id": cid.split("_c")[0],
        "text": text,
        "title": title or cid,
        "url": f"https://example.gov/{cid}",
        "source_type": "test",
        "jurisdiction": jurisdiction,
        "audience": audience,
        "effective_date": "2026-01-01",
    }


CHUNKS = [
    _chunk("medicare_c000", "Medicare Part B covers doctor visits outpatient care and preventive services", title="Medicare Part B"),
    _chunk("insulin_c000", "Medicare Part D covers insulin and prescription drugs for diabetes"),
    _chunk("ca_c000", "California health insurance consumers can file an independent medical review complaint", "CA"),
    _chunk("tx_c000", "Texas health insurance consumers can file a complaint with the Texas Department of Insurance", "TX"),
    _chunk("fl_c000", "Florida health insurance consumers can contact the Division of Consumer Services", "FL"),
    _chunk("copd_c000", "COPD chronic obstructive pulmonary disease makes it hard to breathe"),
    _chunk("copd_clin_c000", "COPD exacerbation management guideline bronchodilators corticosteroids", audience="clinician"),
]


@pytest.fixture
def chunks():
    return [dict(c) for c in CHUNKS]


@pytest.fixture
def qdrant(chunks):
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION_NAME, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    )
    vecs = list(HashingEmbedder().embed([c["text"] for c in chunks]))
    client.upsert(
        COLLECTION_NAME,
        points=[models.PointStruct(id=i, vector=vecs[i].tolist(), payload=c) for i, c in enumerate(chunks)],
    )
    return client


@pytest.fixture
def cache():
    return QueryCache(client=fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture
def api(monkeypatch, qdrant, chunks, cache):
    """The real FastAPI app with model-backed globals swapped for stand-ins.
    Startup hooks are not run (TestClient is not used as a context manager)."""
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setattr(main, "_embedder", HashingEmbedder())
    monkeypatch.setattr(main, "_client", qdrant)
    monkeypatch.setattr(main, "_bm25_index", BM25Index(chunks))
    monkeypatch.setattr(main, "_cache", cache)
    monkeypatch.setattr(main, "_reranker", OverlapReranker())
    return TestClient(main.app, raise_server_exceptions=False)
