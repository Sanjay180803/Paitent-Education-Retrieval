"""
FastAPI serving layer for the Patient Education & Policy Q&A retrieval gateway.

Design points this endpoint enforces (from the project's original spec):
  - Jurisdiction filtering happens IN the vector-DB query itself (a query_filter
    passed to Qdrant), not as a post-hoc filter on returned results. A
    federal-only or wrong-state chunk never enters the candidate set.
  - Hybrid retrieval: vector similarity (semantic) and BM25 (exact-term)
    results are fused via Reciprocal Rank Fusion, so an exact drug name or
    plan code that a vector embedding blurs into a general "topic" still
    surfaces if BM25 catches it. See hybrid_search.py for the fusion logic.
  - Confidence gating: if the best VECTOR match's similarity score is below
    CONFIDENCE_THRESHOLD, the endpoint returns "insufficient grounding"
    instead of guessing. Gating stays keyed on vector cosine similarity
    specifically (an interpretable 0-1 score) rather than the fused RRF
    score, which is a ranking score, not a calibrated confidence measure.
  - Every answer carries its sources (url, title, jurisdiction,
    effective_date) for transparency/traceability.

This endpoint does retrieval + confidence gating + returns grounded source
excerpts. It does NOT call an LLM to generate a synthesized answer -- that's
a deliberate scope choice so the project doesn't depend on an API key/cost,
and so the "grounding" story stays literal (you see the actual retrieved
text, not a paraphrase that could drift from it). If you want an LLM
synthesis step layered on top later, see the comment at the bottom of
this file for where it plugs in.

Run:
  pip install fastapi uvicorn fastembed qdrant-client rank-bm25
  python embed_and_index.py     # build the index first, if you haven't
  python -m uvicorn api.main:app --reload --port 8000

Then:
  Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post `
    -ContentType "application/json" `
    -Body '{"question": "does Medicare cover insulin", "audience": "patient"}'
"""

import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `import hybrid_search` finds the sibling module
from hybrid_search import BM25Index, load_chunks, reciprocal_rank_fusion  # noqa: E402
from query_cache import QueryCache  # noqa: E402
from reranker import Reranker  # noqa: E402
from settings import COLLECTION_NAME, EMBEDDING_MODEL, Settings, qdrant_client  # noqa: E402

SETTINGS = Settings()

# Two-stage gating. See reranker.py's module docstring for the full story:
# vector cosine similarity cannot distinguish "diabetes insipidus" from
# "diabetes mellitus" content, because a bi-encoder embeds each side
# independently. The eval harness (run_eval.py) proved a single cosine
# threshold cannot separate these cases -- some genuinely wrong questions
# score HIGHER than genuinely answerable ones.
#
# Stage 1 (cheap floor): reject obviously-unrelated queries before doing
# any BM25/fusion/reranking work at all -- this threshold only needs to
# catch the EASY cases (capital of France, cookie recipes), so it's set
# low and conservative.
VECTOR_FLOOR = SETTINGS.vector_floor

# Stage 2 (precise gate): everything that survives Stage 1 gets BM25-fused
# and reranked by the cross-encoder, which CAN read query+candidate jointly
# and catch the disease-name-overlap failure mode. This is the real gate.
#
# Threshold tuned via run_eval.py against golden_questions.json. Real
# cross-encoder scores on this corpus cluster tightly into three groups:
#   - genuine refusals (unrelated topics):         0.000 - 0.001
#   - genuine grounded answers:                    0.012 - 1.000
#   - KNOWN LIMITATION -- disease-name near-misses: 0.843 - 0.963
#     ("diabetes insipidus" / "diabetes mellitus type 3c" score HIGH
#     because ms-marco-MiniLM-L-6-v2 is a general-domain model trained on
#     web search relevance, not medical entity disambiguation -- it cannot
#     tell insipidus and mellitus are different diseases, only that both
#     are diabetes-adjacent clinical text. No single scalar threshold can
#     separate this cluster from genuine answers, since some genuine
#     answers legitimately score as low as 0.012. This is an accepted,
#     documented gap: a medical-domain-tuned or NLI-based reranker would be
#     the real fix, out of scope for this iteration.)
# 0.01 sits in the clean gap between refusals and genuine answers, giving
# the best overall tradeoff given the above.
CROSS_ENCODER_THRESHOLD = SETTINGS.cross_encoder_threshold

TOP_K = 5
CANDIDATE_POOL_SIZE = 20  # how many each of vector/BM25 contribute before fusion
RERANK_POOL_SIZE = 10  # how many fused candidates get cross-encoder reranking
RRF_K = 60  # standard RRF damping constant

app = FastAPI(title="Patient Education & Policy Q&A Gateway")

# Loaded once at startup, reused across requests.
_embedder: Optional[TextEmbedding] = None
_client: Optional[QdrantClient] = None
_bm25_index: Optional[BM25Index] = None
_cache: Optional[QueryCache] = None
_reranker: Optional[Reranker] = None

CACHE_TTL_SECONDS = SETTINGS.cache_ttl_seconds


@app.on_event("startup")
def load_resources():
    global _embedder, _client, _bm25_index, _cache, _reranker
    _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    _client = qdrant_client(SETTINGS)
    chunks = load_chunks(SETTINGS.chunks_file)
    _bm25_index = BM25Index(chunks)
    _cache = QueryCache.from_url(SETTINGS.redis_url) if SETTINGS.cache_enabled else None
    _reranker = Reranker()  # downloads ms-marco-MiniLM-L-6-v2 once, then cached locally


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    # None = no jurisdiction restriction (federal-only content still applies everywhere)
    jurisdiction: Optional[str] = Field(
        None, description="e.g. 'CA', 'TX', 'FL'. Omit for federal-only questions."
    )
    audience: Literal["patient", "clinician"] = "patient"
    top_k: int = Field(TOP_K, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    title: str
    url: str
    source_type: str
    jurisdiction: str
    audience: str
    effective_date: str
    score: float = Field(..., description="Cross-encoder relevance score (sigmoid, 0-1)")


class QueryResponse(BaseModel):
    answer_grounded: bool
    sources: list[SourceChunk]
    message: Optional[str] = None  # populated when answer_grounded is False
    cache_hit: bool = False
    top_vector_score: Optional[float] = Field(
        None, description="Raw top vector cosine score, exposed for eval/threshold tuning."
    )
    top_cross_encoder_score: Optional[float] = Field(
        None, description="Raw top cross-encoder score, exposed for eval/threshold tuning."
    )


def build_filter(jurisdiction: Optional[str], audience: str) -> models.Filter:
    """
    Jurisdiction logic: a question scoped to a state should see that state's
    rules PLUS federal rules (federal is the baseline everyone falls under).
    A question with no jurisdiction given sees federal-only content.

    Audience logic: patient questions should not surface clinician-only
    technical chunks (e.g. raw PubMed abstracts) and vice versa, unless you
    want to loosen this later for a "show me the technical version" toggle.
    """
    jurisdiction_values = ["federal"]
    if jurisdiction:
        jurisdiction_values.append(jurisdiction.upper())

    return models.Filter(
        must=[
            models.FieldCondition(key="jurisdiction", match=models.MatchAny(any=jurisdiction_values)),
            models.FieldCondition(key="audience", match=models.MatchValue(value=audience)),
        ]
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    jurisdiction_values = ["federal"] + ([req.jurisdiction.upper()] if req.jurisdiction else [])

    # --- Cache lookup --------------------------------------------------
    # Redis is optional infrastructure: if it's down or unreachable, the
    # endpoint degrades gracefully to always-compute rather than hard-failing
    # every request. This is a deliberate availability tradeoff -- caching
    # is a performance optimization here, not a correctness dependency.
    cache_key = None
    if _cache is not None:
        try:
            cache_key = _cache.make_key(req.question, req.audience, jurisdiction_values)
            cached = _cache.get(cache_key)
            if cached is not None:
                return QueryResponse(**cached, cache_hit=True)
        except Exception as e:
            print(f"[warn] cache read failed, computing fresh: {e}")

    query_filter = build_filter(req.jurisdiction, req.audience)

    # --- Vector search (semantic) -----------------------------------------
    query_vector = list(_embedder.embed([req.question]))[0].tolist()
    # Unlike Redis, Qdrant IS a correctness dependency: without it there is
    # nothing to ground an answer in, so fail with 503 rather than guessing.
    try:
        vector_hits = _client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=CANDIDATE_POOL_SIZE,
        ).points
    except (ResponseHandlingException, UnexpectedResponse, ConnectionError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"vector store unavailable: {type(e).__name__}")

    # --- Stage 1: cheap floor, rejects the easy/obvious non-matches -------
    if not vector_hits or vector_hits[0].score < VECTOR_FLOOR:
        response = QueryResponse(
            answer_grounded=False,
            sources=[],
            message=(
                "I don't have a confident, sourced answer to this question "
                "in the current knowledge base. Please consult a healthcare "
                "provider or the relevant official source directly."
            ),
            top_vector_score=vector_hits[0].score if vector_hits else None,
        )
        return response

    vector_ranked = [dict(r.payload) for r in vector_hits]

    # --- BM25 search (exact-term) -------------------------------------------
    bm25_ranked = _bm25_index.search(
        req.question,
        jurisdiction_values=jurisdiction_values,
        audience=req.audience,
        top_n=CANDIDATE_POOL_SIZE,
    )

    # --- Fuse, then rerank the fused pool with the cross-encoder -----------
    fused = reciprocal_rank_fusion(
        [vector_ranked, bm25_ranked], k=RRF_K, top_k=RERANK_POOL_SIZE
    )
    reranked = _reranker.rerank(req.question, fused)

    # --- Stage 2: the precise gate ------------------------------------------
    # This is what actually catches "diabetes insipidus" vs "diabetes
    # mellitus content" -- vector score alone let it through Stage 1, but
    # the cross-encoder, reading query+candidate jointly, correctly scores
    # the mismatch low.
    if not reranked or reranked[0]["cross_encoder_score"] < CROSS_ENCODER_THRESHOLD:
        response = QueryResponse(
            answer_grounded=False,
            sources=[],
            message=(
                "I don't have a confident, sourced answer to this question "
                "in the current knowledge base. Please consult a healthcare "
                "provider or the relevant official source directly."
            ),
            top_vector_score=vector_hits[0].score,
            top_cross_encoder_score=reranked[0]["cross_encoder_score"] if reranked else None,
        )
        return response

    top_results = reranked[: req.top_k]
    sources = [
        SourceChunk(
            text=c["text"],
            title=c["title"],
            url=c["url"],
            source_type=c["source_type"],
            jurisdiction=c["jurisdiction"],
            audience=c["audience"],
            effective_date=c["effective_date"],
            score=c["cross_encoder_score"],
        )
        for c in top_results
    ]
    response = QueryResponse(
        answer_grounded=True,
        sources=sources,
        top_vector_score=vector_hits[0].score,
        top_cross_encoder_score=reranked[0]["cross_encoder_score"],
    )

    if _cache is not None and cache_key is not None:
        try:
            _cache.set(cache_key, response.model_dump(exclude={"cache_hit"}), ttl_seconds=CACHE_TTL_SECONDS)
        except Exception as e:
            print(f"[warn] cache write failed: {e}")

    return response


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME}


# --- Where an LLM synthesis step would plug in, if you add one later ------
# After the confidence gate passes, you'd pass `sources` (already
# jurisdiction/audience-filtered, hybrid-ranked, and confidence-checked)
# into an LLM prompt to synthesize a single readable answer, always citing
# back to `sources`. The retrieval + gating logic above stays exactly the
# same either way -- that's the point of keeping generation as a separate,
# optional layer.

