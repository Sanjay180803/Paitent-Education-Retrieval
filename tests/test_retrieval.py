from qdrant_client import models

from api.main import build_filter
from hybrid_search import BM25Index, reciprocal_rank_fusion, tokenize
from settings import COLLECTION_NAME
from conftest import HashingEmbedder


def _jurisdictions(results):
    return {r["jurisdiction"] for r in results}


def test_bm25_state_query_sees_only_that_state_and_federal(chunks):
    idx = BM25Index(chunks)
    res = idx.search("health insurance complaint", ["federal", "TX"], "patient", top_n=50)
    assert res
    assert _jurisdictions(res) <= {"federal", "TX"}


def test_bm25_respects_audience(chunks):
    idx = BM25Index(chunks)
    res = idx.search("COPD", ["federal"], "patient", top_n=50)
    assert res and all(r["audience"] == "patient" for r in res)


def test_bm25_filters_after_full_scoring(chunks):
    """Filtering must not under-return: every eligible chunk is still ranked."""
    idx = BM25Index(chunks)
    res = idx.search("zzz-no-match", ["federal"], "patient", top_n=50)
    eligible = [c for c in chunks if c["jurisdiction"] == "federal" and c["audience"] == "patient"]
    assert len(res) == len(eligible)


def test_build_filter_without_state_is_federal_only():
    f = build_filter(None, "patient")
    jur = next(c for c in f.must if c.key == "jurisdiction")
    assert jur.match.any == ["federal"]


def test_build_filter_upper_cases_state():
    f = build_filter("tx", "clinician")
    jur = next(c for c in f.must if c.key == "jurisdiction")
    aud = next(c for c in f.must if c.key == "audience")
    assert jur.match.any == ["federal", "TX"]
    assert aud.match.value == "clinician"


def test_qdrant_filter_excludes_other_states_inside_ann_search(qdrant):
    vec = next(HashingEmbedder().embed(["California health insurance complaint"])).tolist()
    hits = qdrant.query_points(
        COLLECTION_NAME, query=vec, query_filter=build_filter("TX", "patient"), limit=50
    ).points
    assert hits
    assert {h.payload["jurisdiction"] for h in hits} <= {"federal", "TX"}


def test_rrf_scores_mirror_rankings_equally():
    a, b = {"chunk_id": "a"}, {"chunk_id": "b"}
    fused = reciprocal_rank_fusion([[a, b], [b, a]], top_k=2)
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]


def test_rrf_rewards_agreement():
    a, b, c = {"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}
    fused = reciprocal_rank_fusion([[a, b, c], [a, c, b]], top_k=3)
    assert fused[0]["chunk_id"] == "a"


def test_rrf_truncates_to_top_k():
    lst = [{"chunk_id": str(i)} for i in range(10)]
    assert len(reciprocal_rank_fusion([lst], top_k=4)) == 4


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Part-B, COPD!") == ["part", "b", "copd"]
