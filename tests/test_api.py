import pytest
import redis
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

import api.main as main


def ask(api, question, **kw):
    r = api.post("/query", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# --- jurisdiction ---------------------------------------------------------

@pytest.mark.parametrize("state", ["TX", "CA", "FL", "tx"])
def test_state_scoped_answer_never_cites_another_state(api, state):
    d = ask(api, "how do health insurance consumers file a complaint", jurisdiction=state)
    assert d["answer_grounded"]
    assert {s["jurisdiction"] for s in d["sources"]} <= {"federal", state.upper()}


def test_wrong_state_content_excluded_even_when_it_matches_best(api):
    # The question names California, but the asker is in Texas.
    d = ask(api, "California independent medical review complaint", jurisdiction="TX")
    assert all(s["jurisdiction"] != "CA" for s in d["sources"])


def test_no_jurisdiction_means_federal_only(api):
    d = ask(api, "health insurance complaint Department of Insurance")
    assert all(s["jurisdiction"] == "federal" for s in d["sources"])


def test_audience_isolation(api):
    d = ask(api, "COPD exacerbation management bronchodilators", audience="patient")
    assert all(s["audience"] == "patient" for s in d["sources"])


# --- citations and gating ---------------------------------------------------

def test_grounded_answer_carries_full_citations(api):
    d = ask(api, "does Medicare Part D cover insulin")
    assert d["answer_grounded"] and d["sources"]
    for s in d["sources"]:
        for field in ("url", "title", "jurisdiction", "effective_date", "text"):
            assert s[field], f"missing {field}"
        assert 0.0 <= s["score"] <= 1.0
    assert "insulin" in d["sources"][0]["text"]


def test_sources_sorted_by_reranker_score(api):
    d = ask(api, "does Medicare Part D cover insulin")
    scores = [s["score"] for s in d["sources"]]
    assert scores == sorted(scores, reverse=True)


def test_top_k_respected(api):
    d = ask(api, "Medicare covers care", top_k=1)
    assert len(d["sources"]) <= 1


def test_unrelated_question_refused_by_vector_floor(api):
    d = ask(api, "best chocolate chip cookie recipe")
    assert d["answer_grounded"] is False
    assert d["sources"] == []
    assert d["message"]
    assert d["top_cross_encoder_score"] is None  # never reached stage 2


def test_topical_but_wrong_question_refused_by_second_gate(api, monkeypatch):
    monkeypatch.setattr(main, "CROSS_ENCODER_THRESHOLD", 0.99)
    d = ask(api, "does Medicare cover insulin pumps for dogs")
    assert d["answer_grounded"] is False
    assert d["top_cross_encoder_score"] is not None


def test_empty_collection_refuses_instead_of_erroring(api, monkeypatch):
    from qdrant_client import models
    from settings import COLLECTION_NAME

    empty = QdrantClient(":memory:")
    empty.create_collection(COLLECTION_NAME, vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE))
    monkeypatch.setattr(main, "_client", empty)
    d = ask(api, "does Medicare cover insulin")
    assert d["answer_grounded"] is False
    assert d["top_vector_score"] is None


def test_request_validation(api):
    assert api.post("/query", json={"question": ""}).status_code == 422
    assert api.post("/query", json={"question": "x", "audience": "admin"}).status_code == 422
    assert api.post("/query", json={"question": "x", "top_k": 50}).status_code == 422


# --- caching -------------------------------------------------------------

def test_second_identical_query_is_cache_hit(api):
    first = ask(api, "does Medicare Part D cover insulin")
    second = ask(api, "does Medicare Part D cover insulin")
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert first["sources"] == second["sources"]


def test_generation_bump_forces_recompute(api, cache):
    ask(api, "health insurance complaint", jurisdiction="CA")
    cache.bump_generation("CA")
    assert ask(api, "health insurance complaint", jurisdiction="CA")["cache_hit"] is False


def test_refusals_are_not_cached(api):
    ask(api, "best chocolate chip cookie recipe")
    assert ask(api, "best chocolate chip cookie recipe")["cache_hit"] is False


# --- dependency failures --------------------------------------------------

class _BrokenRedis:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise redis.exceptions.ConnectionError("redis down")
        return fail


def test_redis_down_still_answers(api, cache):
    cache.client = _BrokenRedis()
    d = ask(api, "does Medicare Part D cover insulin")
    assert d["answer_grounded"] and d["cache_hit"] is False


class _BrokenQdrant:
    def query_points(self, *a, **k):
        raise ResponseHandlingException(ConnectionRefusedError("qdrant down"))


def test_qdrant_down_returns_503(api, monkeypatch):
    monkeypatch.setattr(main, "_client", _BrokenQdrant())
    r = api.post("/query", json={"question": "does Medicare cover insulin"})
    assert r.status_code == 503


def test_health(api):
    assert api.get("/health").json()["status"] == "ok"
