import query_cache
from query_cache import SCHEMA_VERSION


def test_key_embeds_schema_version_and_generations(cache):
    key = cache.make_key("does Medicare cover insulin", "patient", ["federal", "CA"])
    assert key.startswith(f"query:{SCHEMA_VERSION}:patient:CA-federal:0-0:")


def test_key_ignores_case_and_surrounding_whitespace(cache):
    a = cache.make_key("Does Medicare cover insulin", "patient", ["federal"])
    b = cache.make_key("  does medicare cover INSULIN ", "patient", ["federal"])
    assert a == b


def test_key_depends_on_audience_and_jurisdiction(cache):
    q = "what are my rights"
    keys = {
        cache.make_key(q, "patient", ["federal"]),
        cache.make_key(q, "clinician", ["federal"]),
        cache.make_key(q, "patient", ["federal", "CA"]),
        cache.make_key(q, "patient", ["federal", "TX"]),
    }
    assert len(keys) == 4


def test_state_bump_invalidates_only_that_state(cache):
    q = "complaint process"
    fed, ca, tx = (cache.make_key(q, "patient", j) for j in (["federal"], ["federal", "CA"], ["federal", "TX"]))
    cache.bump_generation("CA")
    assert cache.make_key(q, "patient", ["federal"]) == fed
    assert cache.make_key(q, "patient", ["federal", "CA"]) != ca
    assert cache.make_key(q, "patient", ["federal", "TX"]) == tx


def test_federal_bump_invalidates_every_state_scoped_key(cache):
    q = "complaint process"
    before = [cache.make_key(q, "patient", j) for j in (["federal"], ["federal", "CA"], ["federal", "TX"])]
    cache.bump_generation("federal")
    after = [cache.make_key(q, "patient", j) for j in (["federal"], ["federal", "CA"], ["federal", "TX"])]
    assert all(a != b for a, b in zip(before, after))


def test_stale_entry_is_unreachable_after_bump(cache):
    key = cache.make_key("q", "patient", ["federal", "CA"])
    cache.set(key, {"answer_grounded": True, "sources": []})
    cache.bump_generation("CA")
    assert cache.get(cache.make_key("q", "patient", ["federal", "CA"])) is None
    assert cache.get(key) is not None  # still stored, just never looked up again


def test_schema_version_bump_orphans_old_entries(cache, monkeypatch):
    old_key = cache.make_key("q", "patient", ["federal"])
    cache.set(old_key, {"answer_grounded": True, "sources": []})
    monkeypatch.setattr(query_cache, "SCHEMA_VERSION", "v-next")
    new_key = cache.make_key("q", "patient", ["federal"])
    assert new_key != old_key
    assert cache.get(new_key) is None


def test_entries_expire(cache):
    key = cache.make_key("q", "patient", ["federal"])
    cache.set(key, {"x": 1}, ttl_seconds=120)
    assert 0 < cache.client.ttl(key) <= 120


def test_bump_generation_is_monotonic(cache):
    assert [cache.bump_generation("TX") for _ in range(3)] == [1, 2, 3]
