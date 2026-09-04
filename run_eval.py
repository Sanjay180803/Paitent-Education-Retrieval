"""
Evaluation harness for the retrieval gateway.

Metrics computed (not precision@k in the strict sense, since most golden
questions here have exactly one "correct" topic rather than several equally-
relevant documents -- hit@k and MRR are the right fit for that shape):

  - hit@k: did ANY of the top-k returned sources match the expected topic?
    (1 or 0 per question, averaged across the golden set)
  - MRR (mean reciprocal rank): 1/rank of the first matching source, or 0
    if none matched. Rewards getting the right answer near the TOP of the
    results, not just somewhere in the top-k.
  - jurisdiction_violation_rate: fraction of questions where a returned
    source's jurisdiction fell outside {federal, requested_state}. This
    should always be 0 -- filtering happens at the query layer, so a
    nonzero rate here is a genuine regression, not a ranking-quality issue.
  - audience_violation_rate: same idea, for the audience field.
  - refusal_accuracy: for questions marked expect_grounded=false (out-of-
    scope queries), fraction correctly refused rather than answered.
  - CONFIDENCE_THRESHOLD tuning data: prints the range of top_vector_score
    for questions that SHOULD ground vs questions that SHOULD refuse, so
    you can pick a threshold that actually separates them, instead of the
    placeholder 0.5 guess in api/main.py.

Ground truth matching: each golden question lists expect_keywords_any --
matched case-insensitively against a source's title + url + text combined.
This is deliberately loose (keyword presence, not exact chunk ID) because
chunk boundaries can shift as chunk_corpus.py's parameters change, and
because some source titles/URLs (e.g. MedlinePlus slugs) weren't verifiable
ahead of time -- see the comment in golden_questions.json if you need to
adjust keywords after inspecting your actual corpus.

Run (with the API server already running):
  python run_eval.py
  python run_eval.py --url http://localhost:8000 --golden-set golden_questions.json
"""

import argparse
import json
from pathlib import Path

import requests

DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_questions.json"
DEFAULT_API_URL = "http://localhost:8000"


def load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def matches_keywords(source: dict, keywords: list[str]) -> bool:
    if not keywords:
        return False
    haystack = f"{source.get('title', '')} {source.get('url', '')} {source.get('text', '')}".lower()
    return any(kw.lower() in haystack for kw in keywords)


def evaluate_question(api_url: str, item: dict) -> dict:
    payload = {
        "question": item["question"],
        "audience": item["audience"],
        "top_k": 5,
    }
    if item.get("jurisdiction"):
        payload["jurisdiction"] = item["jurisdiction"]

    resp = requests.post(f"{api_url}/query", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = {
        "question": item["question"],
        "expect_grounded": item["expect_grounded"],
        "actual_grounded": data["answer_grounded"],
        "top_vector_score": data.get("top_vector_score"),
        "top_cross_encoder_score": data.get("top_cross_encoder_score"),
        "hit": 0,
        "reciprocal_rank": 0.0,
        "jurisdiction_violation": False,
        "audience_violation": False,
        "correct_refusal_behavior": None,
    }

    allowed_jurisdictions = {"federal"}
    if item.get("jurisdiction"):
        allowed_jurisdictions.add(item["jurisdiction"].upper())

    if not item["expect_grounded"]:
        # Out-of-scope question: correctness = did it refuse?
        result["correct_refusal_behavior"] = not data["answer_grounded"]
        return result

    # In-scope question: look for the expected topic among returned sources
    for rank, source in enumerate(data.get("sources", []), start=1):
        if source["jurisdiction"] not in allowed_jurisdictions:
            result["jurisdiction_violation"] = True
        if source["audience"] != item["audience"]:
            result["audience_violation"] = True

        if result["hit"] == 0 and matches_keywords(source, item["expect_keywords_any"]):
            result["hit"] = 1
            result["reciprocal_rank"] = 1.0 / rank

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_API_URL)
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET))
    args = parser.parse_args()

    golden_set = load_golden_set(Path(args.golden_set))
    print(f"Loaded {len(golden_set)} golden questions from {args.golden_set}\n")

    results = []
    for item in golden_set:
        try:
            results.append(evaluate_question(args.url, item))
        except requests.exceptions.ConnectionError:
            print(f"[error] Could not reach {args.url} -- is the API server running?")
            return
        except Exception as e:
            print(f"[error] question '{item['question']}' failed: {e}")

    in_scope = [r for r in results if r["expect_grounded"]]
    out_of_scope = [r for r in results if not r["expect_grounded"]]

    print("=" * 70)
    print("RETRIEVAL QUALITY (in-scope questions)")
    print("=" * 70)
    for r in in_scope:
        status = "HIT " if r["hit"] else "MISS"
        jflag = " [JURISDICTION VIOLATION]" if r["jurisdiction_violation"] else ""
        aflag = " [AUDIENCE VIOLATION]" if r["audience_violation"] else ""
        gflag = " [WRONGLY REFUSED]" if not r["actual_grounded"] else ""
        v = r["top_vector_score"]
        v_str = f"{v:.3f}" if v is not None else "N/A"
        ce = r["top_cross_encoder_score"]
        ce_str = f"{ce:.3f}" if ce is not None else "N/A"
        print(f"  [{status}] rr={r['reciprocal_rank']:.2f} | vscore={v_str} | cescore={ce_str} | {r['question'][:50]}{jflag}{aflag}{gflag}")

    if in_scope:
        hit_rate = sum(r["hit"] for r in in_scope) / len(in_scope)
        mrr = sum(r["reciprocal_rank"] for r in in_scope) / len(in_scope)
        jviol_rate = sum(r["jurisdiction_violation"] for r in in_scope) / len(in_scope)
        aviol_rate = sum(r["audience_violation"] for r in in_scope) / len(in_scope)
        print(f"\n  hit@5:                    {hit_rate:.1%}")
        print(f"  MRR:                      {mrr:.3f}")
        print(f"  jurisdiction_violation_rate: {jviol_rate:.1%}  (should always be 0%)")
        print(f"  audience_violation_rate:     {aviol_rate:.1%}  (should always be 0%)")

    print()
    print("=" * 70)
    print("REFUSAL BEHAVIOR (out-of-scope questions)")
    print("=" * 70)
    for r in out_of_scope:
        status = "CORRECTLY REFUSED" if r["correct_refusal_behavior"] else "WRONGLY ANSWERED"
        v = r["top_vector_score"]
        v_str = f"{v:.3f}" if v is not None else "N/A"
        ce = r["top_cross_encoder_score"]
        ce_str = f"{ce:.3f}" if ce is not None else "N/A"
        print(f"  [{status}] vscore={v_str} | cescore={ce_str} | {r['question'][:50]}")

    if out_of_scope:
        refusal_acc = sum(r["correct_refusal_behavior"] for r in out_of_scope) / len(out_of_scope)
        print(f"\n  refusal_accuracy: {refusal_acc:.1%}")

    print()
    print("=" * 70)
    print("CONFIDENCE_THRESHOLD TUNING DATA")
    print("=" * 70)
    should_ground_scores = [r["top_vector_score"] for r in in_scope if r["top_vector_score"] is not None]
    should_refuse_scores = [r["top_vector_score"] for r in out_of_scope if r["top_vector_score"] is not None]
    if should_ground_scores:
        print(f"  [vector] Should-ground: min={min(should_ground_scores):.3f}, max={max(should_ground_scores):.3f}")
    if should_refuse_scores:
        print(f"  [vector] Should-refuse: min={min(should_refuse_scores):.3f}, max={max(should_refuse_scores):.3f}")

    ce_ground_scores = [r["top_cross_encoder_score"] for r in in_scope if r["top_cross_encoder_score"] is not None]
    ce_refuse_scores = [r["top_cross_encoder_score"] for r in out_of_scope if r["top_cross_encoder_score"] is not None]
    if ce_ground_scores:
        print(f"  [cross-encoder] Should-ground: min={min(ce_ground_scores):.3f}, max={max(ce_ground_scores):.3f}")
    if ce_refuse_scores:
        print(f"  [cross-encoder] Should-refuse: min={min(ce_refuse_scores):.3f}, max={max(ce_refuse_scores):.3f}")

    if ce_ground_scores and ce_refuse_scores:
        gap_low = min(ce_ground_scores)
        gap_high = max(ce_refuse_scores)
        if gap_low > gap_high:
            suggested = (gap_low + gap_high) / 2
            print(f"\n  Clean cross-encoder separation exists -- suggested CROSS_ENCODER_THRESHOLD: {suggested:.3f}")
        else:
            print(f"\n  [warn] Overlap remains even with the cross-encoder "
                  f"(should-ground min={gap_low:.3f}, should-refuse max={gap_high:.3f}). "
                  f"Inspect the specific overlapping questions directly.")


if __name__ == "__main__":
    main()
