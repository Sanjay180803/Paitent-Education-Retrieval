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

Question sets (eval/):
  dev.json          23 questions used to choose CROSS_ENCODER_THRESHOLD.
  heldout.json      60 questions written after the threshold was fixed;
                    never used for tuning.
  adversarial.json  30 topically adjacent out-of-scope questions; never
                    used for tuning.
See eval/README.md for the protocol.

Run (with the API server already running):
  python run_eval.py                                   # all three sets
  python run_eval.py --set eval/heldout.json
  python run_eval.py --json-out eval/results/latest.json
"""

import argparse
import json
from pathlib import Path

import requests

EVAL_DIR = Path(__file__).parent / "eval"
DEFAULT_SETS = [EVAL_DIR / "dev.json", EVAL_DIR / "heldout.json", EVAL_DIR / "adversarial.json"]
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


def summarize(results: list[dict]) -> dict:
    """Aggregate metrics for one question set. A question counts as correct
    when an answerable one is grounded AND a top-5 source matches, or an
    out-of-scope one is refused."""
    in_scope = [r for r in results if r["expect_grounded"]]
    out_scope = [r for r in results if not r["expect_grounded"]]

    def rate(xs, key):
        return round(sum(bool(x[key]) for x in xs) / len(xs), 4) if xs else None

    correct = sum(1 for r in in_scope if r["actual_grounded"] and r["hit"]) + sum(
        1 for r in out_scope if r["correct_refusal_behavior"]
    )
    return {
        "n": len(results),
        "n_answerable": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "accuracy": round(correct / len(results), 4) if results else None,
        "hit_at_5": rate(in_scope, "hit"),
        "mrr": round(sum(r["reciprocal_rank"] for r in in_scope) / len(in_scope), 4) if in_scope else None,
        "wrongly_refused": sum(1 for r in in_scope if not r["actual_grounded"]),
        "refusal_accuracy": rate(out_scope, "correct_refusal_behavior"),
        "jurisdiction_violations": sum(r["jurisdiction_violation"] for r in in_scope),
        "audience_violations": sum(r["audience_violation"] for r in in_scope),
    }


def run_set(api_url: str, path: Path) -> list[dict] | None:
    golden_set = load_golden_set(path)
    print(f"\n{'#' * 70}\n# {path.name}: {len(golden_set)} questions\n{'#' * 70}\n")
    results = []
    for item in golden_set:
        try:
            results.append(evaluate_question(api_url, item))
        except requests.exceptions.ConnectionError:
            print(f"[error] Could not reach {api_url} -- is the API server running?")
            return None
        except Exception as e:
            print(f"[error] question '{item['question']}' failed: {e}")
    report(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_API_URL)
    parser.add_argument("--set", dest="sets", action="append", type=Path,
                        help="question set to run (repeatable; default: all of eval/)")
    parser.add_argument("--json-out", type=Path, help="write per-set metrics and per-question results here")
    args = parser.parse_args()

    out = {}
    for path in args.sets or DEFAULT_SETS:
        results = run_set(args.url, path)
        if results is None:
            raise SystemExit(1)
        out[path.stem] = {"metrics": summarize(results), "results": results}

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, block in out.items():
        m = block["metrics"]
        print(f"  {name:12s} n={m['n']:3d}  accuracy={m['accuracy']}  hit@5={m['hit_at_5']}  "
              f"MRR={m['mrr']}  refusal_acc={m['refusal_accuracy']}  "
              f"wrongly_refused={m['wrongly_refused']}  jurisdiction_violations={m['jurisdiction_violations']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json_out}")


def report(results: list[dict]) -> None:
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
