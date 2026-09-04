"""
MedlinePlus ingestion.

Two ways in, both official / no-key-needed:

1. MedlinePlus Connect API — look up a health topic by code (e.g. ICD-10) or
   by search term, get back structured XML/JSON with a summary + link.
   Docs: https://medlineplus.gov/connect/service.html

2. MedlinePlus Web Service (Topic search) — full XML export of health topic
   pages, richer than Connect (full summary text, related topics, sources).
   Docs: https://medlineplus.gov/webservices.html

We use #2 here since you want full explainer text for the "patient tier" of
your RAG corpus, not just a short summary blurb.

Base search endpoint:
  https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=<query>

Run:
  pip install requests --break-system-packages
  python fetch_medlineplus.py
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from common import RateLimiter, fetch_with_cache, check_robots_allowed

HTML_TAG_RE = re.compile(r"<[^>]+>")
BLOCK_TAG_RE = re.compile(r"</?(li|ul|ol|p|br|div)[^>]*>", re.IGNORECASE)


def strip_html(text: str) -> str:
    """MedlinePlus's search API embeds highlighting markup (e.g.
    <span class="qt0">term</span>) as literal text inside content fields,
    not as real child XML elements -- so itertext() alone doesn't catch it.
    Block-level tags are replaced with a space first so removing them
    doesn't glue adjacent sentences together (e.g. "insulin.</li><li>If"
    becoming "insulin.If" instead of "insulin. If")."""
    text = BLOCK_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

# Start narrow — expand this list once the pipeline works end to end.
CONDITIONS = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "chronic kidney disease",
    "depression",
    "COPD",
    "obesity",
]

OUT_DIR = Path.cwd() / "data" / "processed" / "medlineplus"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_topic(term: str, limiter: RateLimiter) -> dict | None:
    params = {"db": "healthTopics", "term": term, "retmax": "1"}
    record = fetch_with_cache(BASE_URL, "medlineplus", limiter, params=params)

    if record["status_code"] != 200:
        print(f"[error] {term}: HTTP {record['status_code']}")
        return None

    root = ET.fromstring(record["text"])
    doc = root.find(".//document")
    if doc is None:
        print(f"[warn] no result for '{term}'")
        return None

    def get_field(name):
        el = doc.find(f"./content[@name='{name}']")
        return strip_html("".join(el.itertext())) if el is not None else ""

    return {
        "source": "medlineplus",
        "query_term": term,
        "url": doc.get("url"),
        "title": get_field("title"),
        "full_summary": get_field("FullSummary"),
        "audience": "patient",  # MedlinePlus is written at consumer reading level
        "fetched_at": record["fetched_at"],
    }


def main():
    if not check_robots_allowed("https://wsearch.nlm.nih.gov/robots.txt"):
        print("[warn] robots.txt disallows this path — verify manually before proceeding")

    limiter = RateLimiter(min_interval=1.0)  # 1 req/sec, polite default
    results = []

    for term in CONDITIONS:
        print(f"Fetching: {term}")
        topic = fetch_topic(term, limiter)
        if topic:
            results.append(topic)
            out_file = OUT_DIR / f"{term.replace(' ', '_')}.json"
            out_file.write_text(__import__("json").dumps(topic, indent=2))

    print(f"\nDone. {len(results)}/{len(CONDITIONS)} topics saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
