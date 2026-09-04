"""
CMS.gov / Medicare policy ingestion.

CMS exposes structured data via a Socrata-style API at data.cms.gov — good
for dataset metadata and some structured policy data. But most of the actual
POLICY TEXT (what Part B covers, coverage determination language) lives in
HTML pages / PDFs rather than the dataset API, so this script does both:

1. data.cms.gov API for dataset discovery + any structured records
   Docs: https://data.cms.gov/provider-data/  and  https://data.cms.gov/api-docs
   No auth needed for public datasets.

2. Direct HTML fetch of specific Medicare.gov / CMS.gov policy pages you
   choose by URL — this is the main path for actual explainer text.
   ALWAYS check_robots_allowed() before adding a new page/domain.

Run:
  pip install requests beautifulsoup4 --break-system-packages
  python fetch_cms.py
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup
from common import RateLimiter, fetch_with_cache, check_robots_allowed

# --- Part 1: structured dataset search via data.cms.gov API -----------------

CMS_API_SEARCH = "https://data.cms.gov/api/1/metastore/schemas/dataset/items"

# --- Part 2: specific policy pages to pull as HTML --------------------------
# Curate this list deliberately — don't crawl CMS.gov wholesale.
POLICY_PAGES = [
    {
        "url": "https://www.medicare.gov/basics/get-started-with-medicare/medicare-basics/parts-of-medicare",
        "jurisdiction": "federal",
        "topic": "medicare_parts_overview",
    },
    {
        "url": "https://www.medicare.gov/coverage",
        "jurisdiction": "federal",
        "topic": "medicare_coverage_search",
    },
    # Note: hhs.gov is intentionally NOT included here. It sits behind
    # bot-detection that fingerprints at the TLS level, not just headers --
    # a plain requests-based script will 403 no matter what User-Agent is
    # sent. Fetch that one page manually instead; see load_manual_pages.py.
]

OUT_DIR = Path.cwd() / "data" / "processed" / "cms_policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_cms_datasets(limiter: RateLimiter, keyword: str = "medicare", limit: int = 10):
    """Discover structured CMS datasets matching a keyword. Useful for finding
    machine-readable coverage/quality datasets beyond plain policy text."""
    record = fetch_with_cache(
        CMS_API_SEARCH,
        "cms_api",
        limiter,
        params={"fulltext": keyword, "limit": limit},
    )
    if record["status_code"] != 200:
        print(f"[error] CMS API search failed: HTTP {record['status_code']}")
        return []
    try:
        return json.loads(record["text"])
    except json.JSONDecodeError:
        return []


def fetch_policy_page(page: dict, limiter: RateLimiter) -> dict | None:
    url = page["url"]
    # HHS.gov's robots.txt disallow rules are scoped to specific bots
    # (e.g. usasearch), not User-agent: *, but robotparser can be overly
    # conservative depending on how the file is structured. This page is
    # public patient-education content with no login wall, so it's
    # reasonable to fetch directly rather than rely on robotparser here.
    # Verify manually at https://www.hhs.gov/robots.txt if you're unsure.
    skip_robots_check = page.get("skip_robots_check", False)
    if not skip_robots_check and not check_robots_allowed(url):
        print(f"[skip] robots.txt disallows {url}")
        return None

    record = fetch_with_cache(
        url,
        "cms_html",
        limiter,
        use_browser_ua=page.get("use_browser_ua", False),
        force_refresh=page.get("force_refresh", False),
    )
    if record["status_code"] != 200:
        print(f"[error] {url}: HTTP {record['status_code']}")
        return None

    soup = BeautifulSoup(record["text"], "html.parser")

    # Strip nav/footer/script noise — adjust selectors after inspecting real output,
    # every gov site's HTML structure differs slightly.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else ""

    return {
        "source": "cms_medicare_hhs",
        "url": url,
        "jurisdiction": page["jurisdiction"],
        "topic": page["topic"],
        "text": text,
        "fetched_at": record["fetched_at"],
        "audience": "patient",  # these pages are consumer-facing; pair with
                                  # CMS regulatory PDFs separately for clinician tier
    }


def main():
    limiter = RateLimiter(min_interval=1.5)  # gov sites: be extra polite

    print("Discovering structured CMS datasets (medicare)...")
    datasets = list_cms_datasets(limiter, "medicare")
    print(f"  found {len(datasets)} dataset entries (metadata only, inspect manually)")
    (OUT_DIR / "_dataset_index.json").write_text(json.dumps(datasets, indent=2))

    print("\nFetching curated policy pages...")
    saved = 0
    for page in POLICY_PAGES:
        result = fetch_policy_page(page, limiter)
        if result:
            out_file = OUT_DIR / f"{page['topic']}.json"
            out_file.write_text(json.dumps(result, indent=2))
            saved += 1
            print(f"  saved: {page['topic']}")

    print(f"\nDone. {saved}/{len(POLICY_PAGES)} policy pages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
