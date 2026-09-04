"""
State Department of Insurance ingestion.

No unified API across states — this is genuine page-by-page HTML scraping,
and every state's site structure is different. Start with 3 states (enough
to prove jurisdiction-precedence logic), expand later.

You MUST fill in real URLs for each state's consumer guide pages yourself —
the ones below are placeholders showing the expected shape. Search
"<state> department of insurance consumer guide health insurance" to find
the real pages, then check_robots_allowed() before adding each one.

Run:
  pip install requests beautifulsoup4 --break-system-packages
  python fetch_state_insurance.py
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup
from common import RateLimiter, fetch_with_cache, check_robots_allowed

# Fill these in with real, verified URLs per state before running.
STATE_PAGES = [
    {
        "url": "https://www.insurance.ca.gov/01-consumers/110-health/20-look/index.cfm",
        "state": "CA",
        "topic": "health_insurance_consumer_guide",
    },
    {
        "url": "https://www.tdi.texas.gov/pubs/consumer/cb005.html",
        "state": "TX",
        "topic": "health_insurance_consumer_guide",
    },
    {
        "url": "https://myfloridacfo.com/division/ica/fullcoverage/health-life",
        "state": "FL",
        "topic": "health_insurance_consumer_guide",
    },
]

OUT_DIR = Path.cwd() / "data" / "processed" / "state_insurance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_state_page(page: dict, limiter: RateLimiter) -> dict | None:
    url = page["url"]
    if not check_robots_allowed(url):
        print(f"[skip] robots.txt disallows {url}")
        return None

    record = fetch_with_cache(url, "state_html", limiter)
    if record["status_code"] != 200:
        print(f"[error] {url}: HTTP {record['status_code']}")
        return None

    soup = BeautifulSoup(record["text"], "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else ""

    return {
        "source": "state_insurance_dept",
        "url": url,
        "jurisdiction": page["state"],  # e.g. "CA" -- used for precedence logic later
        "topic": page["topic"],
        "text": text,
        "fetched_at": record["fetched_at"],
        "audience": "patient",
    }


def main():
    if not STATE_PAGES:
        print("STATE_PAGES is empty — add real, verified URLs before running.")
        print("See the module docstring for how to find them.")
        return

    limiter = RateLimiter(min_interval=1.5)
    saved = 0
    for page in STATE_PAGES:
        result = fetch_state_page(page, limiter)
        if result:
            out_file = OUT_DIR / f"{page['state']}_{page['topic']}.json"
            out_file.write_text(json.dumps(result, indent=2))
            saved += 1
            print(f"  saved: {page['state']} / {page['topic']}")

    print(f"\nDone. {saved}/{len(STATE_PAGES)} state pages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
