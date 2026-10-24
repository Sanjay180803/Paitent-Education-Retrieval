"""
State Department of Insurance ingestion.

No unified API across states — this is genuine page-by-page HTML scraping,
and every state's site structure is different. Start with 3 states (enough
to prove jurisdiction-precedence logic), expand later.

Pages are listed in config/state_pages.toml. To add a state, add a
[[page]] entry there (no code change needed).

Run:
  python fetch_state_insurance.py                       # all pages in the config
  python fetch_state_insurance.py --state CA --state TX  # a subset
  python fetch_state_insurance.py --config my_pages.toml
"""

import argparse
import json
import tomllib
from pathlib import Path

from bs4 import BeautifulSoup
from common import RateLimiter, fetch_with_cache, check_robots_allowed

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "state_pages.toml"


def load_state_pages(config_path: Path, states: list[str] | None = None) -> list[dict]:
    with config_path.open("rb") as f:
        pages = tomllib.load(f).get("page", [])
    for page in pages:
        missing = {"state", "topic", "url"} - page.keys()
        if missing:
            raise ValueError(f"{config_path}: page entry {page} is missing {sorted(missing)}")
    if states:
        wanted = {st.upper() for st in states}
        pages = [p for p in pages if p["state"].upper() in wanted]
    return pages


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", action="append", help="only fetch this state (repeatable)")
    args = parser.parse_args()

    STATE_PAGES = load_state_pages(args.config, args.state)
    if not STATE_PAGES:
        print(f"No pages to fetch from {args.config}.")
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
