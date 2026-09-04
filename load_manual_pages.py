"""
Loader for pages you fetch manually in a browser instead of scripting
(e.g. hhs.gov, which sits behind bot-detection that blocks requests-based
scripts at the TLS fingerprint level, not just the User-Agent header --
not worth fighting for a single page).

How to use:
  1. Open the page in your browser, e.g.:
     https://www.hhs.gov/hipaa/for-individuals/index.html
  2. Save it: Ctrl+S -> "Webpage, HTML only" -> save into
     data/raw/manual/hipaa_patient_rights.html
     (create the manual/ folder if it doesn't exist)
  3. Add an entry to MANUAL_PAGES below matching the filename.
  4. Run: python load_manual_pages.py

This produces the same processed JSON shape as fetch_cms.py's output, so
it drops into the same chunking/embedding step downstream.
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup

RAW_MANUAL_DIR = Path.cwd() / "data" / "raw" / "manual"
RAW_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path.cwd() / "data" / "processed" / "cms_policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# filename (inside data/raw/manual/) -> metadata
MANUAL_PAGES = {
    "hipaa_patient_rights.html": {
        "source": "cms_medicare_hhs",
        "url": "https://www.hhs.gov/hipaa/for-individuals/index.html",
        "jurisdiction": "federal",
        "topic": "hipaa_patient_rights",
        "audience": "patient",
    },
}


def extract_text(html_path: Path) -> str:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    return main.get_text(separator="\n", strip=True) if main else ""


def main():
    saved = 0
    for filename, meta in MANUAL_PAGES.items():
        html_path = RAW_MANUAL_DIR / filename
        if not html_path.exists():
            print(f"[skip] {filename} not found in {RAW_MANUAL_DIR} -- save it there first")
            continue

        text = extract_text(html_path)
        record = {
            **meta,
            "text": text,
            "fetched_at": "manual",  # you know roughly when you saved it
        }
        out_file = OUT_DIR / f"{meta['topic']}.json"
        out_file.write_text(json.dumps(record, indent=2))
        saved += 1
        print(f"  saved: {meta['topic']} ({len(text)} chars)")

    print(f"\nDone. {saved}/{len(MANUAL_PAGES)} manual pages processed.")


if __name__ == "__main__":
    main()
