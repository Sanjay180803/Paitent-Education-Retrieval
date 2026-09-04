"""
Shared utilities for all ingestion scripts:
- polite rate limiting
- raw response caching (so you don't re-fetch on every run, and it gives you
  a natural version history since you can diff cache files over time)
- robots.txt checking
"""

import hashlib
import json
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

USER_AGENT = "PatientEducationRAG/0.1 (+student portfolio project; contact: you@example.com)"

# Some gov sites (e.g. hhs.gov) reject unrecognized bot-style User-Agents outright,
# even when robots.txt itself allows the path. This browser-style fallback is used
# only where a plain server-level 403 blocks a legitimate, low-volume, non-authenticated
# fetch of public information -- not to bypass robots.txt disallow rules.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Resolved from the current working directory, NOT __file__ location.
# All scripts in this project assume you run `python script.py` from the
# project root (e.g. `cd D:\Masters\Health_SD` first) -- this makes every
# script write to the same data/ tree regardless of exactly where the .py
# files themselves are placed, which was the source of the scattered
# data/, data_pubmed/, data_insurance_policy/ folders earlier.
RAW_DIR = Path.cwd() / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class RateLimiter:
    """Simple sleep-based rate limiter. min_interval in seconds between requests."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


def check_robots_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check robots.txt before scraping a domain. Always call this first for
    any HTML-scraping source (MedlinePlus HTML fallback, HHS.gov, state sites)."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as e:
        print(f"[warn] could not read robots.txt at {robots_url}: {e}")
        return True  # fail open, but log it — use your judgment
    return rp.can_fetch(user_agent, url)


def cache_path_for(url: str, subdir: str) -> Path:
    """Deterministic cache filename from a URL hash, namespaced by source."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    d = RAW_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def fetch_with_cache(
    url: str,
    subdir: str,
    limiter: RateLimiter,
    params: dict | None = None,
    force_refresh: bool = False,
    timeout: int = 20,
    use_browser_ua: bool = False,
) -> dict:
    """
    GET a URL, caching the raw response + metadata to disk keyed by URL hash.
    Returns dict: {url, params, fetched_at, status_code, text}
    Re-running the pipeline reuses the cache unless force_refresh=True —
    this also becomes your "effective_date" / version signal: compare
    fetched_at timestamps across runs to see what changed.

    use_browser_ua: set True only for sites that hard-403 a plain script
    User-Agent even though the content is public and robots.txt allows it
    (e.g. hhs.gov). Leave False everywhere else.
    """
    cache_file = cache_path_for(url + json.dumps(params or {}, sort_keys=True), subdir)

    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text())

    limiter.wait()
    ua = BROWSER_USER_AGENT if use_browser_ua else USER_AGENT
    resp = requests.get(url, params=params, headers={"User-Agent": ua}, timeout=timeout)
    # requests sometimes guesses the wrong encoding when a server doesn't
    # declare charset cleanly, producing mojibake (e.g. "don't" -> "donâ€™t").
    # apparent_encoding sniffs the actual bytes instead of trusting a
    # missing/wrong header.
    if resp.encoding is None or resp.encoding.lower() in ("iso-8859-1", "ascii"):
        resp.encoding = resp.apparent_encoding
    record = {
        "url": resp.url,
        "params": params,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status_code": resp.status_code,
        "text": resp.text,
    }
    cache_file.write_text(json.dumps(record))
    return record
