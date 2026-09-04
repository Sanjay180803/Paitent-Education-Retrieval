"""
PubMed / PMC ingestion via NCBI E-utilities.

This gives you the "clinician tier" technical explainer to pair against each
MedlinePlus "patient tier" explainer for the same condition.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Get a free API key (optional but recommended, raises rate limit 3->10 req/sec):
  https://www.ncbi.nlm.nih.gov/account/  -> Settings -> API Key Management

Flow:
  1. esearch: term -> list of PMIDs (or PMCIDs for full text)
  2. efetch: PMIDs -> abstract/article XML

We pull PMC open-access full text where available (richer than an abstract),
falling back to PubMed abstracts otherwise.

Run:
  pip install requests --break-system-packages
  export NCBI_API_KEY=your_key_here   # optional
  python fetch_pubmed.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from common import RateLimiter, fetch_with_cache

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

API_KEY = os.environ.get("NCBI_API_KEY")  # None is fine, just slower

# Match these to the same conditions list in fetch_medlineplus.py
CONDITIONS = [
    "diabetes management guidelines",
    "hypertension treatment guidelines",
    "asthma management guidelines",
    "heart failure management",
    "chronic kidney disease management",
    "major depressive disorder treatment",
    "COPD management guidelines",
    "obesity management guidelines",
]

OUT_DIR = Path.cwd() / "data" / "processed" / "pubmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NCBI wants >=3 params on every call: db, plus your term/id, plus api_key if you have one
def _params(extra: dict) -> dict:
    p = {"retmode": "xml"}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(extra)
    return p


def search_pmids(term: str, limiter: RateLimiter, retmax: int = 3) -> list[str]:
    record = fetch_with_cache(
        ESEARCH_URL,
        "pubmed",
        limiter,
        params=_params({"db": "pubmed", "term": term, "retmax": retmax, "sort": "relevance"}),
    )
    root = ET.fromstring(record["text"])
    return [id_el.text for id_el in root.findall(".//Id")]


def fetch_abstract(pmid: str, limiter: RateLimiter) -> dict | None:
    record = fetch_with_cache(
        EFETCH_URL,
        "pubmed",
        limiter,
        params=_params({"db": "pubmed", "id": pmid, "rettype": "abstract"}),
    )
    try:
        root = ET.fromstring(record["text"])
    except ET.ParseError:
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        return None

    title_el = article.find(".//ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    abstract_parts = article.findall(".//AbstractText")
    abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts)

    year_el = article.find(".//PubDate/Year")
    year = year_el.text if year_el is not None else None

    return {
        "source": "pubmed",
        "pmid": pmid,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "title": title,
        "abstract": abstract,
        "pub_year": year,
        "audience": "clinician",  # PubMed abstracts are technical/clinical register
        "fetched_at": record["fetched_at"],
    }


def main():
    limiter = RateLimiter(min_interval=0.35 if API_KEY else 0.4)  # ~3/sec no key, headroom with key
    all_results = []

    for term in CONDITIONS:
        print(f"Searching: {term}")
        pmids = search_pmids(term, limiter)
        for pmid in pmids:
            article = fetch_abstract(pmid, limiter)
            if article:
                article["query_term"] = term
                all_results.append(article)
                out_file = OUT_DIR / f"{pmid}.json"
                out_file.write_text(__import__("json").dumps(article, indent=2))
                print(f"  saved PMID {pmid}: {article['title'][:70]}")

    print(f"\nDone. {len(all_results)} articles saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
