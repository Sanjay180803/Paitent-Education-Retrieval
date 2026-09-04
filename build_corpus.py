"""
Consolidates processed JSON records from your three data roots into a single
unified corpus file, normalizing each record to a common schema before the
chunking/embedding step.

Your current layout (adjust ROOTS below if yours differs):
  data/processed/medlineplus/*.json
  data_pubmed/processed/pubmed/*.json
  data_insurance_policy/processed/cms_policy/*.json
  data_insurance_policy/processed/state_insurance/*.json

Run from the same folder that contains data/, data_pubmed/, data_insurance_policy/:
  python build_corpus.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path.cwd()

# Each entry: (glob pattern relative to PROJECT_ROOT, source_type tag)
# All sources now write to the same data/processed/ tree (see common.py's
# comment on Path.cwd()-based resolution) -- no more scattered
# data_pubmed/, data_insurance_policy/ roots.
SOURCES = [
    ("data/processed/medlineplus/*.json", "medlineplus"),
    ("data/processed/pubmed/*.json", "pubmed"),
    ("data/processed/cms_policy/*.json", "cms_medicare_hhs"),
    ("data/processed/state_insurance/*.json", "state_insurance_dept"),
]

OUT_FILE = PROJECT_ROOT / "data_corpus" / "corpus.jsonl"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def normalize(record: dict, source_type: str) -> dict:
    """Map each source's slightly different field names onto one common schema."""
    # Text content lives under different keys per source
    text = (
        record.get("full_summary")  # medlineplus
        or record.get("abstract")   # pubmed
        or record.get("text")       # cms_policy / state_insurance
        or ""
    )

    return {
        "source_type": source_type,
        "url": record.get("url", ""),
        "title": record.get("title") or record.get("topic") or record.get("query_term") or "",
        "text": text,
        "audience": record.get("audience", "patient"),  # reading_level tag
        "jurisdiction": record.get("jurisdiction", "federal"),  # medlineplus/pubmed default to federal/general
        "effective_date": record.get("pub_year") or record.get("fetched_at", ""),
        "fetched_at": record.get("fetched_at", ""),
    }


def main():
    all_records = []
    counts = {}
    skipped = 0

    for pattern, source_type in SOURCES:
        matches = list(PROJECT_ROOT.glob(pattern))
        counts[source_type] = 0
        for path in matches:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"[warn] could not parse {path}: {e}")
                continue

            if not isinstance(raw, dict):
                # e.g. _dataset_index.json from fetch_cms.py is a list of
                # dataset metadata, not a single record -- skip it here.
                print(f"[skip] {path.name}: not a single record (got {type(raw).__name__})")
                skipped += 1
                continue

            all_records.append(normalize(raw, source_type))
            counts[source_type] += 1

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Corpus build summary:")
    for source_type, n in counts.items():
        print(f"  {source_type}: {n} records")
    if skipped:
        print(f"  skipped: {skipped} non-record files")
    print(f"\nTotal: {len(all_records)} records written to {OUT_FILE}")


if __name__ == "__main__":
    main()
