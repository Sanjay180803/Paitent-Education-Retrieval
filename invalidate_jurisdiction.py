"""
Run this after re-fetching/re-embedding a jurisdiction's data, to atomically
invalidate stale cached answers for that jurisdiction. See query_cache.py's
module docstring for why a generation bump is race-free where a
delete-matching-keys approach would not be.

Usage:
  python invalidate_jurisdiction.py federal   # after updating Medicare/HIPAA/MedlinePlus/PubMed content
  python invalidate_jurisdiction.py CA        # after updating California-specific content
  python invalidate_jurisdiction.py TX
  python invalidate_jurisdiction.py FL

Typical workflow after updating a source:
  python fetch_cms.py                  # or fetch_state_insurance.py, etc.
  python build_corpus.py
  python chunk_corpus.py
  python embed_and_index.py
  python invalidate_jurisdiction.py federal
"""

import sys

from query_cache import QueryCache

VALID_JURISDICTIONS = {"federal", "CA", "TX", "FL"}  # extend as you add states


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <jurisdiction>")
        print(f"Valid jurisdictions: {', '.join(sorted(VALID_JURISDICTIONS))}")
        sys.exit(1)

    jurisdiction = sys.argv[1]
    if jurisdiction not in VALID_JURISDICTIONS and jurisdiction.upper() not in VALID_JURISDICTIONS:
        print(f"[warn] '{jurisdiction}' isn't in the known list ({VALID_JURISDICTIONS}), proceeding anyway")

    cache = QueryCache()
    old_gen = cache.get_generation(jurisdiction)
    new_gen = cache.bump_generation(jurisdiction)
    print(f"Bumped '{jurisdiction}' generation: {old_gen} -> {new_gen}")
    print("All cached answers depending on this jurisdiction are now orphaned "
          "and will be recomputed on next request.")


if __name__ == "__main__":
    main()
