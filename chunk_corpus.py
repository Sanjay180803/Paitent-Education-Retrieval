"""
Chunks the unified corpus (data_corpus/corpus.jsonl) into retrieval-sized
pieces, tagged with the metadata your retrieval layer will filter on later
(source_type, jurisdiction, audience/reading_level, effective_date).

Strategy: paragraph-aware chunking, not a blind fixed-token slice.
  - Split each doc's text on blank lines / paragraph breaks (already
    newline-separated from the BeautifulSoup extraction in fetch_cms.py /
    fetch_state_insurance.py; MedlinePlus/PubMed text is closer to a single
    block, so those get sentence-grouped instead).
  - Greedily pack paragraphs into chunks up to TARGET_WORDS, never splitting
    a paragraph mid-sentence.
  - If a single paragraph alone exceeds TARGET_WORDS (rare, but some policy
    pages have huge unbroken blocks), fall back to sentence-level splitting
    for just that paragraph.
  - Add a small word-overlap between consecutive chunks from the same doc so
    context isn't lost at a chunk boundary.

This keeps the "don't split a coverage table / dosage line mid-sentence"
property that's the actual point of section-aware chunking, without needing
an NLP sentence tokenizer for the simple cases (paragraph-level is often
enough here since most sources are prose, not code).

Run:
  python chunk_corpus.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path.cwd()
IN_FILE = PROJECT_ROOT / "data_corpus" / "corpus.jsonl"
OUT_FILE = PROJECT_ROOT / "data_corpus" / "chunks.jsonl"

TARGET_WORDS = 220   # ~roughly 300 tokens, a common embedding-model sweet spot
OVERLAP_WORDS = 40    # context carried into the next chunk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    # Handles both "\n\n"-separated HTML extraction and single-blob text
    # (MedlinePlus/PubMed) -- falls back to sentence grouping for the latter.
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return paras if len(paras) > 1 else split_sentences(text)


def pack_chunks(units: list[str], target_words: int, overlap_words: int) -> list[str]:
    """Greedily pack paragraph/sentence units into ~target_words chunks,
    carrying a small word-overlap forward for context continuity."""
    chunks = []
    current_words: list[str] = []

    for unit in units:
        unit_words = unit.split()

        # A single unit bigger than the whole target -- split it internally
        # by sentence rather than truncating or overflowing wildly.
        if len(unit_words) > target_words:
            for sent in split_sentences(unit):
                sent_words = sent.split()
                if len(current_words) + len(sent_words) > target_words and current_words:
                    chunks.append(" ".join(current_words))
                    current_words = current_words[-overlap_words:]
                current_words.extend(sent_words)
            continue

        if len(current_words) + len(unit_words) > target_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:]

        current_words.extend(unit_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def main():
    if not IN_FILE.exists():
        print(f"[error] {IN_FILE} not found -- run build_corpus.py first")
        return

    all_chunks = []
    doc_count = 0
    skipped_empty = 0

    with IN_FILE.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = (doc.get("text") or "").strip()

            if not text:
                skipped_empty += 1
                continue

            doc_count += 1
            doc_id = f"doc_{line_num:04d}"

            units = split_paragraphs(text)
            chunk_texts = pack_chunks(units, TARGET_WORDS, OVERLAP_WORDS)

            for i, chunk_text in enumerate(chunk_texts):
                all_chunks.append({
                    "chunk_id": f"{doc_id}_c{i:03d}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks_in_doc": len(chunk_texts),
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    # metadata carried through for retrieval-time filtering
                    "source_type": doc.get("source_type", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "audience": doc.get("audience", "patient"),
                    "jurisdiction": doc.get("jurisdiction", "federal"),
                    "effective_date": doc.get("effective_date", ""),
                })

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Processed {doc_count} documents ({skipped_empty} skipped for empty text)")
    print(f"Produced {len(all_chunks)} chunks")
    if all_chunks:
        avg_words = sum(c["word_count"] for c in all_chunks) / len(all_chunks)
        print(f"Average chunk size: {avg_words:.0f} words")
    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
