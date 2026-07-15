"""Stratified sample for human annotation.

Goal: pick N prompts from a JSONL such that we cover as many unique
source docs as possible (max 1 prompt per doc), then top up randomly
from remaining if N exceeds the doc count.

Why: annotation budget is small (~100 / annotator / layer for stable
inter-annotator agreement). Spreading across docs avoids over-weighting
any single document and gives broader coverage of the corpus.

Usage:
    python scripts/sample_for_annotation.py \\
        --input data/in-scope-eloq-final.jsonl \\
        --output data/in-scope-eloq-sample.jsonl \\
        --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """At most 1 prompt per doc until n is hit; then randomly fill from rest."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r.get("doc_id") or "_unknown"].append(r)

    rng = random.Random(seed)
    picked: list[dict] = []
    seen_doc_ids: set[str] = set()

    # Pass 1: one per doc (random pick from each doc's pool)
    docs = list(by_doc.keys())
    rng.shuffle(docs)
    for doc_id in docs:
        if len(picked) >= n:
            break
        picked.append(rng.choice(by_doc[doc_id]))
        seen_doc_ids.add(doc_id)

    # Pass 2: top up by sampling from remaining (multi-question docs)
    if len(picked) < n:
        leftover = [r for doc_id in seen_doc_ids
                    for r in by_doc[doc_id]
                    if r not in picked]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])

    # Stable order: by prompt_id
    picked.sort(key=lambda r: r.get("prompt_id", ""))
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    print(f"Input: {len(rows)} rows from {args.input}")

    sample = stratified_sample(rows, args.n, args.seed)
    docs = {r.get("doc_id") for r in sample}
    print(f"Sampled: {len(sample)} rows | {len(docs)} unique docs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
