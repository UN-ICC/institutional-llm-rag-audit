"""Stage 3 of OOS pipeline: stratified sample for human review.

Pipeline: generation → retriever_validation → sample_for_review → human annotation.

Reads the retriever-validated annotated JSONL (every prompt with all its
filter signals attached) and pulls a stratified ~10% sample across:

  - random:               unbiased estimate of overall quality
  - borderline_retriever: top-1 similarity score near the pass threshold
                          (where the LLM judge could have gone either way)
  - filter_failure:       prompts that DIDN'T pass either auto-filter
                          (verifies the filters are doing the right thing)

Output: review_sample.jsonl with `sample_stratum` field.
The annotation viewer is then built from this file for human labeling.

Run:
  python -m evalsuite.generators.out_of_scope.sample_for_review
  python -m evalsuite.generators.out_of_scope.sample_for_review --n-random 20
"""

from __future__ import annotations

from evalsuite._io import read_jsonl, save_jsonl

import argparse
import random
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/out-of-scope/eloq")
# Stratification thresholds
BORDERLINE_BAND = 0.10  # within ±0.10 of pass-threshold top-1 cosine sim


def stratify(
    rows: list[dict],
    n_random: int = 20,
    n_borderline: int = 10,
    n_failure: int = 10,
    median_threshold: float | None = None,
    seed: int = 42,
) -> list[dict]:
    """Build a stratified sample. `rows` should be the annotated jsonl
    (`prompts_with_retriever.jsonl`)."""
    rng = random.Random(seed)

    # Reasonable default: median of the top-1 retriever scores defines the
    # "borderline" band centre. Users can override.
    if median_threshold is None:
        sims = sorted(
            r["retriever_top_k_scores"][0]
            for r in rows
            if r.get("retriever_top_k_scores")
        )
        median_threshold = sims[len(sims) // 2] if sims else 0.5

    # filter_failure: retriever validation said the question is answerable from
    # the corpus (i.e., it leaked back into scope and was dropped). Schema:
    # single-embedder runs write `retriever_pass_this_run`; the combined
    # multi-embedder file writes `retriever_pass`. Treat either as authoritative.
    def _passed(r):
        if "retriever_pass" in r:
            return r["retriever_pass"]
        return r.get("retriever_pass_this_run")
    failures = [r for r in rows if _passed(r) is False]
    failure = rng.sample(failures, min(n_failure, len(failures)))
    failure_ids = {r["prompt_id"] for r in failure}

    # borderline_retriever: top-1 sim within ±BORDERLINE_BAND of median, AND
    # not already in failure
    border_pool = [
        r for r in rows
        if r["prompt_id"] not in failure_ids
        and r.get("retriever_top_k_scores")
        and abs(r["retriever_top_k_scores"][0] - median_threshold) <= BORDERLINE_BAND
    ]
    borderline = rng.sample(border_pool, min(n_borderline, len(border_pool)))
    border_ids = {r["prompt_id"] for r in borderline}

    # random: from everything else
    chosen = failure_ids | border_ids
    rest = [r for r in rows if r["prompt_id"] not in chosen]
    randoms = rng.sample(rest, min(n_random, len(rest)))

    sample = failure + borderline + randoms
    rng.shuffle(sample)

    for r in sample:
        if r["prompt_id"] in failure_ids:
            r["sample_stratum"] = "filter_failure"
        elif r["prompt_id"] in border_ids:
            r["sample_stratum"] = "borderline_retriever"
        else:
            r["sample_stratum"] = "random"
    return sample


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4: stratified sample for human review (OOS)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input", type=Path, default=None,
                   help="Input file (default: <output-dir>/prompts_with_retriever.jsonl, "
                        "falls back to prompts_filtered.jsonl if no retriever stage yet)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-borderline", type=int, default=10)
    p.add_argument("--n-failure", type=int, default=10)
    p.add_argument("--threshold", type=float, default=None,
                   help="Borderline band centre (default: median of top-1 scores)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.input is None:
        # Prefer the cross-verified combined file (has retriever_pass = AND
        # across all per-embedder runs); fall back to the legacy single
        # combined file; then the most recent per-embedder file; finally,
        # raw prompts.jsonl.
        combined  = args.output_dir / "prompts_with_retriever_combined.jsonl"
        legacy    = args.output_dir / "prompts_with_retriever.jsonl"
        per_embedder = sorted(args.output_dir.glob("prompts_with_retriever__*.jsonl"))
        if combined.exists():
            args.input = combined
            print(f"  (using cross-verified combined file: {args.input.name})")
        elif legacy.exists():
            args.input = legacy
        elif per_embedder:
            args.input = per_embedder[-1]
            print(f"  (using single-embedder file: {args.input.name})")
        else:
            args.input = args.output_dir / "prompts.jsonl"
            print(f"  (retriever stage not run yet; sampling from {args.input})")
    if args.output is None:
        args.output = args.output_dir / "review_sample.jsonl"

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    sample = stratify(rows, args.n_random, args.n_borderline, args.n_failure,
                      args.threshold, args.seed)
    save_jsonl(sample, args.output)

    strat = Counter(r["sample_stratum"] for r in sample)
    docs = len(set(r["doc_id"] for r in sample))
    print(f"Wrote {len(sample)} prompts → {args.output}")
    print(f"Strata: {dict(strat)}")
    print(f"Unique docs covered: {docs}")


if __name__ == "__main__":
    main()
