"""Stratified sampler for in-scope-eloq human-vs-LLM-judge annotation.

For each scored run (Apertus, GPT-5) reads scores_inscope_eloq.jsonl
(per-row LLM-judge verdicts) and joins to the dataset's derived_answer.
Stratifies the sample across the three LLM-judge outcomes so the
annotator sees balanced examples of:

  correct=True (judge says answered right)
  correct=False, refused=False (judge says wrong-attempt)
  refused=True (judge says model refused)

Writes per-bench, per-model sample.jsonl files that build_1b_inscope_viewers.sh
turns into annotation HTML for κ vs LLM-judge agreement.

Usage:
    PYTHONPATH=src python scripts/sample_inscope_eloq_for_annotation.py \\
        --apertus-dir results/1b-in-scope-eloq/apertus \\
        --gpt5-dir    results/1b-in-scope-eloq/gpt5 \\
        --dataset     data/in-scope-eloq-final.jsonl \\
        --out-dir     annotation/_inscope_eloq_sample \\
        --n-per-model 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _strat(scored: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Take ~⅓ correct, ~⅓ wrong-attempt, ~⅓ refused (or all if not enough)."""
    correct = [r for r in scored if r.get("llm_judge_correct") is True]
    refused = [r for r in scored if r.get("llm_judge_refused") is True]
    wrong   = [r for r in scored
               if r.get("llm_judge_correct") is False
               and r.get("llm_judge_refused") is False]
    third = n // 3
    take_refused = min(len(refused), max(third, 3))
    take_wrong   = min(len(wrong),   max(third, 3))
    take_correct = max(0, n - take_refused - take_wrong)
    take_correct = min(take_correct, len(correct))
    # If we ran out of refused/wrong, top up from correct
    deficit = n - (take_refused + take_wrong + take_correct)
    if deficit > 0 and len(correct) > take_correct:
        take_correct = min(take_correct + deficit, len(correct))
    rng.shuffle(correct); rng.shuffle(wrong); rng.shuffle(refused)
    picked = correct[:take_correct] + wrong[:take_wrong] + refused[:take_refused]
    rng.shuffle(picked)
    return picked, {"correct": take_correct, "wrong_attempt": take_wrong, "refused": take_refused}


def sample_one(run_dir: Path, dataset_by_id: dict, n: int, seed: int) -> list[dict]:
    scored = [json.loads(l) for l in open(run_dir / "scores_inscope_eloq.jsonl")]
    scored = [r for r in scored if r.get("llm_judge_correct") is not None]
    rng = random.Random(seed)
    picked, counts = _strat(scored, n, rng)
    # Enrich each picked row with the gold derived_answer for the annotator
    for r in picked:
        ds = dataset_by_id.get(r.get("id", ""))
        if ds:
            r["derived_answer"] = ds.get("derived_answer", "")
            r["doc_title"] = ds.get("doc_title", "")
    return picked, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus-dir", required=True, type=Path)
    ap.add_argument("--gpt5-dir",    required=True, type=Path)
    ap.add_argument("--dataset",     required=True, type=Path)
    ap.add_argument("--out-dir",     required=True, type=Path)
    ap.add_argument("--n-per-model", type=int, default=20)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_by_id = {r["prompt_id"]: r
                     for r in (json.loads(l) for l in open(args.dataset))}

    for tag, run_dir in [("apertus", args.apertus_dir), ("gpt5", args.gpt5_dir)]:
        picked, counts = sample_one(run_dir, dataset_by_id, args.n_per_model, args.seed)
        out_path = args.out_dir / f"sample_{tag}.jsonl"
        with open(out_path, "w") as f:
            for r in picked:
                # Drop the bulky judge_raw to keep file small; keep llm_judge_*
                # fields so the viewer can show the LLM judge's verdict for context.
                slim = {k: v for k, v in r.items() if k not in ("judge_raw",)}
                f.write(json.dumps(slim) + "\n")
        print(f"  {tag}: pool={sum(counts.values())}  sample={len(picked)}  "
              f"(correct={counts['correct']} / wrong={counts['wrong_attempt']} / "
              f"refused={counts['refused']})  → {out_path}")


if __name__ == "__main__":
    main()
