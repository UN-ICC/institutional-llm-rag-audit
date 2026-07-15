"""Stratified 1-A annotation sampler.

For each benchmark (harmbench / jailbreakbench / strongreject), reads
the per-benchmark scores file, joins with responses, and samples N rows
balanced across jailbroken-vs-not (so Cohen's κ has cases on both sides
of the LLM judge's verdict).

Writes one sample.jsonl per benchmark into a target dir. Each row
carries `id`, `prompt`, `response`, `source`, the LLM judge's verdict
fields, plus the (optional) `behavior` / `context_string` for HarmBench
and `model`.

Usage:
    PYTHONPATH=src python scripts/sample_1a_for_annotation.py \\
        --run-dir results/1a-jailbreak/apertus \\
        --out-dir annotation/_1a_sample \\
        --n-per-benchmark 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BENCHMARKS = ("harmbench", "jailbreakbench", "strongreject")


def _stratify(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Half jailbroken, half not (whichever side is shorter caps the split)."""
    yes = [r for r in rows if r.get("jailbroken")]
    no = [r for r in rows if not r.get("jailbroken")]
    half = n // 2
    take_yes = min(len(yes), half)
    take_no = min(len(no), n - take_yes)
    # If we ran out of one side, top up from the other
    if take_yes + take_no < n:
        deficit = n - (take_yes + take_no)
        if len(yes) - take_yes >= deficit:
            take_yes += deficit
        else:
            take_no += min(deficit, len(no) - take_no)
    rng.shuffle(yes); rng.shuffle(no)
    out = yes[:take_yes] + no[:take_no]
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-per-benchmark", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for bench in BENCHMARKS:
        scores_path = args.run_dir / f"scores_{bench}.jsonl"
        if not scores_path.exists():
            print(f"  [skip] {scores_path} missing")
            continue
        rows = [json.loads(line) for line in open(scores_path)]
        # rows are already filtered to this benchmark by score_*.py
        sample = _stratify(rows, args.n_per_benchmark, rng)
        yes = sum(1 for r in sample if r.get("jailbroken"))
        no = len(sample) - yes
        out_path = args.out_dir / f"sample_{bench}.jsonl"
        with open(out_path, "w") as f:
            for r in sample:
                # Keep only fields useful for annotation; drop judge_raw to keep files small
                slim = {k: v for k, v in r.items() if k != "judge_raw"}
                f.write(json.dumps(slim) + "\n")
        print(f"{bench:>16s}: pool={len(rows):4d}  sample={len(sample)}  "
              f"(jailbroken={yes} / refused={no})  → {out_path}")


if __name__ == "__main__":
    main()
