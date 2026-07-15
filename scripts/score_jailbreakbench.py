"""score_jailbreakbench.py — JailbreakBench classifier (Chao et al. 2024).

Scores only `source=="jailbreakbench"` rows in responses.jsonl using
the verbatim Llama3JailbreakJudge.USER_PROMPT from
jailbreakbench/classifier.py, routed through the project's
Llama-3.3-70B judge.

Writes <run-dir>/scores_jailbreakbench.jsonl + <run-dir>/summary_jailbreakbench.json.

Usage:
    PYTHONPATH=src python scripts/score_jailbreakbench.py --run-dir results/1a-jailbreak/<model>
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.jailbreakbench import score_one


def _score(rec: dict) -> dict:
    s = score_one(rec.get("prompt", ""), rec.get("response", ""))
    rec["jbb_label"] = s.label
    rec["jailbroken"] = s.jailbroken
    rec["judge_raw"] = s.raw_output
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    responses_path = args.run_dir / "responses.jsonl"
    if not responses_path.exists():
        sys.exit(f"Not found: {responses_path}")

    all_records = [json.loads(line) for line in open(responses_path)]
    records = [r for r in all_records if r.get("source") == "jailbreakbench"]
    print(f"Loaded {len(records)} jailbreakbench rows from {responses_path}")

    scored: list[dict | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_score, rec): i for i, rec in enumerate(records)}
        for f in tqdm(as_completed(futures), total=len(futures), desc="JailbreakBench"):
            scored[futures[f]] = f.result()

    n = len(scored)
    jb = sum(1 for r in scored if r and r.get("jailbroken"))
    asr = jb / n if n else 0.0
    model = (scored[0] or {}).get("model") if scored else None

    summary = {
        "benchmark": "jailbreakbench",
        "paper": "Chao et al. 2024 (arXiv:2404.01318)",
        "judge_prompt": "Llama3JailbreakJudge.USER_PROMPT (JailbreakBench/classifier.py)",
        "run_id": args.run_dir.name,
        "model": model,
        "n": n,
        "jailbroken": jb,
        "asr": round(asr, 4),
    }

    scores_path = args.run_dir / "scores_jailbreakbench.jsonl"
    summary_path = args.run_dir / "summary_jailbreakbench.json"
    with open(scores_path, "w") as f:
        for rec in scored:
            f.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nJailbreakBench  ({model}, {args.run_dir.name}):  {jb}/{n}  ASR={asr:.1%}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
