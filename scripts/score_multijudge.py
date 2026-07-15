"""score_multijudge.py — cross-judge robustness for L3 correctness + FRR.

The headline correctness and FRR numbers in §5.2 of the paper are produced
by a single LLM judge (Llama-3.3-70B). This script re-runs the same per-row
judgments with additional judge models from different vendors so the
sensitivity of the headline numbers to judge choice can be reported.

For each (judge, dataset, response-source) it emits a per-row scores file
and a per-judge summary file under results/1b-multijudge/<judge_slug>/.
A combined comparison table is also written to
results/1b-multijudge/comparison.json.

Judges default to a 3-vendor panel (Llama-3.3-70B, GPT-4o, Claude-3.5-Sonnet,
all via OpenRouter). Pass --judges to override.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/score_multijudge.py
    PYTHONPATH=src .venv/bin/python scripts/score_multijudge.py \\
        --judges meta-llama/llama-3.3-70b-instruct openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.inscope_correctness import judge_one, judge_one_v2


# Each dataset is (dataset_name, dataset_jsonl, gold_field, response_paths)
# where response_paths is {source_name: Path-to-responses.jsonl}
DATASETS = [
    ("in_scope_eloq",
     Path("data/in-scope-eloq-final.jsonl"),
     "derived_answer",
     {
         "apertus": Path("results/1b-in-scope-eloq/apertus/responses.jsonl"),
         "gpt5":    Path("results/1b-in-scope-eloq/gpt5/responses.jsonl"),
     }),
    ("in_domain_ragas",
     Path("data/in-domain-ragas-final.jsonl"),
     "reference_answer",
     {
         "apertus": Path("outputs/runs/2026-05-28_1b-in-domain-ragas/responses.jsonl"),
         "gpt5":    Path("outputs/runs/2026-05-29_1b-in-domain-ragas_gpt5/responses.jsonl"),
     }),
]

DEFAULT_JUDGES = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
]


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def _judge_row(rec: dict, gold: str, model: str, prompt_version: str) -> dict:
    resp = rec.get("response", "") or ""
    if not gold or not resp.strip():
        return {**rec, "judge_correct": None, "judge_refused": None, "judge_model": model}
    if prompt_version == "v1":
        j = judge_one(rec.get("prompt", ""), resp, gold, model=model)
        return {
            **rec,
            "judge_correct": j.correct,
            "judge_refused": j.refused,
            "judge_raw": j.raw,
            "judge_error": j.error or None,
            "judge_model": model,
            "prompt_version": "v1",
        }
    elif prompt_version == "v2":
        j = judge_one_v2(rec.get("prompt", ""), resp, gold, model=model)
        return {
            **rec,
            "judge_factual_consistent": j.factual_consistent,
            "judge_covers_key_facts": j.covers_key_facts,
            "judge_correct": j.correct,  # derived: factual ∧ covers
            "judge_refused": j.refused,
            "judge_reasoning": j.reasoning,
            "judge_raw": j.raw,
            "judge_error": j.error or None,
            "judge_model": model,
            "prompt_version": "v2",
        }
    raise ValueError(f"unknown prompt_version {prompt_version!r}")


def _run_one(judge: str, dataset_name: str, dataset_path: Path, gold_field: str,
             responses_path: Path, response_source: str, out_root: Path,
             workers: int, prompt_version: str) -> dict | None:
    by_id = {r["prompt_id"]: r for r in (json.loads(l) for l in open(dataset_path))}
    if not responses_path.exists():
        print(f"  [skip] {responses_path} missing")
        return None
    rows = [json.loads(l) for l in open(responses_path)]
    scorable = [r for r in rows if (by_id.get(r.get("id", "")) or {}).get(gold_field)]
    print(f"  [{judge} | {dataset_name} | {response_source} | prompt={prompt_version}] "
          f"scoring {len(scorable)}/{len(rows)} rows")

    scored = [None] * len(rows)
    by_index = {id(r): i for i, r in enumerate(rows)}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_judge_row, r, by_id[r["id"]][gold_field], judge, prompt_version): id(r)
                for r in scorable}
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"{_slug(judge)[:24]}|{dataset_name[:10]}|{response_source}"):
            res = f.result()
            scored[by_index[futs[f]]] = res
    # fill unscorable
    for i, r in enumerate(rows):
        if scored[i] is None:
            scored[i] = {**r, "judge_correct": None, "judge_refused": None, "judge_model": judge}

    corrects = [r["judge_correct"] for r in scored if r.get("judge_correct") is not None]
    refuseds = [r["judge_refused"] for r in scored if r.get("judge_refused") is not None]
    n_correct = sum(1 for v in corrects if v)
    n_refused = sum(1 for v in refuseds if v)
    summary = {
        "judge": judge,
        "prompt_version": prompt_version,
        "dataset": dataset_name,
        "response_source": response_source,
        "n_total": len(scored),
        "n_scored": len(corrects),
        "n_correct": n_correct,
        "n_refused": n_refused,
        "correctness_rate": round(n_correct / len(corrects), 4) if corrects else None,
        "false_refusal_rate": round(n_refused / len(refuseds), 4) if refuseds else None,
    }
    if prompt_version == "v2":
        n_fact = sum(1 for r in scored if r.get("judge_factual_consistent") is True)
        n_cov = sum(1 for r in scored if r.get("judge_covers_key_facts") is True)
        summary["factual_consistent_rate"] = round(n_fact / len(corrects), 4) if corrects else None
        summary["covers_key_facts_rate"] = round(n_cov / len(corrects), 4) if corrects else None

    out_dir = out_root / _slug(judge) / dataset_name / response_source
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.jsonl", "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                    help="OpenRouter model ids for judges (default: 3-vendor panel)")
    ap.add_argument("--out-root", type=Path, default=Path("results/1b-multijudge"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v1",
                    help="v1 = bespoke single-correct binary; "
                         "v2 = MT-Bench-style with factual_consistent ∧ covers_key_facts + CoT")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for judge in args.judges:
        for dataset_name, dataset_path, gold_field, response_paths in DATASETS:
            if not dataset_path.exists():
                print(f"[skip dataset] {dataset_path} missing")
                continue
            for src, responses_path in response_paths.items():
                s = _run_one(judge, dataset_name, dataset_path, gold_field,
                             responses_path, src, args.out_root, args.workers,
                             args.prompt_version)
                if s:
                    all_summaries.append(s)

    # Combined comparison table
    with open(args.out_root / "comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Pretty print
    print("\n=== Multi-judge comparison ===")
    print(f"{'judge':<48}  {'dataset':<18}  {'source':<8}  {'correct':>9}  {'frr':>7}")
    for s in all_summaries:
        cr = f"{s['correctness_rate']:.1%}" if s["correctness_rate"] is not None else "-"
        fr = f"{s['false_refusal_rate']:.1%}" if s["false_refusal_rate"] is not None else "-"
        print(f"{s['judge'][:48]:<48}  {s['dataset']:<18}  {s['response_source']:<8}  "
              f"{cr:>9}  {fr:>7}")
    print(f"\nWrote {args.out_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
