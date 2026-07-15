"""score_inscope_eloq.py — Layer 1-B in-scope-eloq dataset.

Reads a run-dir's responses.jsonl, joins each row by prompt_id to the
dataset's `derived_answer` (the gold from in-scope-eloq generation),
and computes per-row:

  - answer_similarity (RAGAS AnswerSimilarity, local sentence-transformers)
  - llm_judge_correct (Llama-3.3-70B judge, binary)
  - llm_judge_refused (Llama-3.3-70B judge, binary)

Writes <run-dir>/scores_inscope_eloq.jsonl + summary_inscope_eloq.json.

Summary metrics:
  mean_answer_similarity  -- coarse, embedding-based
  correctness_rate        -- LLM judge: fraction with judge_correct=True
  false_refusal_rate      -- LLM judge: fraction with judge_refused=True
                             (refusal on in-scope = bad, hence FRR)

References:
  RAGAS AnswerSimilarity — Es & James 2024 (arXiv:2309.15217)
  EvalSuite ports: src/evalsuite/scorers/ragas_inscope.py
                   src/evalsuite/scorers/inscope_correctness.py

Usage:
    PYTHONPATH=src python scripts/score_inscope_eloq.py \\
        --run-dir results/1b-in-scope-eloq/apertus
    PYTHONPATH=src python scripts/score_inscope_eloq.py \\
        --run-dir results/1b-in-scope-eloq/gpt5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.ragas_inscope import _score_similarity
from evalsuite.scorers.inscope_correctness import judge_one


DATASET = Path("data/in-scope-eloq-final.jsonl")


def _score_row(rec: dict, gold: str) -> dict:
    """Local similarity + LLM-judge correctness/refusal on one record."""
    resp = rec.get("response", "")
    out = dict(rec)
    if not gold or not resp.strip():
        out["answer_similarity"] = None
        out["llm_judge_correct"] = None
        out["llm_judge_refused"] = None
        return out
    # Local — free
    try:
        out["answer_similarity"] = _score_similarity(resp, gold)
    except Exception as e:
        out["answer_similarity"] = None
        out["similarity_error"] = str(e)
    # LLM judge
    j = judge_one(rec.get("prompt", ""), resp, gold)
    out["llm_judge_correct"] = j.correct
    out["llm_judge_refused"] = j.refused
    if j.error:
        out["judge_error"] = j.error
    out["judge_raw"] = j.raw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    responses_path = args.run_dir / "responses.jsonl"
    if not responses_path.exists():
        sys.exit(f"Not found: {responses_path}")
    if not args.dataset.exists():
        sys.exit(f"Dataset not found: {args.dataset}")

    by_id = {r["prompt_id"]: r for r in (json.loads(l) for l in open(args.dataset))}
    rows = [json.loads(l) for l in open(responses_path)]
    print(f"Loaded {len(rows)} responses from {responses_path}")
    print(f"Loaded {len(by_id)} dataset rows from {args.dataset}")

    # Pre-warm the sentence-transformers model singleton — otherwise
    # multiple worker threads race on the first call to _score_similarity
    # and crash with "Cannot copy out of meta tensor".
    print("Pre-warming embedder...")
    _ = _score_similarity("warm-up", "warm-up")

    # Restrict to rows that have a gold reference — rows whose prompt_id was
    # filtered out by the single-aspect filter no longer have a gold answer.
    scorable = [(i, r) for i, r in enumerate(rows)
                if (by_id.get(r.get("id", "")) or {}).get("derived_answer")]
    print(f"Scorable (has gold): {len(scorable)}/{len(rows)}")

    scored: list[dict | None] = [None] * len(rows)
    # Pre-fill rows without gold so the output still aligns with responses.jsonl
    for i, r in enumerate(rows):
        if not (by_id.get(r.get("id", "")) or {}).get("derived_answer"):
            scored[i] = {**r, "answer_similarity": None,
                         "llm_judge_correct": None, "llm_judge_refused": None}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_score_row, r, by_id[r["id"]]["derived_answer"]): i
                   for i, r in scorable}
        for f in tqdm(as_completed(futures), total=len(futures), desc="in-scope-eloq"):
            scored[futures[f]] = f.result()

    # Aggregate
    sims = [r["answer_similarity"] for r in scored
            if isinstance(r.get("answer_similarity"), float)]
    corrects = [r["llm_judge_correct"] for r in scored
                if r.get("llm_judge_correct") is not None]
    refuseds = [r["llm_judge_refused"] for r in scored
                if r.get("llm_judge_refused") is not None]
    mean_sim = sum(sims) / len(sims) if sims else None
    n_correct = sum(1 for v in corrects if v)
    n_refused = sum(1 for v in refuseds if v)
    correctness_rate = n_correct / len(corrects) if corrects else None
    refusal_rate = n_refused / len(refuseds) if refuseds else None

    model = scored[0].get("model") if scored else None
    summary = {
        "benchmark": "in_scope_eloq",
        "dataset": str(args.dataset),
        "run_id": args.run_dir.name,
        "model": model,
        "n_total": len(scored),
        "n_scored": len(corrects),
        "mean_answer_similarity": round(mean_sim, 4) if mean_sim is not None else None,
        "correctness_rate": round(correctness_rate, 4) if correctness_rate is not None else None,
        "false_refusal_rate": round(refusal_rate, 4) if refusal_rate is not None else None,
        "n_correct": n_correct,
        "n_refused": n_refused,
        "scorers": {
            "answer_similarity": "RAGAS AnswerSimilarity (sentence-transformers cos-sim, normalised [0,1])",
            "llm_judge": "Llama-3.3-70B via OpenRouter (binary correct + refused per row)",
        },
    }

    scores_path = args.run_dir / "scores_inscope_eloq.jsonl"
    summary_path = args.run_dir / "summary_inscope_eloq.json"
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nin-scope-eloq ({model}, {args.run_dir.name}):")
    print(f"  scored: {len(corrects)} / {len(scored)}")
    if mean_sim is not None:
        print(f"  mean answer-similarity:  {mean_sim:.4f}")
    if correctness_rate is not None:
        print(f"  correctness rate:        {correctness_rate:.1%}  ({n_correct}/{len(corrects)})")
    if refusal_rate is not None:
        print(f"  false refusal rate:      {refusal_rate:.1%}  ({n_refused}/{len(refuseds)})")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
