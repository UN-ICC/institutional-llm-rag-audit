"""score_in_domain_ragas.py — Layer 1-B in-domain-ragas dataset.

Same dual-scorer pattern as score_inscope_eloq.py — RAGAS in-domain
prompts use the field name `reference_answer` (RAGAS's name) where the
ELOQ in-scope prompts use `derived_answer`; otherwise the row schema and
metric definitions are identical.

Reads a run-dir's responses.jsonl, joins each row by prompt_id to the
dataset's `reference_answer` (RAGAS-generated gold), and computes:

  - answer_similarity (RAGAS AnswerSimilarity, local sentence-transformers)
  - llm_judge_correct (Llama-3.3-70B judge, binary)
  - llm_judge_refused (Llama-3.3-70B judge, binary)

Writes <run-dir>/scores_in_domain_ragas.jsonl + summary_in_domain_ragas.json.

Summary metrics:
  mean_answer_similarity  -- coarse, embedding-based
  correctness_rate        -- LLM judge: fraction with judge_correct=True
  false_refusal_rate (FRR) -- LLM judge: fraction with judge_refused=True

References:
  RAGAS AnswerSimilarity — Es & James 2024 (arXiv:2309.15217)
  EvalSuite ports: src/evalsuite/scorers/ragas_inscope.py
                   src/evalsuite/scorers/inscope_correctness.py

Note: this dataset is dual-use (also feeds 2-B hallucination
Faithfulness). For Faithfulness, see ragas_inscope.FactualCorrectness
or wire RAGAS's Faithfulness metric against retrieved contexts — not
covered here (1-B side only).

Usage:
    PYTHONPATH=src python scripts/score_in_domain_ragas.py \\
        --run-dir results/1b-in-domain-ragas/apertus
    PYTHONPATH=src python scripts/score_in_domain_ragas.py \\
        --run-dir results/1b-in-domain-ragas/gpt5
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


DATASET = Path("data/in-domain-ragas-final.jsonl")


def _score_row(rec: dict, gold: str) -> dict:
    """Local similarity + LLM-judge correctness/refusal on one record."""
    resp = rec.get("response", "")
    out = dict(rec)
    if not gold or not resp.strip():
        out["answer_similarity"] = None
        out["llm_judge_correct"] = None
        out["llm_judge_refused"] = None
        return out
    try:
        out["answer_similarity"] = _score_similarity(resp, gold)
    except Exception as e:
        out["answer_similarity"] = None
        out["similarity_error"] = str(e)
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

    # Pre-warm the embedder singleton (avoids first-call race in workers).
    print("Pre-warming embedder...")
    _ = _score_similarity("warm-up", "warm-up")

    # Keep only rows with a matching gold reference_answer.
    scorable = [(i, r) for i, r in enumerate(rows)
                if (by_id.get(r.get("id", "")) or {}).get("reference_answer")]
    print(f"Scorable (has gold reference_answer): {len(scorable)}/{len(rows)}")

    scored: list[dict | None] = [None] * len(rows)
    for i, r in enumerate(rows):
        if not (by_id.get(r.get("id", "")) or {}).get("reference_answer"):
            scored[i] = {**r, "answer_similarity": None,
                         "llm_judge_correct": None, "llm_judge_refused": None}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_score_row, r, by_id[r["id"]]["reference_answer"]): i
                   for i, r in scorable}
        for f in tqdm(as_completed(futures), total=len(futures), desc="in-domain-ragas"):
            scored[futures[f]] = f.result()

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
        "benchmark": "in_domain_ragas",
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
            "answer_similarity": "RAGAS AnswerSimilarity (sentence-transformers cos-sim, normalised [0,1]) — Es & James 2024",
            "llm_judge": "Llama-3.3-70B (binary correct + refused per row) — project's inscope_correctness.judge_one",
        },
        "note": "Dual-use dataset — same prompts also feed 2-B faithfulness scoring (not in this script).",
    }

    scores_path = args.run_dir / "scores_in_domain_ragas.jsonl"
    summary_path = args.run_dir / "summary_in_domain_ragas.json"
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nin-domain-ragas ({model}, {args.run_dir.name}):")
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
