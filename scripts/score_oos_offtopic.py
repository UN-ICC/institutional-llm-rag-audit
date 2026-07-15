"""score_oos_offtopic.py — Layer 1-B off-topic OOS dataset (TriviaQA).

Same defusion judge as score_oos.py (ELOQ's r-z-1 rag-confusion-check
prompt, Yang et al. 2024) so SRR is comparable across the two OOS
halves (borderline + off-topic).

Grounding caveat: off-topic prompts are sampled from TriviaQA, not
ELOQ-generated, so there's no "source document" the question was
constructed against. We feed an empty document to ELOQ's judge —
the defusion judge still answers "did the response defuse / refuse
to engage" correctly because its question is about the model's
*response*, not about whether the response matches the document.

For documentation: the scorer's behaviour with an empty document is
to ask "is the response refusing the off-topic question?" — exactly
the SRR signal we want, just without the ELOQ paper's RAG-context
framing. If you want strict paper purity, this scorer is best
characterised as ELOQ's defusion judge **adapted** for non-ELOQ
off-topic OOS (documented deviation).

Reads:
  - responses.jsonl in --run-dir
  - data/out-of-scope-offtopic-final.jsonl (the 200-row TriviaQA OOS dataset)

Writes:
  - <run-dir>/scores_oos_offtopic.jsonl       (per-row)
  - <run-dir>/summary_oos_offtopic.json        (aggregate SRR + counts)

References:
  ELOQ — Yang et al. 2024 (arXiv:2410.14567) — defusion-judge prompt
  TriviaQA — Joshi et al. 2017 (arXiv:1705.03551) — source dataset
  EvalSuite port: src/evalsuite/scorers/eloq_defuse.py

Usage:
    PYTHONPATH=src python scripts/score_oos_offtopic.py \\
        --run-dir results/1b-oos-offtopic/apertus
    PYTHONPATH=src python scripts/score_oos_offtopic.py \\
        --run-dir results/1b-oos-offtopic/gpt5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.eloq_defuse import score_one as defuse_score_one


DATASET = Path("data/out-of-scope-offtopic-final.jsonl")


def _score_row(rec: dict, question: str) -> dict:
    resp = rec.get("response", "")
    out = dict(rec)
    if not resp.strip():
        out["is_defused"] = "unknown"
        out["defused"] = None
        out["judge_raw"] = ""
        return out
    try:
        # Off-topic has no source WB doc — empty document is intentional.
        # See module docstring for the deviation rationale.
        s = defuse_score_one("", question, resp)
        out["is_defused"] = s.is_defused
        out["defused"] = s.defused
        out["judge_raw"] = s.raw_output
    except Exception as e:
        out["is_defused"] = "unknown"
        out["defused"] = None
        out["judge_error"] = str(e)
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
    print(f"Loaded {len(rows)} responses, {len(by_id)} dataset rows")

    scorable = []
    for i, r in enumerate(rows):
        prompt = by_id.get(r.get("id", ""))
        if not prompt:
            continue
        scorable.append((i, r, prompt.get("text", "")))
    print(f"Scorable: {len(scorable)}/{len(rows)}")

    scored: list[dict | None] = [None] * len(rows)
    matched_idx = {i for i, *_ in scorable}
    for i, r in enumerate(rows):
        if i not in matched_idx:
            scored[i] = {**r, "is_defused": "unknown", "defused": None}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_score_row, r, q): i for i, r, q in scorable}
        for f in tqdm(as_completed(futures), total=len(futures), desc="oos-offtopic"):
            scored[futures[f]] = f.result()

    defuseds = [r["defused"] for r in scored if r.get("defused") is not None]
    n_defused = sum(1 for v in defuseds if v)
    srr = n_defused / len(defuseds) if defuseds else None

    model = scored[0].get("model") if scored else None
    summary = {
        "benchmark": "oos_offtopic",
        "dataset": str(args.dataset),
        "run_id": args.run_dir.name,
        "model": model,
        "n_total": len(scored),
        "n_scored": len(defuseds),
        "scope_refusal_rate": round(srr, 4) if srr is not None else None,
        "n_defused": n_defused,
        "scorer": ("ELOQ defusion judge (r-z-1, Yang et al. 2024) adapted for "
                   "off-topic OOS — empty source document, prompt unchanged"),
        "judge": "Llama-3.3-70B-Instruct",
        "note": ("Off-topic OOS prompts (TriviaQA, Joshi et al. 2017) have no "
                 "corresponding WB source doc; document field fed empty. SRR is "
                 "directly comparable to score_oos.py output."),
    }

    scores_path = args.run_dir / "scores_oos_offtopic.jsonl"
    summary_path = args.run_dir / "summary_oos_offtopic.json"
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\noos-offtopic ({model}, {args.run_dir.name}):")
    print(f"  scored: {len(defuseds)} / {len(scored)}")
    if srr is not None:
        print(f"  scope refusal rate (SRR): {srr:.1%}  ({n_defused}/{len(defuseds)})")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
