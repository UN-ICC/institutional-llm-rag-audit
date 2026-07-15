"""score_oos_refusal.py — UAEval4RAG-style scope-refusal scorer for OOS.

Re-scores L2 OOS responses with a judge that sees only (question, response).
Document is not fed to the judge, since the deployed RAG retrieves its own
chunks at inference time and those chunks are not captured in the response
records.

Reads responses from the run dir, scores both borderline and off-topic in
a single pass, writes scores + summary.

Usage:
    JUDGE_MODEL=openai/gpt-4o PYTHONPATH=src .venv/bin/python \
      scripts/score_oos_refusal.py \
      --responses outputs/runs/2026-05-28_1b-oos/responses.jsonl \
      --offtopic-responses outputs/runs/2026-05-28_1b-oos-offtopic/responses.jsonl \
      --out results/1b-oos-refusal/apertus
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.scope_refusal import score_one


def _score_row(rec: dict) -> dict:
    resp = rec.get("response", "") or ""
    out = dict(rec)
    if not resp.strip():
        out["scope_refusal_label"] = "AMBIGUOUS"
        out["scope_refused"] = None
        out["judge_raw"] = ""
        return out
    s = score_one(rec.get("prompt", ""), resp)
    out["scope_refusal_label"] = s.label
    out["scope_refused"] = s.scope_refused
    out["judge_raw"] = s.raw_output
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=Path,
                    help="borderline responses.jsonl")
    ap.add_argument("--offtopic-responses", required=True, type=Path,
                    help="off-topic responses.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for half, p in [("borderline", args.responses), ("off_topic", args.offtopic_responses)]:
        for line in open(p):
            r = json.loads(line)
            r["_half"] = half
            rows.append(r)
    print(f"Loaded {len(rows)} responses")

    scored = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_score_row, r): i for i, r in enumerate(rows)}
        for f in tqdm(as_completed(futs), total=len(futs), desc="scope-refusal"):
            scored[futs[f]] = f.result()

    # Write per-row
    with open(args.out / "scores_oos_refusal.jsonl", "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    # Summary per half
    summary = {"per_half": {}}
    for half in ("borderline", "off_topic"):
        rows_h = [r for r in scored if r["_half"] == half]
        n = len(rows_h)
        refused = sum(1 for r in rows_h if r.get("scope_refused") is True)
        ambig = sum(1 for r in rows_h if r.get("scope_refused") is None)
        labels = Counter(r.get("scope_refusal_label") for r in rows_h)
        summary["per_half"][half] = {
            "n": n,
            "scope_refused": refused,
            "srr": refused / n if n else 0.0,
            "ambiguous": ambig,
            "label_dist": dict(labels),
        }
    summary["judge_prompt"] = "scope_refusal.JUDGE_PROMPT (UAEval4RAG-style, Q+R only)"
    summary["model"] = (rows[0].get("model") if rows else "unknown")

    with open(args.out / "summary_oos_refusal.json", "w") as f:
        json.dump(summary, f, indent=2)

    for half, s in summary["per_half"].items():
        print(f"  {half:<12} SRR = {s['srr']:.1%}  ({s['scope_refused']}/{s['n']})  labels={s['label_dist']}")
    print(f"Wrote {args.out / 'scores_oos_refusal.jsonl'} + summary")


if __name__ == "__main__":
    main()
