"""score_oos.py — Layer 1-B out-of-scope ELOQ dataset (borderline OOS).

Paper-faithful scorer: ports ELOQ's `check_response_for_defusion` (the
r-z-1 prompt from `eloq/prompts/rag-confusion-check.json`) verbatim. The
judge returns "yes" iff the response defused the confusion — i.e.
clarified that the question can't be answered from the source document,
OR pointed out the missing entity, OR explicitly abstained.

Headline metric:
  scope_refusal_rate (SRR) = fraction defused = fraction "is_defused == yes"

Per-row output:
  is_defused : "yes" / "no" / "unknown"
  defused    : bool (True iff is_defused == "yes")
  judge_raw  : raw judge output

Reads:
  - responses.jsonl in --run-dir
  - data/out-of-scope-eloq-final.jsonl (the 496-row canonical OOS dataset)
  - data/worldbank-zip/documents.jsonl (corpus — to fetch the source doc
                                          text the OOS prompt was built from)

Writes:
  - <run-dir>/scores_oos.jsonl       (per-row)
  - <run-dir>/summary_oos.json        (aggregate SRR + counts)

References:
  ELOQ — Yang et al. 2024 (arXiv:2410.14567)
    eloq/prompts/rag-confusion-check.json   ← prompt source
    eloq/src/utils/prompt_util.py           ← parsing logic
  EvalSuite port: src/evalsuite/scorers/eloq_defuse.py

Usage:
    PYTHONPATH=src python scripts/score_oos.py \\
        --run-dir results/1b-oos-eloq/apertus
    PYTHONPATH=src python scripts/score_oos.py \\
        --run-dir results/1b-oos-eloq/gpt5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalsuite.scorers.eloq_defuse import score_one as defuse_score_one


DATASET = Path("data/out-of-scope-eloq-final.jsonl")
CORPUS = Path("data/worldbank-zip/documents.jsonl")


def _load_corpus(corpus_path: Path) -> dict:
    """doc_id -> text. Loaded once, used to fetch source doc per prompt."""
    by_id = {}
    with corpus_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_id[r.get("doc_id")] = r.get("text") or r.get("text_raw") or ""
    return by_id


def _score_row(rec: dict, document: str, question: str) -> dict:
    resp = rec.get("response", "")
    out = dict(rec)
    if not resp.strip():
        out["is_defused"] = "unknown"
        out["defused"] = None
        out["judge_raw"] = ""
        return out
    try:
        s = defuse_score_one(document, question, resp)
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
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="documents.jsonl with doc_id+text — needed to feed "
                         "ELOQ's defusion judge the source document")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    responses_path = args.run_dir / "responses.jsonl"
    for p, label in [(responses_path, "responses"), (args.dataset, "dataset"),
                     (args.corpus, "corpus")]:
        if not p.exists():
            sys.exit(f"Not found: {p}  ({label})")

    print(f"Loading corpus from {args.corpus}...")
    corpus_text_by_doc_id = _load_corpus(args.corpus)
    print(f"  {len(corpus_text_by_doc_id)} docs in corpus")

    by_id = {r["prompt_id"]: r for r in (json.loads(l) for l in open(args.dataset))}
    rows = [json.loads(l) for l in open(responses_path)]
    print(f"Loaded {len(rows)} responses, {len(by_id)} dataset rows")

    # For each response, find its prompt's source doc_id, then fetch text.
    scorable = []
    missing_doc = 0
    for i, r in enumerate(rows):
        pid = r.get("id", "")
        prompt = by_id.get(pid)
        if not prompt:
            continue
        doc_id = prompt.get("doc_id")
        document = corpus_text_by_doc_id.get(doc_id, "")
        if not document:
            missing_doc += 1
            continue
        scorable.append((i, r, document, prompt.get("text", "")))

    print(f"Scorable (has source doc in corpus): {len(scorable)}/{len(rows)}"
          f"  (missing doc text: {missing_doc})")

    scored: list[dict | None] = [None] * len(rows)
    for i, r in enumerate(rows):
        if i not in {idx for idx, *_ in scorable}:
            scored[i] = {**r, "is_defused": "unknown", "defused": None}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_score_row, r, doc, q): i
                   for i, r, doc, q in scorable}
        for f in tqdm(as_completed(futures), total=len(futures), desc="oos-eloq"):
            scored[futures[f]] = f.result()

    defuseds = [r["defused"] for r in scored if r.get("defused") is not None]
    n_defused = sum(1 for v in defuseds if v)
    srr = n_defused / len(defuseds) if defuseds else None

    model = scored[0].get("model") if scored else None
    summary = {
        "benchmark": "oos_eloq",
        "dataset": str(args.dataset),
        "run_id": args.run_dir.name,
        "model": model,
        "n_total": len(scored),
        "n_scored": len(defuseds),
        "scope_refusal_rate": round(srr, 4) if srr is not None else None,
        "n_defused": n_defused,
        "scorer": "ELOQ check_response_for_defusion (r-z-1 from rag-confusion-check.json) — Yang et al. 2024",
        "judge": "Llama-3.3-70B-Instruct",
    }

    scores_path = args.run_dir / "scores_oos.jsonl"
    summary_path = args.run_dir / "summary_oos.json"
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\noos-eloq ({model}, {args.run_dir.name}):")
    print(f"  scored: {len(defuseds)} / {len(scored)}")
    if srr is not None:
        print(f"  scope refusal rate (SRR): {srr:.1%}  ({n_defused}/{len(defuseds)})")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
