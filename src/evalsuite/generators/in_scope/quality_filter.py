"""Stage 2 of the in-scope ELOQ pipeline: binary LLM-as-judge filter.

Two checks per row, each a separate LLM call returning a binary verdict:
  - answerability — can the question be answered from the source doc?
  - standalone    — is the question's subject identifiable WITHOUT the doc?

Binary (not Likert) per LLM-judge consistency research (Zheng et al.,
NeurIPS 2023): models are inconsistent on graded scales but reproducible
on YES/NO. Two separate calls (not bundled JSON) for the same reason —
per-criterion focus is cleaner than asking the LLM to score multiple at
once.

Pipeline: generation → quality_filter → sample_for_review → human

Reads:  <output_dir>/prompts.jsonl
Writes: <output_dir>/prompts_filtered.jsonl     annotated, all rows
        <output_dir>/prompts_kept.jsonl         final: kept iff BOTH pass

Pass: keep iff answer_extractable AND standalone_check.

Run: python -m evalsuite.generators.in_scope.quality_filter
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from evalsuite._io import read_jsonl, append_jsonl, save_jsonl
from evalsuite.corpus.extract import load_corpus
from evalsuite.generators.in_scope.quality_filter_prompts import (
    check_single_aspect,
    check_standalone,
    derive_answer,
    is_extractable,
)


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("data/in-scope/eloq")
DEFAULT_CORPUS_DIR = Path("data/worldbank-zip")
# Judge defaults to gpt-4o (stronger than generator's gpt-4o-mini) to avoid
# LLM-as-judge self-preference bias. Override with --model.
DEFAULT_JUDGE_MODEL = "gpt-4o"


# ── JSONL helpers ───────────────────────────────────────────────────


# ── Pipeline ────────────────────────────────────────────────────────

def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    model: str = DEFAULT_JUDGE_MODEL,
    limit: int | None = None,
) -> None:
    input_path = output_dir / "prompts.jsonl"
    annotated_path = output_dir / "prompts_filtered.jsonl"
    kept_path = output_dir / "prompts_kept.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run in_scope.eloq first.")

    prompts = read_jsonl(input_path)
    if limit:
        prompts = prompts[:limit]
    done_ids = {r["prompt_id"] for r in read_jsonl(annotated_path)}
    todo = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"Total prompts: {len(prompts)}  done: {len(done_ids)}  todo: {len(todo)}")
    print(f"Judge model: {model}\n")

    if todo:
        corpus = load_corpus(corpus_dir)
        docs_by_id = {d["doc_id"]: d for d in corpus}

        for i, p in enumerate(todo):
            doc = docs_by_id.get(p["doc_id"])
            if not doc:
                print(f"  [{i+1}/{len(todo)}] {p['prompt_id']}: doc {p['doc_id']} not in corpus — skip")
                continue
            try:
                t0 = time.time()
                # Three binary calls — one per criterion (separate calls for
                # per-criterion focus; see Zheng 2023 on LLM-as-judge consistency).
                derived = derive_answer(p["text"], doc["text"], model)
                extractable = is_extractable(derived)
                standalone = check_standalone(p["text"], model)
                single_aspect = check_single_aspect(p["text"], model)
                kept = extractable and standalone and single_aspect

                record = {
                    **p,
                    "answer_extractable": extractable,
                    "derived_answer": derived,
                    "standalone_check": standalone,
                    "single_aspect_check": single_aspect,
                    "filter_pass": kept,
                    "filter_model": model,
                }
                append_jsonl(record, annotated_path)
                flag = "✓" if kept else "✗"
                print(f"  [{i+1}/{len(todo)}] {p['prompt_id']} "
                      f"answer={'Y' if extractable else 'N'} "
                      f"standalone={'Y' if standalone else 'N'} "
                      f"single={'Y' if single_aspect else 'N'} "
                      f"{flag} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [{i+1}/{len(todo)}] {p['prompt_id']}: ERROR {e}")
                continue

    rows = read_jsonl(annotated_path)
    kept = [r for r in rows if r.get("filter_pass")]
    save_jsonl(kept, kept_path)
    _summarize(rows, kept, annotated_path, kept_path)


def _summarize(rows: list[dict], kept: list[dict],
               annotated_path: Path, kept_path: Path) -> None:
    if not rows:
        return
    n = len(rows)
    n_extractable = sum(1 for r in rows if r.get("answer_extractable"))
    n_standalone  = sum(1 for r in rows if r.get("standalone_check"))
    n_single      = sum(1 for r in rows if r.get("single_aspect_check"))
    print(f"\n{'='*60}")
    print(f"In-scope binary filter — {n} prompts judged")
    print(f"{'='*60}")
    print(f"  answerable from doc:     {n_extractable:5d} ({n_extractable/n:.0%})")
    print(f"  standalone:              {n_standalone:5d} ({n_standalone/n:.0%})")
    print(f"  single-aspect:           {n_single:5d} ({n_single/n:.0%})")
    print(f"  KEPT (all three pass):   {len(kept):5d} ({len(kept)/n:.0%})")
    print(f"\n  annotated: {annotated_path}")
    print(f"  kept:      {kept_path}")


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: binary in-scope filter")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                   help=f"LLM judge (default: {DEFAULT_JUDGE_MODEL}). "
                        "Should be stronger than the generator (gpt-4o-mini) "
                        "to avoid LLM-as-judge self-preference bias.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.output_dir, args.corpus_dir, args.model, args.limit)


if __name__ == "__main__":
    main()
