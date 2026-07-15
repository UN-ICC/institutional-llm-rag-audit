"""Quality filters for in-domain (RAGAS) prompts.

Three LLM-judge filters applied to RAGAS TestsetGenerator output. The
RAGAS pipeline guarantees the answer is grounded in indexed chunks, but
it does NOT guarantee the question is well-formed in isolation — that's
where the 22% fail rate from annotator-B's review came from. These
filters enforce the question-quality side of the contract post-hoc.

Filters (each row must pass all three to be kept):
  1. single_aspect — Q targets exactly one fact/decision/number/event.
                     Drops compound 'X and Y' multi-topic questions.
                     (RAGAS multi-hop synthesizers love these.)
  2. clear_subject — subject identifiable from the question text alone.
                     Drops 'the policy', 'the project' references with
                     no named entity. (Same failure mode as OOS-eloq v1.)
  3. derivability — given ONLY the reference_contexts, an LLM can
                    extract a non-empty answer (no NO_ANSWER sentinel).
                    Verifies RAGAS' grounding invariant actually holds.

All three reuse the LLM-judge prompts from
`in_scope/quality_filter_prompts.py` (same checks proved out on ELOQ
in-scope — 438→189 row drop with single-aspect alone). Runs are
checkpointed: re-running picks up where the last invocation left off.

Run:
  PYTHONPATH=src python -m evalsuite.generators.in_domain.quality \\
      --input data/in-domain/ragas/prompts.jsonl \\
      --output data/in-domain/ragas/prompts_filtered.jsonl \\
      --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalsuite._io import read_jsonl, save_jsonl, append_jsonl
from evalsuite.generators._common import DEFAULT_MODEL
from evalsuite.generators.in_scope.quality_filter_prompts import (
    check_single_aspect,
    check_standalone,
    derive_answer,
    is_extractable,
)


def _contexts_text(row: dict) -> str:
    """Concatenate reference_contexts into one document string for the
    derivability judge. RAGAS stores them as a list of chunk strings."""
    ctxs = row.get("reference_contexts") or []
    if isinstance(ctxs, str):
        return ctxs
    return "\n\n---\n\n".join(c if isinstance(c, str) else str(c) for c in ctxs)


def filter_row(row: dict, model: str = DEFAULT_MODEL) -> dict:
    """Apply all three filters; mutate and return row with `quality_*` fields."""
    q = row.get("text") or row.get("question") or ""

    pass_single = check_single_aspect(q, model=model)
    pass_clear = check_standalone(q, model=model)

    ctx = _contexts_text(row)
    if ctx:
        derived = derive_answer(q, ctx, model=model)
        pass_derive = is_extractable(derived)
    else:
        derived = ""
        pass_derive = False  # if no contexts, can't verify groundedness

    row["quality_single_aspect"] = pass_single
    row["quality_clear_subject"] = pass_clear
    row["quality_derivable"] = pass_derive
    row["quality_derived_answer"] = derived
    row["quality_pass"] = bool(pass_single and pass_clear and pass_derive)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="In-domain RAGAS quality filters")
    p.add_argument("--input", type=Path,
                   default=Path("data/in-domain/ragas/prompts.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("data/in-domain/ragas/prompts_filtered.jsonl"))
    p.add_argument("--rejected", type=Path,
                   default=Path("data/in-domain/ragas/prompts_rejected.jsonl"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("data/in-domain/ragas/_quality_checkpoint.jsonl"))
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"LLM-judge model (default: {DEFAULT_MODEL})")
    args = p.parse_args()

    rows = read_jsonl(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    done_ids = {r["prompt_id"] for r in read_jsonl(args.checkpoint)}
    if done_ids:
        print(f"Checkpoint: {len(done_ids)} rows already filtered (resuming)")

    todo = [r for r in rows if r.get("prompt_id") not in done_ids]
    print(f"Remaining: {len(todo)}; model={args.model}\n")

    for i, row in enumerate(todo):
        try:
            row = filter_row(row, args.model)
        except Exception as e:
            row["quality_error"] = str(e)
            row["quality_pass"] = False
        append_jsonl(row, args.checkpoint)
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            marks = "".join("." if r.get("quality_pass") else "x"
                            for r in [row])
            print(f"  [{i+1}/{len(todo)}] last={marks} q_pass={row.get('quality_pass')}")

    # Reload everything from checkpoint (includes prior-run rows + this run's)
    all_filtered = read_jsonl(args.checkpoint)
    passed = [r for r in all_filtered if r.get("quality_pass")]
    rejected = [r for r in all_filtered if not r.get("quality_pass")]

    save_jsonl(passed, args.output)
    save_jsonl(rejected, args.rejected)

    n = len(all_filtered)
    if n:
        # Per-filter pass rates so we can see which filter is doing the work
        n_sa = sum(1 for r in all_filtered if r.get("quality_single_aspect"))
        n_cs = sum(1 for r in all_filtered if r.get("quality_clear_subject"))
        n_dv = sum(1 for r in all_filtered if r.get("quality_derivable"))
        print(f"\nFilter pass rates:")
        print(f"  single_aspect : {n_sa}/{n} ({n_sa/n*100:.0f}%)")
        print(f"  clear_subject : {n_cs}/{n} ({n_cs/n*100:.0f}%)")
        print(f"  derivable     : {n_dv}/{n} ({n_dv/n*100:.0f}%)")
        print(f"  ALL THREE     : {len(passed)}/{n} ({len(passed)/n*100:.0f}%)")
    print(f"\nWrote:")
    print(f"  passed   ({len(passed):>4} rows): {args.output}")
    print(f"  rejected ({len(rejected):>4} rows): {args.rejected}")


if __name__ == "__main__":
    main()
