"""Combine multiple retriever_validation runs into the final cross-verified output.

Each call to `retriever_validation.py` produces one
`prompts_with_retriever__<embedder_slug>.jsonl` file. Run that script once
per embedder, then call this combine step to produce:

  - `prompts_with_retriever_combined.jsonl` — every prompt with its judgement
    from EACH embedder run, plus a final `retriever_pass` field
  - `prompts_validated.jsonl` — final kept set (subset where retriever_pass)

Pass criterion (strict AND): a prompt passes only if EVERY embedder run
judged it "not derivable" from the corpus. This is more rigorous than
CRUMQs Step IV (one embedder); per Bettina's cross-verification suggestion.

Run:
  # Two-embedder cross-verification (recommended)
  python -m evalsuite.generators.out_of_scope.retriever_validation \\
    --embed-model BAAI/bge-large-en-v1.5
  python -m evalsuite.generators.out_of_scope.retriever_validation \\
    --embed-model intfloat/e5-large-v2
  python -m evalsuite.generators.out_of_scope.combine_retriever_runs

  # With explicit input files:
  python -m evalsuite.generators.out_of_scope.combine_retriever_runs \\
    --inputs prompts_with_retriever__BAAI_bge_large_en_v1.5.jsonl \\
             prompts_with_retriever__intfloat_e5_large_v2.jsonl
"""

from __future__ import annotations

from evalsuite._io import read_jsonl, save_jsonl

import argparse
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/out-of-scope/eloq")


def _discover_runs(output_dir: Path) -> list[Path]:
    """Find all prompts_with_retriever__*.jsonl files in output_dir."""
    return sorted(output_dir.glob("prompts_with_retriever__*.jsonl"))


def combine(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    input_files: list[Path] | None = None,
) -> None:
    if input_files is None:
        input_files = _discover_runs(output_dir)
    if len(input_files) < 2:
        raise SystemExit(
            f"Need ≥ 2 retriever runs to cross-verify; found {len(input_files)} "
            f"in {output_dir}. Run retriever_validation.py with two different "
            "--embed-model values first."
        )

    # Per-prompt judgements from each run
    per_prompt: dict[str, dict] = {}
    per_run_counts: dict[str, int] = {}
    for path in input_files:
        rows = read_jsonl(path)
        embedder = rows[0].get("retriever_embed_model") if rows else path.stem
        per_run_counts[embedder] = len(rows)
        for r in rows:
            pid = r["prompt_id"]
            entry = per_prompt.setdefault(pid, {
                "prompt_id": pid,
                **{k: r[k] for k in (
                    "text", "doc_id", "doc_title", "claim_index",
                    "original_claim", "hallucinated_claim",
                    "source", "category", "expected_behavior",
                    "generator_version", "generator_model",
                ) if k in r},
                "per_embedder": {},
            })
            entry["per_embedder"][embedder] = {
                "derivable_judgement": r.get("derivable_judgement"),
                "top_k_doc_ids": r.get("retriever_top_k_doc_ids", []),
                "top_k_scores": r.get("retriever_top_k_scores", []),
                "judge_model": r.get("retriever_judge_model"),
                "top_k": r.get("retriever_top_k"),
            }

    print("Per-run prompt counts:")
    for e, n in per_run_counts.items():
        print(f"  {e:50s}  {n}")

    # Compute AND criterion + final pass
    combined: list[dict] = []
    n_judged_by_all = 0
    for pid, entry in per_prompt.items():
        per_emb = entry["per_embedder"]
        # Only count prompts that were judged by EVERY run
        if any(emb not in per_emb for emb in per_run_counts):
            continue
        n_judged_by_all += 1
        all_not_derivable = all(
            per_emb[emb]["derivable_judgement"] is False
            for emb in per_run_counts
        )
        any_derivable = any(
            per_emb[emb]["derivable_judgement"] is True
            for emb in per_run_counts
        )
        entry["retriever_pass"] = all_not_derivable
        entry["retriever_any_derivable"] = any_derivable
        entry["embedders_used"] = sorted(per_emb.keys())
        combined.append(entry)

    annotated_path = output_dir / "prompts_with_retriever_combined.jsonl"
    validated_path = output_dir / "prompts_validated.jsonl"
    save_jsonl(combined, annotated_path)
    validated = [r for r in combined if r.get("retriever_pass")]
    save_jsonl(validated, validated_path)

    n = len(combined)
    n_kept = len(validated)
    n_either_leak = sum(1 for r in combined if r.get("retriever_any_derivable"))
    print(f"\n{'='*60}")
    print(f"Cross-verified retriever validation — {n} prompts (judged by all runs)")
    print(f"{'='*60}")
    print(f"  embedders combined ({len(per_run_counts)}):")
    for e in per_run_counts:
        print(f"    - {e}")
    print(f"  derivable per ANY embedder (LEAK; dropped): {n_either_leak:4d} ({n_either_leak/n:.0%})")
    print(f"  unanswerable per ALL (KEPT):                 {n_kept:4d} ({n_kept/n:.0%})")
    print(f"\n  combined annotated: {annotated_path}")
    print(f"  validated (kept):   {validated_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combine multiple retriever-validation runs into the final cross-verified output")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--inputs", type=Path, nargs="*", default=None,
                   help="Explicit list of prompts_with_retriever__*.jsonl files. "
                        "If omitted, all such files in --output-dir are combined.")
    args = p.parse_args()
    combine(args.output_dir, args.inputs)


if __name__ == "__main__":
    main()
