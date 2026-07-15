"""Main evaluation loop: load test cases -> query RAG -> score -> report.

Phases (typically run on different machines):
  1. Run (UNICC): send prompts to RAG endpoint, save responses
  2. Eval (us): score saved responses

Layer 1-A jailbreak rows are scored with each benchmark's canonical
open-model judge (HarmBench LLAMA2_CLS_PROMPT, JailbreakBench
Llama3JailbreakJudge, StrongREJECT rubric autograder), routed through
the project's Llama-3.3-70B endpoint. Other layers fall back to the
generic RAG refuse/comply judge in scorers/llm_judge.py.

Usage:
  python -m evalsuite.runners.run_layer --layer 1a --phase run
  python -m evalsuite.runners.run_layer --layer 1a --phase eval
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from evalsuite.clients.rag_client import RAGClient
from evalsuite.datasets.io import (
    append_jsonl,
    load_checkpoint,
    load_csv,
    save_csv,
    save_jsonl,
)
from evalsuite.datasets.loaders import (
    load_all_jailbreaking,
    load_in_scope_eloq,
    load_in_domain_ragas,
    load_oos,
    load_oos_offtopic,
    load_political_sycophancy,
)


OUTPUTS_DIR = Path("outputs/runs")


# Layer codes:
#   1a                  — jailbreaking benchmarks (HarmBench / JBB / StrongREJECT)
#   1b-in-scope-eloq    — ELOQ single-doc in-scope Qs (FRR control half)
#   1b-in-domain-ragas  — RAGAS in-domain Qs. SAME dataset is scored by 2-B
#                         hallucination Faithfulness; layer code drives which
#                         scorer runs, not which prompts go in.
#   1b-oos              — ELOQ out-of-scope Qs (borderline OOS — topic-relevant,
#                         fact missing from corpus)
#   1b-oos-offtopic     — TriviaQA off-topic Qs (gross OOS — wholly different
#                         domain). Complement to 1b-oos.
LAYER_LOADERS = {
    "1a":                       load_all_jailbreaking,
    "1b-in-scope-eloq":         load_in_scope_eloq,
    "1b-in-domain-ragas":       load_in_domain_ragas,
    "1b-oos":                   load_oos,
    "1b-oos-offtopic":          load_oos_offtopic,
    "2c-political-sycophancy":  load_political_sycophancy,
}


# ── Phase 1: run prompts through the RAG ─────────────────────────────

def run_phase(layer: str, run_id: str | None = None, sample: bool = False):
    run_id = run_id or f"{datetime.now().strftime('%Y-%m-%d')}_{layer}{'_sample' if sample else ''}"
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    responses_path = run_dir / "responses.csv"

    loader = LAYER_LOADERS.get(layer)
    if loader is None:
        print(f"Unknown layer {layer!r}. Choices: {sorted(LAYER_LOADERS)}")
        return
    cases = loader(sample=sample) if layer != "1a" else loader()
    print(f"Loaded {len(cases)} test cases for layer {layer}"
          f"{' (sample)' if sample else ''}")

    done_ids = load_checkpoint(responses_path, key="id")
    remaining = [c for c in cases if c.id not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")

    client = RAGClient()
    rows = []
    for tc in tqdm(remaining, desc=f"Layer {layer} queries"):
        result = client.run_test_case(tc)
        row = {
            "id": tc.id,
            "prompt": tc.prompt,
            "source": tc.source,
            "category": tc.category,
            "response": result.response,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp,
            "model": result.model,
        }
        if tc.metadata.get("behavior"):
            row["behavior"] = tc.metadata["behavior"]
        if tc.metadata.get("context_string"):
            row["context_string"] = tc.metadata["context_string"]
        rows.append(row)
        append_jsonl(row, run_dir / "responses.jsonl")

    if rows:
        all_rows = load_csv(responses_path) if responses_path.exists() else []
        all_rows.extend(rows)
        save_csv(all_rows, responses_path)

    print(f"\nSaved {len(rows)} new responses to {run_dir}")
    print(f"Total: {len(done_ids) + len(rows)} responses")


# ── Phase 2: score saved responses ───────────────────────────────────

def _load_records(run_dir: Path) -> list[dict] | None:
    responses_path = run_dir / "responses.jsonl"
    if not responses_path.exists():
        responses_path = run_dir / "responses.csv"
    if not responses_path.exists():
        print(f"No responses found in {run_dir}")
        return None
    if responses_path.suffix == ".csv":
        return load_csv(str(responses_path))
    from evalsuite.datasets.io import load_jsonl
    return load_jsonl(responses_path)


def _resolve_run_dir(layer: str, run_id: str | None) -> Path | None:
    """Locate the run directory for `layer`. Run-dir naming convention:
        outputs/runs/<date>_<layer>/             ← default Apertus run
        outputs/runs/<date>_<layer>_<model>/     ← non-default model (e.g. gpt5)
    Auto-pick returns the latest plain-Apertus dir; pass --run-id to score
    a non-default model run."""
    if run_id:
        return OUTPUTS_DIR / run_id
    # Match dirs ending exactly in `_<layer>` (no model suffix).
    candidates = [p for p in OUTPUTS_DIR.glob(f"*_{layer}") if p.is_dir()]
    candidates.sort(reverse=True)
    if not candidates:
        print(f"No Apertus runs found for layer {layer}. Try --run-id <dir>")
        return None
    return candidates[0]


# ── Dataset / corpus joins (for 1-B canonical scoring) ──────────────

DATA_DIR = Path("data")

LAYER_DATASET_PATHS = {
    "1b-in-scope-eloq":   DATA_DIR / "in-scope-eloq-final.jsonl",
    "1b-in-domain-ragas": DATA_DIR / "in-domain-ragas-final.jsonl",
    "1b-oos":             DATA_DIR / "out-of-scope-eloq-final.jsonl",
}
CORPUS_PATH = DATA_DIR / "worldbank-zip" / "documents.jsonl"


def _load_dataset_by_id(layer: str) -> dict[str, dict]:
    path = LAYER_DATASET_PATHS.get(layer)
    if path is None or not path.exists():
        return {}
    from evalsuite.datasets.io import load_jsonl
    return {r["prompt_id"]: r for r in load_jsonl(path)}


def _load_corpus_by_id() -> dict[str, str]:
    if not CORPUS_PATH.exists():
        return {}
    from evalsuite.datasets.io import load_jsonl
    return {r["doc_id"]: r.get("text", "") for r in load_jsonl(CORPUS_PATH)}


def _eval_layer_1a(records: list[dict]) -> None:
    """Per-benchmark canonical scoring for Layer 1-A.

    Dispatches by `source`: HarmBench / JailbreakBench / StrongREJECT.
    Each scorer writes its label + a unified `jailbroken` boolean back
    onto each record in-place.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "")].append(r)

    if hb := by_source.get("harmbench"):
        from evalsuite.scorers.harmbench import score_batch
        print(f"\n--- HarmBench classifier ({len(hb)} responses) ---")
        for rec, s in zip(hb, score_batch(hb)):
            rec["harmbench_label"] = s.label
            rec["jailbroken"] = s.jailbroken
            rec["judge_raw"] = s.raw_output

    if jbb := by_source.get("jailbreakbench"):
        from evalsuite.scorers.jailbreakbench import score_batch
        print(f"\n--- JailbreakBench judge ({len(jbb)} responses) ---")
        for rec, s in zip(jbb, score_batch(jbb)):
            rec["jbb_label"] = s.label
            rec["jailbroken"] = s.jailbroken
            rec["judge_raw"] = s.raw_output

    if sr := by_source.get("strongreject"):
        from evalsuite.scorers.strongreject import score_batch
        print(f"\n--- StrongREJECT autograder ({len(sr)} responses) ---")
        for rec, s in zip(sr, score_batch(sr)):
            rec["strongreject_refusal"] = s.refusal_score
            rec["strongreject_convincing"] = s.convincing_score
            rec["strongreject_specific"] = s.specific_score
            rec["strongreject_score"] = s.final_score
            rec["jailbroken"] = s.jailbroken


def _eval_layer_other(records: list[dict]) -> None:
    """Generic refuse/comply judging for non-1A layers."""
    from evalsuite.scorers.llm_judge import judge_batch

    print(f"\n--- RAG refuse/comply judge ({len(records)} responses) ---")
    results = judge_batch([r["prompt"] for r in records], [r["response"] for r in records])
    for rec, jr in zip(records, results):
        rec["judge_verdict"] = jr.verdict
        rec["judge_reason"] = jr.refusal_reason
        rec["severity"] = jr.severity
        rec["jailbroken"] = jr.verdict == "complied"
        if jr.error:
            rec["judge_error"] = jr.error


def _eval_layer_1b(
    records: list[dict],
    layer: str,
    *,
    ragas_factual: bool = False,
    include_context_metrics: bool = False,
) -> None:
    """Per-half canonical scoring for Layer 1-B.

    Always runs the generic refuse/comply judge (for FRR / CRR).

    In-scope halves additionally run RAGAS AnswerSimilarity vs the
    dataset's gold answer (derived_answer / reference_answer). Set
    ragas_factual=True to also run FactualCorrectness (LLM-heavy).

    OOS half additionally runs ELOQ's is_defused judge against the
    source document text.
    """
    if include_context_metrics:
        raise NotImplementedError(
            "RAGAS Faithfulness / ContextPrecision / ContextRecall need the "
            "retrieved passages, which Apertus's /chat does not return. "
            "Re-enable after UNICC adds a `contexts` field to the response payload."
        )

    # Stage 1: generic refusal judge → FRR / CRR
    _eval_layer_other(records)

    # Stage 2: join to dataset for the per-half canonical scorer
    by_id = _load_dataset_by_id(layer)
    if not by_id:
        print(f"\n[skip] No dataset found at {LAYER_DATASET_PATHS.get(layer)} — "
              "skipping canonical 1-B scorer.")
        return

    if layer == "1b-oos":
        corpus = _load_corpus_by_id()
        joined = []
        for r in records:
            ds = by_id.get(r.get("id", ""))
            doc_text = corpus.get((ds or {}).get("doc_id", ""), "") if ds else ""
            joined.append({**r, "document": doc_text})
        from evalsuite.scorers.eloq_defuse import score_batch
        print(f"\n--- ELOQ defusion judge ({len(joined)} OOS responses) ---")
        for rec, s in zip(records, score_batch(joined)):
            rec["eloq_is_defused"] = s.is_defused
            rec["eloq_defused"] = s.defused
            rec["eloq_raw"] = s.raw_output
        return

    # In-scope halves: AnswerSimilarity (and optionally FactualCorrectness)
    joined = []
    for r in records:
        ds = by_id.get(r.get("id", ""))
        ref = ""
        if ds:
            ref = ds.get("reference_answer") or ds.get("derived_answer") or ""
        joined.append({**r, "reference_answer": ref})
    from evalsuite.scorers.ragas_inscope import score_batch
    print(f"\n--- RAGAS in-scope correctness ({len(joined)} responses, "
          f"factual={ragas_factual}) ---")
    for rec, s in zip(records, score_batch(joined, factual_correctness=ragas_factual)):
        rec["answer_similarity"] = s.answer_similarity
        if ragas_factual:
            rec["factual_correctness"] = s.factual_correctness
        if s.error:
            rec["ragas_error"] = s.error


def _summarize_1a(records: list[dict], layer: str, run_id: str) -> dict:
    total = len(records)
    jailbroken = sum(1 for r in records if r.get("jailbroken"))

    per_source: dict[str, dict] = {}
    for src in sorted({r.get("source", "") for r in records}):
        src_recs = [r for r in records if r.get("source") == src]
        n = len(src_recs)
        jb = sum(1 for r in src_recs if r.get("jailbroken"))
        entry = {"n": n, "jailbroken": jb, "asr": round(jb / n, 4) if n else 0.0}
        if src == "strongreject":
            scores = [r["strongreject_score"] for r in src_recs
                      if isinstance(r.get("strongreject_score"), (int, float))
                      and r["strongreject_score"] == r["strongreject_score"]]  # drop NaN
            entry["mean_score"] = round(sum(scores) / len(scores), 4) if scores else None
            entry["scored"] = len(scores)
        per_source[src] = entry

    print(f"\n{'='*52}")
    print(f"Layer {layer} canonical-judge summary")
    print(f"{'='*52}")
    print(f"Total: {total} | jailbroken: {jailbroken} | overall ASR: {jailbroken/total:.1%}")
    for src, m in per_source.items():
        extra = f", mean SR={m['mean_score']}" if m.get("mean_score") is not None else ""
        print(f"  {src}: {m['jailbroken']}/{m['n']} ({m['asr']:.1%}){extra}")

    return {
        "layer": layer,
        "run_id": run_id,
        "total": total,
        "jailbroken": jailbroken,
        "asr": round(jailbroken / total, 4) if total else 0.0,
        "per_source": per_source,
    }


def _summarize_1b(records: list[dict], layer: str, run_id: str) -> dict:
    total = len(records)
    refused = sum(1 for r in records if r.get("judge_verdict") == "refused")
    complied = sum(1 for r in records if r.get("judge_verdict") == "complied")
    reason_scope = sum(1 for r in records if r.get("judge_reason") in ("scope", "both"))
    reason_safety = sum(1 for r in records if r.get("judge_reason") in ("safety", "both"))

    print(f"\n{'='*52}")
    print(f"Layer {layer} canonical-judge summary")
    print(f"{'='*52}")
    print(f"Total: {total} | refused: {refused} | complied: {complied}")

    summary: dict = {"layer": layer, "run_id": run_id, "total": total,
                     "refused": refused, "complied": complied,
                     "refusal_reason_scope": reason_scope,
                     "refusal_reason_safety": reason_safety}

    if layer == "1b-oos":
        # OOS: refusal IS the desired behavior → Correct Refusal Rate
        crr = refused / total if total else 0
        defused = sum(1 for r in records if r.get("eloq_defused"))
        scored = sum(1 for r in records if r.get("eloq_is_defused") in ("yes", "no"))
        defuse_rate = defused / scored if scored else 0
        print(f"Correct Refusal Rate (CRR):  {crr:.1%}")
        print(f"ELOQ defuse rate (yes/scored): {defused}/{scored} ({defuse_rate:.1%})")
        summary.update(correct_refusal_rate=round(crr, 4),
                       eloq_defused=defused, eloq_scored=scored,
                       eloq_defuse_rate=round(defuse_rate, 4))
    else:
        # in-scope: refusal is the wrong behavior → False Refusal Rate
        frr = refused / total if total else 0
        sims = [r["answer_similarity"] for r in records
                if isinstance(r.get("answer_similarity"), (int, float))]
        mean_sim = sum(sims) / len(sims) if sims else None
        facts = [r["factual_correctness"] for r in records
                 if isinstance(r.get("factual_correctness"), (int, float))]
        mean_fact = sum(facts) / len(facts) if facts else None
        print(f"False Refusal Rate (FRR):    {frr:.1%}")
        print(f"Mean answer similarity:      {mean_sim:.3f}" if mean_sim is not None else
              "Mean answer similarity:      n/a")
        if mean_fact is not None:
            print(f"Mean factual correctness:    {mean_fact:.3f}")
        summary.update(false_refusal_rate=round(frr, 4),
                       mean_answer_similarity=round(mean_sim, 4) if mean_sim is not None else None,
                       mean_factual_correctness=round(mean_fact, 4) if mean_fact is not None else None)
    return summary


def _summarize_other(records: list[dict], layer: str, run_id: str) -> dict:
    total = len(records)
    refused = sum(1 for r in records if r.get("judge_verdict") == "refused")
    complied = sum(1 for r in records if r.get("judge_verdict") == "complied")
    ambiguous = sum(1 for r in records if r.get("judge_verdict") == "ambiguous")
    asr = complied / total if total else 0

    print(f"\n{'='*52}")
    print(f"Layer {layer} judge summary")
    print(f"{'='*52}")
    print(f"Total: {total} | refused: {refused} | complied: {complied} | ambiguous: {ambiguous}")
    print(f"ASR: {asr:.1%}")

    severity_dist = {f"severity_{s}": sum(1 for r in records if r.get("severity") == s)
                     for s in range(1, 6)}
    severity_dist = {k: v for k, v in severity_dist.items() if v}

    return {
        "layer": layer,
        "run_id": run_id,
        "total": total,
        "refused": refused,
        "complied": complied,
        "ambiguous": ambiguous,
        "asr": round(asr, 4),
        **severity_dist,
    }


def eval_phase(
    layer: str,
    run_id: str | None = None,
    *,
    ragas_factual: bool = False,
    include_context_metrics: bool = False,
):
    run_dir = _resolve_run_dir(layer, run_id)
    if run_dir is None:
        return
    records = _load_records(run_dir)
    if records is None:
        return
    print(f"Loaded {len(records)} responses from {run_dir}")

    if layer == "1a":
        _eval_layer_1a(records)
        summary = _summarize_1a(records, layer, run_dir.name)
    elif layer.startswith("1b-"):
        _eval_layer_1b(records, layer,
                       ragas_factual=ragas_factual,
                       include_context_metrics=include_context_metrics)
        summary = _summarize_1b(records, layer, run_dir.name)
    else:
        _eval_layer_other(records)
        summary = _summarize_other(records, layer, run_dir.name)

    save_csv(records, run_dir / "scores.csv")
    save_jsonl(records, run_dir / "scores.jsonl")
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScores: {run_dir/'scores.csv'} / {run_dir/'scores.jsonl'}")
    print(f"Summary: {run_dir/'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run red-teaming evaluation layer")
    parser.add_argument("--layer", required=True,
                        help=f"Layer code. Choices: {sorted(LAYER_LOADERS)}")
    parser.add_argument("--phase", required=True, choices=["run", "eval"])
    parser.add_argument("--run-id", help="Run ID (default: auto/latest)")
    parser.add_argument("--sample", action="store_true",
                        help="Use the 50-row stratified annotation sample (1-B only). "
                             "Outputs go to outputs/runs/<date>_<layer>_sample/.")
    parser.add_argument("--ragas-factual", action="store_true",
                        help="1-B in-scope only: additionally compute RAGAS "
                             "FactualCorrectness (LLM-heavy, ~1 call/row).")
    parser.add_argument("--include-context-metrics", action="store_true",
                        help="1-B only: also run RAGAS Faithfulness / "
                             "ContextPrecision / ContextRecall. Requires retrieval "
                             "traces in responses.jsonl which Apertus does NOT "
                             "currently return — errors out for now.")
    args = parser.parse_args()

    if args.phase == "run":
        run_phase(args.layer, args.run_id, sample=args.sample)
    else:
        eval_phase(args.layer, args.run_id,
                   ragas_factual=args.ragas_factual,
                   include_context_metrics=args.include_context_metrics)


if __name__ == "__main__":
    main()
