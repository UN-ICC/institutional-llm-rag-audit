#!/bin/bash
# Apply the in-domain RAGAS quality filter to every row in prompts.jsonl.
# Three LLM-judge filters (single_aspect / clear_subject / derivable).
# Idempotent: per-row checkpoint at _quality_checkpoint.jsonl, so
# re-running picks up new rows added by a fresh regen batch.
#
# Usage:
#   bash scripts/filter_in_domain_ragas.sh [JUDGE_MODEL]
#
# Default JUDGE_MODEL = gpt-4o (matches the generator side; cheaper alternative is gpt-4o-mini).
#
# What the three filters do:
#   single_aspect : drops compound "X and Y" multi-topic Qs
#                   (calibrated for ELOQ single-fact Qs; rejects ~90% on
#                    RAGAS — too strict to gate on alone; kept as a
#                    per-row strict-mode flag)
#   clear_subject : drops bare-reference "the project / the policy" Qs
#                   without a named entity (RAGAS already passes this
#                   well — ~90% pass)
#   derivable     : asks an LLM to extract the answer from
#                   reference_contexts alone; True iff non-empty answer.
#                   Verifies RAGAS' grounding invariant (~85% pass).
#
# Output:
#   data/in-domain/ragas/_quality_checkpoint.jsonl   — per-row labels (idempotent)
#   data/in-domain/ragas/prompts_filtered.jsonl     — rows where ALL THREE pass (the strict set)
#   data/in-domain/ragas/prompts_rejected.jsonl     — the rest
#
# IMPORTANT: scripts/promote_in_domain_ragas.sh uses _quality_checkpoint.jsonl
# (not prompts_filtered.jsonl) so it can apply the looser "clear AND derive"
# rule + regex compound filter without re-running the LLM judge.

set -euo pipefail
cd "$(dirname "$0")/.."

JUDGE="${1:-gpt-4o}"

echo "==> RAGAS in-domain quality filter"
echo "    judge model = $JUDGE"
echo

mkdir -p logs
LOG="logs/ragas_quality_$(date +%Y%m%d_%H%M%S).log"
echo "    logging to  = $LOG"
echo

PYTHONPATH=src python -u -m evalsuite.generators.in_domain.quality \
  --input  data/in-domain/ragas/prompts.jsonl \
  --output data/in-domain/ragas/prompts_filtered.jsonl \
  --rejected data/in-domain/ragas/prompts_rejected.jsonl \
  --model "$JUDGE" \
  2>&1 | tee "$LOG"

echo
echo "==> Per-criterion pass rates (from _quality_checkpoint.jsonl):"
PYTHONPATH=src python - <<'PY'
import json
rows = [json.loads(l) for l in open('data/in-domain/ragas/_quality_checkpoint.jsonl')]
ks = ['quality_single_aspect','quality_clear_subject','quality_derivable']
for k in ks:
    ys = sum(1 for r in rows if r.get(k) is True)
    ns = sum(1 for r in rows if r.get(k) is False)
    pct = 100*ys/(ys+ns) if (ys+ns) else 0
    print(f"    {k:<30} {ys}/{ys+ns} ({pct:.0f}%)")
n_strict = sum(1 for r in rows if r.get('quality_pass'))
n_clear_and_derive = sum(1 for r in rows
                         if r.get('quality_clear_subject') is True
                         and r.get('quality_derivable') is True)
print(f"    strict (all 3)                 {n_strict}/{len(rows)}")
print(f"    clear AND derive               {n_clear_and_derive}/{len(rows)}")
print()
print(f"    Next step: bash scripts/promote_in_domain_ragas.sh")
PY
