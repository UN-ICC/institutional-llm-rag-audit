#!/bin/bash
# Run ONE batch of in-domain RAGAS generation. Output appends to
# data/in-domain/ragas/{prompts,raw_<model>,_docs_used}.jsonl. Each batch
# samples from corpus docs NOT in _docs_used.jsonl, so chaining batches
# gets fresh non-overlapping docs.
#
# Usage:
#   bash scripts/regen_in_domain_ragas.sh [BATCH_ID] [MAX_DOCS] [TESTSET_SIZE]
#
# Defaults:
#   BATCH_ID      = batch_<timestamp>
#   MAX_DOCS      = 80
#   TESTSET_SIZE  = 130
#
# Provider/model are pinned: OpenAI direct + gpt-4o. This is the
# combination that succeeded on May 2 and again today (May 25). Do NOT
# use openrouter-llama for RAGAS — Parasail backend silently returns
# empty tool_calls and instructor retries forever (see ragas_llama_*.log).
#
# Crash watch: RAGAS 0.4.x has a known bug where instructor exhausts
# retries on max_tokens during "Generating Samples", returns nan into the
# scenario list, and `for sample in scenario_sample_list[i]` chokes with
# `TypeError: 'float' object is not iterable`. Smaller batches reduce
# scenario complexity and the bug rate. Today's 80/130 ratio is approx.
# 1.6 raw cases per doc — change with care.

set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_ID="${1:-batch_$(date +%Y%m%d_%H%M%S)}"
MAX_DOCS="${2:-80}"
TESTSET="${3:-130}"

echo "==> RAGAS in-domain regen"
echo "    batch_id       = $BATCH_ID"
echo "    max_docs       = $MAX_DOCS"
echo "    testset_size   = $TESTSET"
echo "    provider/model = openai / gpt-4o"
echo

mkdir -p logs
LOG="logs/ragas_regen_${BATCH_ID}.log"
echo "    logging to     = $LOG"
echo

PYTHONPATH=src python -u -m evalsuite.generators.in_domain.ragas \
  --provider openai --model gpt-4o \
  --batch-id "$BATCH_ID" \
  --max-docs "$MAX_DOCS" \
  --testset-size "$TESTSET" \
  2>&1 | tee "$LOG"

echo
echo "==> Done. data/in-domain/ragas/ now contains:"
ls -la data/in-domain/ragas/ 2>/dev/null | tail -5
