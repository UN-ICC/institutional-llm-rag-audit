#!/bin/bash
# Sample N off-topic OOS prompts from an official benchmark (TriviaQA
# by default). No synthetic generation. Lightweight anti-WB keyword
# filter ensures questions don't accidentally touch WB-domain content.
#
# Output: data/out-of-scope/offtopic/prompts.jsonl  (200 by default)
#
# Usage:
#   bash scripts/regen_offtopic_oos.sh [N] [SEED]
# Defaults: N=200, SEED=42

set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-200}"
SEED="${2:-42}"

echo "==> Off-topic OOS sampler"
echo "    benchmark = TriviaQA (mandarjoshi/trivia_qa / rc.nocontext / train)"
echo "    n         = $N"
echo "    seed      = $SEED"
echo

mkdir -p logs
LOG="logs/offtopic_sample_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=src python -u -m evalsuite.generators.out_of_scope.offtopic \
  --n "$N" --seed "$SEED" 2>&1 | tee "$LOG"
