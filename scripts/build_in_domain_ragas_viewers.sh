#!/bin/bash
# Build the in-domain RAGAS annotation viewer for both annotators.
#
# This is the unified RAGAS dataset that feeds BOTH 1-B in-scope FRR and
# 2-B hallucination Faithfulness — see src/evalsuite/generators/in_domain/
# ragas.py for the rationale.
#
# Source preference (first match wins):
#   1. data/in-domain-ragas-sample.jsonl       (stratified sample for human IAA)
#   2. data/in-domain-ragas-final.jsonl        (full canonical, post quality filter)
#   3. data/in-domain/ragas/prompts_filtered.jsonl  (working copy, post filter)
#   4. data/in-domain/ragas/prompts.jsonl      (raw RAGAS output, pre-filter)
#
# Usage:
#   bash scripts/build_in_domain_ragas_viewers.sh
#   bash scripts/build_in_domain_ragas_viewers.sh data/in-domain/ragas/prompts.jsonl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${1:-}" ]; then
  SRC="$1"
elif [ -f "data/in-domain-ragas-sample.jsonl" ]; then
  SRC="data/in-domain-ragas-sample.jsonl"
elif [ -f "data/in-domain-ragas-final.jsonl" ]; then
  SRC="data/in-domain-ragas-final.jsonl"
elif [ -f "data/in-domain/ragas/prompts_filtered.jsonl" ]; then
  SRC="data/in-domain/ragas/prompts_filtered.jsonl"
elif [ -f "data/in-domain/ragas/prompts.jsonl" ]; then
  SRC="data/in-domain/ragas/prompts.jsonl"
else
  echo "No in-domain RAGAS dataset found. Run in_domain.ragas first."
  exit 1
fi

echo "Source: $SRC ($(wc -l < "$SRC") prompts)"

mkdir -p annotation/annotator-a/in-domain-ragas annotation/annotator-b/in-domain-ragas

python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-a/in-domain-ragas/viewer.html \
  --name in_domain_ragas_annotator_a \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric inscope_advanced

python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-b/in-domain-ragas/viewer.html \
  --name in_domain_ragas_annotator_b \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric inscope_advanced

echo
echo "Built:"
echo "  annotation/annotator-a/in-domain-ragas/viewer.html"
echo "  annotation/annotator-b/in-domain-ragas/viewer.html"
