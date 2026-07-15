#!/bin/bash
# Build the in-scope ELOQ annotation viewer for both annotators.
#
# Source preference (first match wins):
#   1. data/in-scope-eloq-sample.jsonl              (stratified sample for human IAA)
#   2. data/in-scope-eloq-final.jsonl               (canonical final, post-filter)
#   3. data/in-scope/eloq/prompts_kept.jsonl        (post-filter working copy)
#   4. data/in-scope/eloq/prompts_filtered.jsonl    (annotated, all rows incl. failures)
#   5. data/in-scope/eloq/prompts.jsonl             (raw generation output)
#
# Usage:
#   bash scripts/build_in_scope_eloq_viewers.sh
#   bash scripts/build_in_scope_eloq_viewers.sh data/in-scope/eloq/prompts.jsonl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${1:-}" ]; then
  SRC="$1"
elif [ -f "data/in-scope-eloq-sample.jsonl" ]; then
  SRC="data/in-scope-eloq-sample.jsonl"
elif [ -f "data/in-scope-eloq-final.jsonl" ]; then
  SRC="data/in-scope-eloq-final.jsonl"
elif [ -f "data/in-scope/eloq/prompts_kept.jsonl" ]; then
  SRC="data/in-scope/eloq/prompts_kept.jsonl"
elif [ -f "data/in-scope/eloq/prompts_filtered.jsonl" ]; then
  SRC="data/in-scope/eloq/prompts_filtered.jsonl"
elif [ -f "data/in-scope/eloq/prompts.jsonl" ]; then
  SRC="data/in-scope/eloq/prompts.jsonl"
else
  echo "No in-scope ELOQ dataset found. Run in_scope.eloq first."
  exit 1
fi

echo "Source: $SRC ($(wc -l < "$SRC") prompts)"

mkdir -p annotation/annotator-a/in-scope-eloq annotation/annotator-b/in-scope-eloq

python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-a/in-scope-eloq/viewer.html \
  --name in_scope_eloq_annotator_a \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric inscope_advanced

python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-b/in-scope-eloq/viewer.html \
  --name in_scope_eloq_annotator_b \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric inscope_advanced

echo
echo "Built:"
echo "  annotation/annotator-a/in-scope-eloq/viewer.html"
echo "  annotation/annotator-b/in-scope-eloq/viewer.html"
