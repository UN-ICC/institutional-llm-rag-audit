#!/bin/bash
# Build the OOS annotation viewer for both annotators.
#
# Each annotator gets the SAME source dataset but exports their labels
# independently. Run this once the OOS pipeline output exists.
#
# Source preference (first match wins):
#   1. data/out-of-scope-eloq-sample.jsonl             (stratified sample for human IAA)
#   2. data/out-of-scope-eloq-final.jsonl              (canonical final, post-retriever-validation)
#   3. data/out-of-scope/eloq/prompts_validated.jsonl  (cross-verified, multi-embedder)
#   4. data/out-of-scope/eloq/prompts.jsonl            (post-generation)
#
# Usage:
#   bash scripts/build_oos_viewers.sh
#   bash scripts/build_oos_viewers.sh data/out-of-scope/eloq/review_sample.jsonl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Resolve source file
if [ -n "${1:-}" ]; then
  SRC="$1"
elif [ -f "data/out-of-scope-eloq-sample.jsonl" ]; then
  SRC="data/out-of-scope-eloq-sample.jsonl"
elif [ -f "data/out-of-scope-eloq-final.jsonl" ]; then
  SRC="data/out-of-scope-eloq-final.jsonl"
elif [ -f "data/out-of-scope/eloq/prompts_validated.jsonl" ]; then
  SRC="data/out-of-scope/eloq/prompts_validated.jsonl"
elif [ -f "data/out-of-scope/eloq/prompts.jsonl" ]; then
  SRC="data/out-of-scope/eloq/prompts.jsonl"
else
  echo "No OOS dataset found. Run out_of_scope.eloq first."
  exit 1
fi

echo "Source dataset: $SRC ($(wc -l < "$SRC") prompts)"

# Build viewer for annotator-a
python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-a/oos/viewer.html \
  --name out_of_scope_annotator_a \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric oos_advanced

# Build viewer for annotator-b (same source, different localStorage key)
python scripts/build_annotation_viewer.py \
  --input "$SRC" \
  --output annotation/annotator-b/oos/viewer.html \
  --name out_of_scope_annotator_b \
  --corpus data/worldbank-zip/documents.jsonl \
  --rubric oos_advanced

echo
echo "Built:"
echo "  annotation/annotator-a/oos/viewer.html  ← annotator A opens this in a browser"
echo "  annotation/annotator-b/oos/viewer.html  ← annotator B opens this"
echo
echo "Each annotator's labels export to a JSONL alongside the viewer; commit yours."
