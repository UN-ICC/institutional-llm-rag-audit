#!/bin/bash
# Build 1-A annotation viewers (HarmBench / JailbreakBench / StrongREJECT)
# for both annotators. Samples 20 stratified rows per benchmark from the
# Apertus run (default), with jailbroken/refused balance, so Cohen's κ
# has cases on both sides of the LLM judge's verdict.
#
# Usage:
#   bash scripts/build_1a_viewers.sh
#   bash scripts/build_1a_viewers.sh results/1a-jailbreak/gpt5   # alt model

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:-results/1a-jailbreak/apertus}"
SAMPLE_DIR="annotation/_1a_sample"
N_PER=20

echo "Run dir: $RUN_DIR"
PYTHONPATH=src python3 scripts/sample_1a_for_annotation.py \
    --run-dir "$RUN_DIR" --out-dir "$SAMPLE_DIR" --n-per-benchmark "$N_PER"

for BENCH in harmbench jailbreakbench strongreject; do
  SRC="$SAMPLE_DIR/sample_${BENCH}.jsonl"
  if [ ! -f "$SRC" ]; then
    echo "[skip] $SRC not found"; continue
  fi
  # Per-benchmark rubric mirrors the upstream LLM judge's decision points
  # so κ_human-vs-llm is measured on the same criteria the judge used.
  RUBRIC="${BENCH}_judge"
  for ANN in a b; do
    DST_DIR="annotation/annotator-$ANN/1a-$BENCH"
    mkdir -p "$DST_DIR"
    cp "$SRC" "$DST_DIR/sample.jsonl"
    python3 scripts/build_annotation_viewer.py \
        --input "$DST_DIR/sample.jsonl" \
        --output "$DST_DIR/viewer.html" \
        --name "1a_${BENCH}_annotator_$ANN" \
        --rubric "$RUBRIC"
    echo "  built $DST_DIR/viewer.html  (rubric: $RUBRIC)"
  done
done

echo
echo "Done. Annotators: open the viewer.html files and score each row."
