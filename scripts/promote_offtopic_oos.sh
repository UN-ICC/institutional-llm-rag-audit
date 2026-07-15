#!/bin/bash
# Promote off-topic OOS to canonical. No retriever check — off-topic is
# OOS by construction (questions are from TriviaQA, not the WB corpus).
#
# Pipeline:
#   data/out-of-scope/offtopic/prompts.jsonl  ← from regen_offtopic_oos.sh
#                          │
#                          ▼
#   data/out-of-scope-offtopic-final.jsonl    (canonical, full set)
#   data/out-of-scope-offtopic-sample.jsonl   (N random, seed=42)
#   annotation/annotator-{a,b}/offtopic/viewer.html
#
# Usage:
#   bash scripts/promote_offtopic_oos.sh [SAMPLE_SIZE]
# Default SAMPLE_SIZE = 30

set -euo pipefail
cd "$(dirname "$0")/.."

N_SAMPLE="${1:-30}"

SRC="data/out-of-scope/offtopic/prompts.jsonl"
if [ ! -s "$SRC" ]; then
  echo "FAIL: $SRC missing. Run scripts/regen_offtopic_oos.sh first."
  exit 1
fi

echo "==> Promote off-topic OOS"
echo "    sample size = $N_SAMPLE"
echo

# Canonical = the full sampled set (no filtering)
cp "$SRC" data/out-of-scope-offtopic-final.jsonl
N_FINAL=$(wc -l < data/out-of-scope-offtopic-final.jsonl)
echo "  canonical: $N_FINAL rows -> data/out-of-scope-offtopic-final.jsonl"

# Stratified random sample
PYTHONPATH=src N_SAMPLE="$N_SAMPLE" python - <<'PY'
import json, os, random
from pathlib import Path
N = int(os.environ['N_SAMPLE'])
rows = [json.loads(l) for l in open('data/out-of-scope-offtopic-final.jsonl')]
random.Random(42).shuffle(rows)
sample = rows[:N]
Path('data/out-of-scope-offtopic-sample.jsonl').write_text(
    '\n'.join(json.dumps(r, ensure_ascii=False) for r in sample) + '\n')
print(f"  sample:    {len(sample)} (seed=42) -> data/out-of-scope-offtopic-sample.jsonl")
PY

# Viewers — reuse oos_advanced rubric (it asks `truly_oos`, `subject_grounded`,
# `plausible`, `clear_subject`; for off-topic, truly_oos should be trivially
# YES, subject_grounded NO, plausible varies, clear_subject YES)
mkdir -p annotation/annotator-a/offtopic annotation/annotator-b/offtopic
for ANN in a b; do
  python3 scripts/build_annotation_viewer.py \
    --input  data/out-of-scope-offtopic-sample.jsonl \
    --output "annotation/annotator-$ANN/offtopic/viewer.html" \
    --name   "offtopic_annotator_$ANN" \
    --rubric oos_advanced 2>&1 | tail -2
done

echo
echo "==> Done. Final state:"
echo "    canonical: $(wc -l < data/out-of-scope-offtopic-final.jsonl) rows -> data/out-of-scope-offtopic-final.jsonl"
echo "    sample:    $(wc -l < data/out-of-scope-offtopic-sample.jsonl) rows -> data/out-of-scope-offtopic-sample.jsonl"
echo "    viewers:   annotation/annotator-{a,b}/offtopic/viewer.html"
