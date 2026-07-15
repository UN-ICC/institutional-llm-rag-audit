#!/usr/bin/env bash
# Usage: bash scripts/ragas_progress.sh
# Prints a one-line-per-stage summary of every RAGAS log in logs/ragas_*.log

set -u
LOGS_DIR="$(dirname "$0")/../logs"

shopt -s nullglob
logs=("$LOGS_DIR"/ragas_*.log)
if [ ${#logs[@]} -eq 0 ]; then
  echo "No ragas_*.log files found in $LOGS_DIR"
  exit 0
fi

for log in "${logs[@]}"; do
  name=$(basename "$log" .log)
  size=$(wc -c < "$log" | tr -d ' ')
  mtime=$(stat -f "%Sm" "$log")
  errors=$(grep -c "OutputParserException" "$log" 2>/dev/null || echo 0)

  # Latest progress line per stage (parse tqdm output)
  latest=$(tr '\r' '\n' < "$log" | grep -E "^Applying|Generating" | tail -1)

  echo "─── $name ────────────────────────────────────────"
  echo "   mtime    : $mtime"
  echo "   size     : ${size} bytes"
  echo "   errors   : $errors parser failures"
  echo "   current  : ${latest:-(no stage started yet)}"
  echo
done

echo "─── Output files ────────────────────────────────────────"
BENCH_DIR="$(cd "$LOGS_DIR/.." && pwd)/data/benchmarks"
shopt -s nullglob
outs=("$BENCH_DIR"/2b_*.csv "$BENCH_DIR"/2b_*.jsonl)
if [ ${#outs[@]} -eq 0 ]; then
  echo "   (no 2b_* datasets written yet)"
else
  for f in "${outs[@]}"; do
    printf "   %s  (%s bytes)\n" "$f" "$(wc -c < "$f" | tr -d ' ')"
  done
fi
