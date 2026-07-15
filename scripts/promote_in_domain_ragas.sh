#!/bin/bash
# Promote in-domain RAGAS to canonical files. Two-stage filter:
#
#   stage 1 — LLM quality filter (already computed by filter_in_domain_ragas.sh):
#             keep rows where clear_subject AND derivable.
#             (Strict 3-of-3 with single_aspect is too aggressive — the
#              LLM judge calibrates for ELOQ single-fact Qs and rejects
#              ~90% of RAGAS-style holistic policy Qs. Annotator-B's
#              human-judged RAGAS sample passed `specific` at 82%.)
#             The `derivable` check uses RAGAS's `reference_contexts`
#             (the chunks RAGAS itself picked when synthesizing the Q),
#             which already enforces "answer is grounded in corpus."
#   stage 2 — Regex compound filter: catches obvious "X, and what Y"
#             multi-topic Qs that the LLM judge let through.
#
# No separate retriever check stage — in-domain RAGAS is in-scope by
# construction (RAGAS picks chunks first, synthesizes Q from them) and
# stage 1's `derivable` already verifies LLM-judged answerability.
# (A retriever check stage was added then removed — the gold-doc-ID
# match is not meaningful for in-domain; if you need to verify Apertus's
# runtime retriever surfaces the gold chunks, that's a separate concern
# tied to the scorer wiring, not the dataset filter.)
#
# Per-row flags are preserved so a colleague can re-filter with tighter
# rules (e.g., quality_pass_strict = all three LLM filters including
# single_aspect).
#
# Outputs:
#   data/in-domain-ragas-final.jsonl     — the canonical eval set
#   data/in-domain-ragas-sample.jsonl    — N-row random sample (default N=30)
#   annotation/annotator-{a,b}/in-domain-ragas/viewer.html
#
# Usage:
#   bash scripts/promote_in_domain_ragas.sh [SAMPLE_SIZE]
# Default SAMPLE_SIZE = 30

set -euo pipefail
cd "$(dirname "$0")/.."

N_SAMPLE="${1:-30}"

CK="data/in-domain/ragas/_quality_checkpoint.jsonl"
if [ ! -s "$CK" ]; then
  echo "FAIL: $CK missing. Run scripts/filter_in_domain_ragas.sh first."
  exit 1
fi

echo "==> Promote in-domain RAGAS"
echo "    sample size = $N_SAMPLE"
echo

PYTHONPATH=src N_SAMPLE="$N_SAMPLE" python - <<'PY'
import json, os, re, random
from pathlib import Path

N_SAMPLE = int(os.environ['N_SAMPLE'])
CK = Path('data/in-domain/ragas/_quality_checkpoint.jsonl')
FINAL = Path('data/in-domain-ragas-final.jsonl')
SAMPLE = Path('data/in-domain-ragas-sample.jsonl')

rows = [json.loads(l) for l in CK.open()]

# stage 1: LLM filter — clear_subject AND derivable
def llm_pass(r):
    return r.get('quality_clear_subject') is True and r.get('quality_derivable') is True

# stage 2: regex compound filter (catches what single_aspect missed)
COMPOUND_RE = re.compile(
    r"\b("
    r",\s+and\s+(what|how|why|which|are|is|do|did|does|the)\b"
    r"|\sand\s+(what|how|why|which)\s+(role|impact|specific|the|are|is|do|did)\b"
    r"|\bin\s+relation\s+to\b"
    r")",
    re.IGNORECASE,
)
def is_compound(q):
    return bool(COMPOUND_RE.search(q))

# Tag every row with the three flags. Canonical kept set passes both stages.
passed = []
for r in rows:
    r['quality_pass'] = llm_pass(r)
    r['quality_pass_strict'] = bool(
        r.get('quality_single_aspect') and r.get('quality_clear_subject') and r.get('quality_derivable')
    )
    r['quality_pass_no_compound'] = not is_compound(r.get('text', ''))
    if r['quality_pass'] and r['quality_pass_no_compound']:
        passed.append(r)

FINAL.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in passed) + '\n')
print(f"  canonical: {len(passed)}/{len(rows)} -> {FINAL}")

# Quality metrics on the kept set (heuristic, free)
def has_named(q):
    return bool(re.search(r"\$\s*[\d,]+|\b(19|20)\d{2}\b|\b[A-Z]{3,}\b|(?<=\w\s)[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4}", q))

n_compound_kept = sum(1 for r in passed if is_compound(r['text']))
n_named = sum(1 for r in passed if has_named(r['text']))
n_strict = sum(1 for r in passed if r['quality_pass_strict'])
avg_len = sum(len(r['text'].split()) for r in passed) / max(1, len(passed))
print(f"    compound (after regex):     {n_compound_kept}/{len(passed)}")
print(f"    named entity (heuristic):   {n_named}/{len(passed)} ({100*n_named/max(1,len(passed)):.0f}%)")
print(f"    also strict-pass (all 3):   {n_strict}/{len(passed)}")
print(f"    avg length:                 {avg_len:.1f} words")

# Sample (random, fixed seed for reproducibility)
random.Random(42).shuffle(passed)
sample = passed[:N_SAMPLE]
SAMPLE.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in sample) + '\n')
print(f"  sample:    {len(sample)} (seed=42) -> {SAMPLE}")
PY

echo
echo "==> Build annotation viewers"
bash scripts/build_in_domain_ragas_viewers.sh data/in-domain-ragas-sample.jsonl

echo
echo "==> Done. Next:"
echo "   - review data/in-domain-ragas-final.jsonl"
echo "   - open annotation/annotator-{a,b}/in-domain-ragas/viewer.html"
echo "   - update data/FINAL_DATASETS.md with the new row count"
