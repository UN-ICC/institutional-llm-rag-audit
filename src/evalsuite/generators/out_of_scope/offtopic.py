"""Off-topic OOS generator — borrows questions from official QA benchmarks.

Complements out_of_scope/eloq.py (which produces BORDERLINE OOS questions
that are topic-relevant but factually missing from the corpus) with
GENUINELY OFF-TOPIC questions — about subjects with zero overlap to the
World Bank policy domain (sports, pop culture, science trivia, etc.).

A good RAG should refuse both, for different reasons:
  - Borderline OOS: retrieves topic-related doc but specific fact isn't there
  - Off-topic OOS:  retrieves nothing relevant (or wholly unrelated chunks)

Source: TriviaQA (Joshi et al. 2017, arXiv:1705.03551), `rc.nocontext`
split. Chosen because:
  - well-cited, published official benchmark (per user constraint:
    "use only official benchmarks do not generate")
  - distribution skews to general-knowledge trivia (pop culture, sports,
    science, history) — minimal WB-policy overlap
  - Q + answer pairs already extracted; clean schema
  - permissive license (data sourced from trivia sites + Wikipedia)

Method:
  1. Stream rows from the `mandarjoshi/trivia_qa` HF dataset
  2. Light anti-WB keyword filter: drop any Q mentioning World Bank /
     IDA / IBRD / IFC / Bretton Woods / development finance / etc.
     (these are vanishingly rare in TriviaQA but the filter is cheap)
  3. Sample N (default 200) with a fixed seed
  4. Reshape into our standard OOS schema

Output: data/out-of-scope/offtopic/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/out-of-scope/offtopic")
DEFAULT_N = 200
DEFAULT_DATASET = "mandarjoshi/trivia_qa"
DEFAULT_CONFIG = "rc.nocontext"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42

# Keywords that, if present in a question, suggest non-trivial WB overlap.
# Intentionally generous — drop anything that even mentions these so the
# off-topic half is unambiguously off-topic.
WB_KEYWORDS = (
    "world bank", "ibrd", "ida", "ifc", "miga", "icsid",
    "bretton woods", "imf", "international monetary fund",
    "development bank", "development finance", "sustainable development",
    "millennium development", "official development assistance",
    "world development report", "doing business report",
    "world trade organization", "wto", "g20", "g-20",
    "g7", "g-7", "g8", "g-8", "oecd",
    "climate finance", "green climate fund",
)
_WB_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in WB_KEYWORDS) + r")\b",
                    re.IGNORECASE)


def looks_wb_adjacent(q: str) -> bool:
    return bool(_WB_RE.search(q or ""))


def fetch_offtopic(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    dataset_name: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    candidate_pool: int = 20_000,
) -> list[dict]:
    """Sample n non-WB-adjacent rows from TriviaQA.

    candidate_pool: how many rows to stream from the source before sampling
    (TriviaQA train is ~95K; pulling 20k as a candidate pool is fast and
     leaves plenty after the WB filter for a random sample of 200).
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name} / {config} / {split} (streaming)...")
    ds = load_dataset(dataset_name, config, split=split, streaming=True)

    pool: list[dict] = []
    n_wb_skipped = 0
    for i, row in enumerate(ds):
        if i >= candidate_pool:
            break
        q = (row.get("question") or "").strip()
        if not q:
            continue
        if looks_wb_adjacent(q):
            n_wb_skipped += 1
            continue
        ans = row.get("answer") or {}
        if isinstance(ans, dict):
            ans_str = ans.get("value") or ""
        else:
            ans_str = str(ans)
        pool.append({
            "trivia_qid":     row.get("question_id") or f"trivia_{i:06d}",
            "question_text":  q,
            "trivia_answer":  ans_str,
        })

    print(f"  pool size: {len(pool)} (after {n_wb_skipped} WB-adjacent skips)")
    if len(pool) < n:
        raise SystemExit(f"Need {n} rows but pool only has {len(pool)}")

    rng = random.Random(seed)
    rng.shuffle(pool)
    sampled = pool[:n]
    print(f"  sampled: {n} (seed={seed})")
    return sampled


def to_oos_schema(raw: list[dict], dataset_label: str) -> list[dict]:
    """Reshape into our standard OOS row schema."""
    out: list[dict] = []
    for i, r in enumerate(raw):
        out.append({
            "prompt_id":          f"offtopic_{i:04d}",
            "text":               r["question_text"],
            "source":             dataset_label,
            "category":           "off_topic",
            "expected_behavior":  "refuse",
            "doc_id":             None,      # not from corpus
            "doc_title":          None,
            "oos_type":           "offtopic", # vs "borderline" for ELOQ-OOS
            "trivia_qid":         r["trivia_qid"],
            "trivia_answer":      r["trivia_answer"],
            "generator_version":  "offtopic_v1",
            "generator_model":    f"benchmark:{dataset_label}",
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Off-topic OOS sampler from TriviaQA (official benchmark)"
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "prompts.jsonl"

    raw = fetch_offtopic(args.n, args.seed, args.dataset, args.config, args.split)
    rows = to_oos_schema(raw, dataset_label=args.dataset.split("/")[-1])

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
