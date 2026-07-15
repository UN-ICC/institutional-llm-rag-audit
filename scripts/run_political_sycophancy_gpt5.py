"""Run political-sycophancy prompts through GPT-5 via OpenRouter.

Writes results in the same responses.jsonl format the rest of the suite
uses, so scripts/score_political_sycophancy.py can score the output
directly.

Usage:
  PYTHONPATH=src python scripts/run_political_sycophancy_gpt5.py \\
      --out results/2c-political-sycophancy/gpt5 \\
      --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

DATASET = Path("data/political-sycophancy-final.jsonl")
MODEL   = "openai/gpt-5"  # via OpenRouter


def _client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _query_one(row: dict, client: OpenAI, model: str) -> dict:
    t0 = time.time()
    prompt_text = row.get("text", "")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        err = None
    except Exception as e:
        text = ""
        err = str(e)
    out = {
        "id":       row["prompt_id"],
        "prompt":   prompt_text,
        "source":   row.get("source", "political-sycophancy-v8"),
        "category": row.get("category", "political_sycophancy"),
        "response": text,
        "processing_time": round(time.time() - t0, 2),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "model":    model.split("/")[-1],  # "gpt-5"
    }
    if err:
        out["_error"] = err
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                     help="Output directory (e.g. results/2c-political-sycophancy/gpt5)")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--model", default=MODEL,
                     help="OpenRouter model id (default: openai/gpt-5)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true",
                     help="Skip prompts that already have a response in the output file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "responses.jsonl"

    # Load dataset
    rows = [json.loads(l) for l in open(args.dataset)]
    print(f"Loaded {len(rows)} prompts from {args.dataset}")

    # Resume: skip ids already in output
    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done)} already done")

    pending = [r for r in rows if r["prompt_id"] not in done]
    print(f"To process: {len(pending)}  (model={args.model}, workers={args.workers})")

    client = _client()
    mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, mode) as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_query_one, r, client, args.model): r["prompt_id"]
                   for r in pending}
        for fut in tqdm(as_completed(futures), total=len(futures),
                         desc=args.model.split("/")[-1]):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": futures[fut], "_error": str(e)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    n_total = len(rows)
    n_done_now = sum(1 for _ in open(out_path))
    print(f"\nWrote {n_done_now}/{n_total} responses → {out_path}")


if __name__ == "__main__":
    main()
