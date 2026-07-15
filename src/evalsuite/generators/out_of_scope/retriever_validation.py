"""Stage 2 of OOS pipeline: retriever validation (run ONCE per embedder).

Pipeline: generation → retriever_validation × N → combine → sample_for_review → human annotation.

This module runs ONE embedder + LLM judge per invocation and writes its
results to a per-embedder file. Run it TWICE (one per embedder) and then
use `combine_retriever_runs.py` to compute the final cross-verified pass.

Why single-embedder per invocation: lets you re-run cheaply with a third
embedder later, swap one out, or re-judge with a different LLM, without
re-doing all the work. Caching is per-embedder.

Catches OOS candidates that look OOS w.r.t. their SOURCE doc but are
answerable from SOME OTHER doc in the corpus.

## Pattern: CRUMQs Step IV — verified against the official code

CRUMQs (Liu et al., arXiv:2510.11956, https://github.com/pybeebee/CRUMQs)
Section 3 Step IV says:
  "The unanswerability of each seed question is verified as in UAEval4RAG:
   each question is used to retrieve the top 10 relevant chunks from the
   original corpus D, and LLM judgment is used to verify these chunks
   cannot answer the question. QA pairs that pass this verification are
   retained."

Verified by reading CRUMQs' `src/crumqs_generation/rag.py`:
  - `RAGVerifier` uses LlamaIndex `VectorStoreIndex`, `top_k=10`
  - Embedder configurable: `BAAI/bge-large-en-v1.5` (default), Cohere, OpenAI
  - One embedder per run (we use three)
  - Optional Cohere reranker + HyDE rewriting (we don't use either)
  - LLM judge over retrieved chunks for "can this answer the question?"

NOTE: CRUMQs cites UAEval4RAG for this step, but UAEval4RAG's released
generation code (`src/taxonomy/unanswerable_generation.py`) actually
verifies against the SOURCE chunk only (single-chunk LLM verify), not
retrieve-then-judge from the whole corpus. CRUMQs' interpretation of "as
in UAEval4RAG" appears to be a generalization of their idea, not a
literal reproduction. So the canonical reference for what we implement
here is **CRUMQs Step IV** (with their own published code), not UAEval4RAG.

## Our additions on top of CRUMQs Step IV

1. **Three embedders + strict AND criterion**.
   Per Bettina's cross-verification suggestion (memory:
   project_layer1b_oos_validation.md). CRUMQs uses one embedder; we use
   three from different lineages so the pass criterion is robust to
   single-embedder blind spots:
     - BAAI/bge-small-en-v1.5                       (BAAI)
     - intfloat/e5-small-v2                         (Microsoft)
     - sentence-transformers/all-MiniLM-L6-v2       (sentence-transformers)
   A prompt is kept iff ALL retrievals' top-K judge it unanswerable.

2. **Judge model ≠ generation model**.
   Generation uses gpt-4o-mini. Judging with the same model risks
   LLM-as-judge self-preference bias (Zheng et al., NeurIPS 2023). We
   default the judge to gpt-4o (stronger, same family) but it is freely
   configurable; using a different-family model (e.g. Llama-3.3-70B) is
   ideal when network reliability allows.

## Algorithm

```
for each OOS candidate Q:
  for each embedder E in {bge-small-en-v1.5, e5-small-v2, all-MiniLM-L6-v2}:
    embed Q with E → cosine-sim vs all corpus chunks → top-10_E
    LLM judge (gpt-4o by default): given Q + top-10_E, is Q answerable?
  keep iff ALL judges say NO
```

Reads:  <output_dir>/prompts.jsonl       (output of out_of_scope/eloq.py)
Writes: <output_dir>/prompts_with_retriever.jsonl    annotated, all rows
        <output_dir>/prompts_validated.jsonl         final: kept iff retriever_pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from evalsuite._io import read_jsonl, append_jsonl
from evalsuite.corpus.extract import load_corpus
from evalsuite.generators._common import (
    DEFAULT_MODEL,
    MAX_DOC_CHARS,
    llm_call,
)


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("data/out-of-scope/eloq")
DEFAULT_CORPUS_DIR = Path("data/worldbank-zip")
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # CRUMQs default; pass any HF model
DEFAULT_TOP_K = 10                                # CRUMQs default

# Recommended pair for cross-verification (run this script twice):
#   1) BAAI/bge-large-en-v1.5      (BAAI, English-Chinese training mix)
#   2) intfloat/e5-large-v2        (Microsoft, web mix)
# Different lineages → less correlated retrieval blind spots.
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNK_CHARS_FOR_JUDGE = 6000  # cap total chunk text fed to the LLM


# ── Corpus → chunks → embeddings (cached) ──────────────────────────

def chunk_corpus(
    corpus_dir: Path, chunk_size: int, chunk_overlap: int,
) -> tuple[list[str], list[str]]:
    """Return (chunks, doc_ids) parallel arrays."""
    docs = load_corpus(corpus_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[str] = []
    doc_ids: list[str] = []
    for d in docs:
        text = (d.get("text") or "")[:MAX_DOC_CHARS]
        if not text:
            continue
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(piece)
                doc_ids.append(d.get("doc_id", ""))
    return chunks, doc_ids


def embed_or_load(
    chunks: list[str], embed_model: str, cache_path: Path,
) -> np.ndarray:
    """Embed chunks with the given model; cache to .npy keyed by chunk count."""
    cache_meta_path = cache_path.with_suffix(".meta.json")
    expected_meta = {"n_chunks": len(chunks), "model": embed_model}
    if cache_path.exists() and cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text())
        if meta == expected_meta:
            print(f"Loading cached embeddings: {cache_path} ({len(chunks)} × ?)")
            return np.load(cache_path)
        print(f"Cache mismatch (meta {meta} ≠ expected {expected_meta}); re-embedding")

    print(f"Embedding {len(chunks)} chunks with {embed_model}...")
    model = SentenceTransformer(embed_model)
    embs = model.encode(chunks, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embs)
    cache_meta_path.write_text(json.dumps(expected_meta))
    print(f"Saved embeddings → {cache_path}")
    return embs


def top_k(
    query_emb: np.ndarray, chunk_embs: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_k_indices, top_k_scores) — embeddings already L2-normalised
    so dot product == cosine similarity."""
    sims = chunk_embs @ query_emb
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


# ── LLM judge ───────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a careful reading-comprehension assistant. Given a question and "
    "a small set of retrieved passages, decide whether the question can be "
    "answered using ONLY information present in the passages."
)

JUDGE_USER = (
    "Question: {question}\n\n"
    "Retrieved passages:\n\"\"\"\n{passages}\n\"\"\"\n\n"
    "Can the question be answered using only the passages above? "
    "Reply with exactly one word: YES (answerable from the passages) "
    "or NO (the passages do not contain the answer)."
)


def judge_derivable(
    question: str, passages: list[str], model: str = DEFAULT_MODEL,
) -> bool:
    """Return True iff the LLM judges Q answerable from the passages."""
    joined = "\n\n---\n\n".join(passages)
    if len(joined) > MAX_CHUNK_CHARS_FOR_JUDGE:
        joined = joined[:MAX_CHUNK_CHARS_FOR_JUDGE] + "\n[...truncated]"
    out = llm_call(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                question=question, passages=joined)},
        ],
        model=model,
        temperature=0.0,
        max_tokens=8,
    )
    return out.strip().upper().startswith("YES")


# ── JSONL helpers ───────────────────────────────────────────────────


# ── Pipeline ────────────────────────────────────────────────────────

def _embedder_slug(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
    judge_model: str = DEFAULT_MODEL,
    top_k_n: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    limit: int | None = None,
) -> None:
    """Run retriever validation with ONE embedder. Output file is per-embedder
    so subsequent runs with a different embedder don't overwrite this one."""
    slug = _embedder_slug(embed_model)
    input_path = output_dir / "prompts.jsonl"
    annotated_path = output_dir / f"prompts_with_retriever__{slug}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run out_of_scope.eloq first.")

    prompts = read_jsonl(input_path)
    if limit:
        prompts = prompts[:limit]
    done_ids = {r["prompt_id"] for r in read_jsonl(annotated_path)}
    todo = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"Embedder:  {embed_model}")
    print(f"Output:    {annotated_path}")
    print(f"Total prompts: {len(prompts)}  done: {len(done_ids)}  todo: {len(todo)}")

    if todo:
        # Corpus + chunks + embeddings (cached per-embedder)
        chunks, chunk_doc_ids = chunk_corpus(corpus_dir, chunk_size, chunk_overlap)
        cache_path = output_dir / f"_chunk_embeddings__{slug}.npy"
        chunk_embs = embed_or_load(chunks, embed_model, cache_path)
        embedder = SentenceTransformer(embed_model)

        for i, p in enumerate(todo):
            try:
                t0 = time.time()
                q_emb = embedder.encode([p["text"]], normalize_embeddings=True)[0]
                idx, scores = top_k(q_emb, chunk_embs, top_k_n)
                top_chunks = [chunks[j] for j in idx]

                derivable = judge_derivable(p["text"], top_chunks, judge_model)

                record = {
                    **p,
                    "retriever_top_k_doc_ids": [chunk_doc_ids[j] for j in idx],
                    "retriever_top_k_scores":  [round(float(s), 4) for s in scores],
                    "derivable_judgement":     derivable,
                    "retriever_pass_this_run": not derivable,
                    "retriever_judge_model":   judge_model,
                    "retriever_embed_model":   embed_model,
                    "retriever_top_k":         top_k_n,
                }
                append_jsonl(record, annotated_path)
                flag = "✓" if not derivable else "✗"
                print(f"  [{i+1}/{len(todo)}] {p['prompt_id']} "
                      f"top-sim={scores[0]:.3f} "
                      f"derivable={'Y' if derivable else 'N'} {flag} "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [{i+1}/{len(todo)}] {p['prompt_id']}: ERROR {e}")
                continue

    rows = read_jsonl(annotated_path)
    _summarize_single(rows, embed_model, annotated_path)


def _summarize_single(rows: list[dict], embed_model: str, annotated_path: Path) -> None:
    if not rows:
        return
    n = len(rows)
    n_derivable = sum(1 for r in rows if r.get("derivable_judgement"))
    print(f"\n{'='*60}")
    print(f"Retriever validation — {n} prompts judged with {embed_model}")
    print(f"{'='*60}")
    print(f"  derivable from top-k (LEAK):         {n_derivable:4d} ({n_derivable/n:.0%})")
    print(f"  unanswerable per this embedder:      {n - n_derivable:4d} ({(n-n_derivable)/n:.0%})")
    print(f"\n  annotated: {annotated_path}")
    print(f"\nNext: run a SECOND time with a different --embed-model, then run")
    print(f"      `python -m evalsuite.generators.out_of_scope.combine_retriever_runs`")
    print(f"      to compute the cross-verified pass.")


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 2: retriever validation (run ONCE per embedder)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help=f"HF model name. Default: {DEFAULT_EMBED_MODEL} (CRUMQs default). "
                        "Run a second time with a different model for cross-verification, "
                        "e.g. intfloat/e5-large-v2.")
    p.add_argument("--judge-model", default=DEFAULT_MODEL,
                   help=f"LLM judge (default {DEFAULT_MODEL}). Should differ from "
                        "the generator model to avoid LLM-as-judge self-preference bias.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.output_dir, args.corpus_dir, args.embed_model,
        args.judge_model, args.top_k,
        args.chunk_size, args.chunk_overlap, args.limit)


if __name__ == "__main__":
    main()
