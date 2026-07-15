"""In-scope ELOQ generator. Single LLM call per doc → top N questions.

Run: python -m evalsuite.generators.in_scope.eloq --num-docs 50

Prompt variants live in `eloq_prompts.py`. Reference: arXiv:2410.14567.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from evalsuite._io import read_jsonl, append_jsonl, save_jsonl
from evalsuite.corpus.extract import load_corpus
from evalsuite.generators._common import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    llm_call,
    parse_numbered_questions,
)
from evalsuite.generators.in_scope.eloq_prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPTS,
    VERSION_TAGS,
)


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_NUM_Q = 5              # questions per document
DEFAULT_NUM_DOCS = 50
DEFAULT_OUTPUT_DIR = Path("data/in-scope/eloq")
DEFAULT_CORPUS_DIR = Path("data/worldbank")


# ── Document sampling ──────────────────────────────────────────────

def sample_documents(
    docs: list[dict],
    num_docs: int,
    min_chars: int = 1000,
    max_chars: int = 30000,
    seed: int = 42,
) -> list[dict]:
    """Random sample of docs in a sensible length range.

    Stratification by `docty` is left to a higher-level wrapper if needed —
    this function keeps the eloq generator simple. The corpus pipeline
    already preserves the `docty` field so a caller can pre-filter.
    """
    eligible = [d for d in docs if min_chars <= d.get("char_count", 0) <= max_chars]
    if len(eligible) < num_docs:
        print(f"  warning: only {len(eligible)} eligible docs; using all")
        return eligible
    rng = random.Random(seed)
    return rng.sample(eligible, num_docs)


# ── Pipeline ────────────────────────────────────────────────────────

@dataclass
class DocResult:
    doc_id: str
    title: str
    questions: list[str] = field(default_factory=list)
    error: str = ""


def generate_questions(
    document: str, num_q: int, model: str, prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[str]:
    """One LLM call → list of in-scope questions."""
    prompt = PROMPTS[prompt_version]
    raw = llm_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.format(
                document=document, num_q=num_q)},
        ],
        model=model,
    )
    return parse_numbered_questions(raw)


def process_document(
    doc: dict, num_q: int, model: str, prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> DocResult:
    doc_id = doc.get("doc_id", "unknown")
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    result = DocResult(doc_id=doc_id, title=title)
    try:
        result.questions = generate_questions(text, num_q, model, prompt_version)
    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")
    return result


def generate_all(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_docs: int = DEFAULT_NUM_DOCS,
    num_q: int = DEFAULT_NUM_Q,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict]:
    """Generate in-scope prompts → <output_dir>/prompts.jsonl.

    Resumable via <output_dir>/checkpoint.jsonl: re-runs skip docs already
    processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.jsonl"
    prompts_path = output_dir / "prompts.jsonl"

    docs = load_corpus(corpus_dir)
    print(f"Loaded corpus: {len(docs)} docs from {corpus_dir}")
    sampled = sample_documents(docs, num_docs=num_docs, seed=seed)
    print(f"Sampled {len(sampled)} docs (seed={seed})")

    done: dict[str, dict] = {r["doc_id"]: r for r in read_jsonl(ckpt_path)}
    if done:
        print(f"Checkpoint: {len(done)} docs already done — resuming")

    doc_results: list[DocResult] = []
    for rec in done.values():
        doc_results.append(DocResult(
            doc_id=rec["doc_id"],
            title=rec.get("title", ""),
            questions=rec.get("questions", []),
            error=rec.get("error", ""),
        ))

    remaining = [d for d in sampled if d["doc_id"] not in done]
    print(f"Remaining: {len(remaining)} documents\n")

    for i, doc in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {doc['doc_id']}: {(doc.get('title') or '')[:70]}")
        result = process_document(doc, num_q, model, prompt_version)
        doc_results.append(result)
        append_jsonl(
            {
                "doc_id": result.doc_id,
                "title": result.title,
                "questions": result.questions,
                "error": result.error,
            },
            ckpt_path,
        )
        print(f"  → {len(result.questions)} questions")

    # Build the unified prompts JSONL
    prompts: list[dict] = []
    idx = 0
    for dr in doc_results:
        if dr.error:
            continue
        for q in dr.questions:
            prompts.append({
                "prompt_id": f"in_scope_{idx:04d}",
                "text": q,
                "source": "eloq",
                "category": "inscope",
                "expected_behavior": "comply",
                "doc_id": dr.doc_id,
                "doc_title": dr.title,
                "generator_version": VERSION_TAGS[prompt_version],
                "generator_model": model,
            })
            idx += 1
    save_jsonl(prompts, prompts_path)

    n_err = sum(1 for dr in doc_results if dr.error)
    print(f"\n{'='*50}")
    print(f"In-scope ELOQ generation")
    print(f"{'='*50}")
    print(f"Documents:  {len(doc_results)} processed ({n_err} errors)")
    print(f"Questions:  {len(prompts)}")
    print(f"Outputs:")
    print(f"  prompts:    {prompts_path}")
    print(f"  checkpoint: {ckpt_path}")

    return prompts


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="In-scope ELOQ question generator")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                   help="Directory containing documents.jsonl (default: data/worldbank)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS)
    p.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OpenAI model (ELOQ paper: {DEFAULT_MODEL})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION,
                   choices=list(PROMPTS.keys()),
                   help="Generation prompt variant. v3=our augmented (min-specificity "
                        "+ Bad/Good example pair). v5=verbatim ELOQ minus entity restriction.")
    args = p.parse_args()

    generate_all(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        num_docs=args.num_docs,
        num_q=args.num_q,
        model=args.model,
        seed=args.seed,
        prompt_version=args.prompt_version,
    )


if __name__ == "__main__":
    main()
