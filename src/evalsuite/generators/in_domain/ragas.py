"""In-domain (RAGAS) question generator — one dataset, two downstream uses.

Uses RAGAS 0.4.x's TestsetGenerator on a knowledge graph built over
chunked documents. Produces (question, reference_answer, reference_contexts)
triples — every answer is grounded in the indexed corpus by construction,
because RAGAS' synthesizers pick chunks first and then synthesize a Q/A
pair whose answer is supported by those chunks.

That grounding invariant means the SAME dataset serves both layers:
  • Layer 1-B in-scope/FRR: score Apertus's answer against `reference_answer`
                            via the LLM-judge correctness + refusal scorer
  • Layer 2-B hallucination: score Apertus's answer against the contexts
                            it retrieved at run-time via RAGAS Faithfulness
                            / FactualCorrectness

This unifies the previously-split in-scope-ragas and hallucination-ragas
datasets (which were the same generation, just different sample sizes and
downstream scorers).

Three question types from RAGAS' default synthesizers:
  - single-hop specific  (one chunk, one fact)
  - multi-hop specific   (multiple chunks, specific facts)
  - multi-hop abstract   (multiple chunks, higher-level synthesis)

For ELOQ-style single-doc in-scope Qs use `in_scope/eloq.py` instead —
complementary methodology (paper-faithful Algorithm 1) but no multi-hop.

Requires the corpus pipeline to have run: `python -m evalsuite.corpus.extract`.

Run:
  python -m evalsuite.generators.in_domain.ragas --testset-size 500 --max-docs 200

Reference:
  Es et al. 2024, "RAGAS: Automated Evaluation of Retrieval Augmented
    Generation" (arXiv:2309.15217).
  RAGAS docs: https://docs.ragas.io/
  Prior in-scope-only run (legacy): data/legacy/in-scope-ragas-final.jsonl
  Prior hallucination-only run:     feat/evalsuite-layer-2b commit ee063bf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

from evalsuite.corpus.extract import load_corpus
from evalsuite._io import read_jsonl, append_jsonl

load_dotenv()


# ── Configuration ───────────────────────────────────────────────────

GENERATOR_VERSION = "in_domain_ragas_v2"
DEFAULT_OUTPUT_DIR = Path("data/in-domain/ragas")
DEFAULT_CORPUS_DIR = Path("data/worldbank-zip")
DEFAULT_MAX_DOCS = 200
DEFAULT_TESTSET_SIZE = 300

# LLM (RAGAS uses this for question generation + extraction stages)
# These can be overridden by --provider in main(), which selects a known config.
# Defaults read from .env (the dotenv-loaded values).
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
GENERATOR_BASE_URL = os.getenv("GENERATOR_BASE_URL", "")
GENERATOR_API_KEY = os.getenv(
    "GENERATOR_API_KEY",
    os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")),
)

# Known-good provider configs. Selecting one resolves the auth+URL+model
# trio inside Python (no shell escaping involved).
PROVIDER_CONFIGS = {
    "openai": {
        "model":    "gpt-4o-mini",
        "base_url": "",   # empty → OpenAI default URL
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter-llama": {
        "model":    "meta-llama/llama-3.3-70b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openrouter-gpt-mini": {
        "model":    "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


def _resolve_provider(name: str) -> tuple[str, str, str]:
    """Returns (api_key, base_url, model) for the named provider.
    Raises if the required env var isn't set."""
    if name not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown --provider {name!r}. Choices: {list(PROVIDER_CONFIGS)}")
    cfg = PROVIDER_CONFIGS[name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(
            f"Provider {name!r} needs env var {cfg['api_key_env']} set "
            "(in .env or shell). It is empty.")
    return api_key, cfg["base_url"], cfg["model"]

# Embeddings (RAGAS uses these for chunk similarity in the knowledge graph)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hf").lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5" if EMBEDDING_PROVIDER == "hf" else "text-embedding-3-small",
)


# ── RAGAS robustness patch ──────────────────────────────────────────

def _patch_headline_splitter() -> None:
    """Make RAGAS's HeadlineSplitter tolerate nodes missing 'headlines'.

    Why: HeadlinesExtractor uses Instructor (Pydantic schema validation).
    When the LLM returns malformed JSON (common with non-OpenAI models,
    occasionally with gpt-4o on edge-case docs), Instructor silently fails
    and the 'headlines' property is never set. The default
    `HeadlineSplitter.split` then raises ValueError and aborts the whole
    transform pipeline at the very first bad node.

    Patch: when 'headlines' is missing or empty, return ([], []) so the
    original node persists unsplit. Downstream extractors (NER, keyphrase,
    themes) and synthesizers handle un-split nodes fine — we just lose
    the section-level chunking for that one doc.

    Idempotent: noop if called twice.
    """
    from ragas.testset.transforms.splitters import headline as _h
    if getattr(_h.HeadlineSplitter, "_evalsuite_patched", False):
        return
    _orig_split = _h.HeadlineSplitter.split

    async def safe_split(self, node):
        headlines = node.properties.get("headlines") if hasattr(node, "properties") else None
        if not headlines:
            return [], []
        try:
            return await _orig_split(self, node)
        except (ValueError, KeyError, AttributeError) as e:
            print(f"  [HeadlineSplitter] skipping node: {type(e).__name__}: {e}")
            return [], []

    _h.HeadlineSplitter.split = safe_split
    _h.HeadlineSplitter._evalsuite_patched = True


# ── Utilities ───────────────────────────────────────────────────────

_BAD_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RAGAS_TAG_RE = re.compile(r"^<[^>]+>\s*")  # strips RAGAS internal tags like <1-hop>


def _sanitize(text: str) -> str:
    """Strip control chars (except \\t \\n \\r) — PDF extraction can leak them."""
    return _BAD_CHARS_RE.sub(" ", text)


def _slug(model: str) -> str:
    return model.split("/")[-1].replace(":", "_").replace(".", "")


def _context_text(ctx: Any) -> str:
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        return (ctx.get("page_content") or ctx.get("text") or ctx.get("content") or "").strip()
    return getattr(ctx, "page_content", getattr(ctx, "text", str(ctx))).strip()


_PYMUPDF_PIC_RE = re.compile(
    r"\*?\*?==>\s*picture\s*\[[^\]]*\]\s*intentionally omitted\s*<==\*?\*?")


def _match_doc_id(ctx_text: str, full_docs: list[Document]) -> str | None:
    """Match a RAGAS-returned context back to its source document.
    RAGAS contexts are raw text without metadata; we recover the doc_id
    by substring-matching probes from the context against the full docs.

    Probing strategy: pymupdf4llm injects identical "==> picture [..]
    intentionally omitted <==" placeholders across many docs, so a single
    head-200-char probe collides. We strip the placeholder, then try
    multiple offset windows (50, 200, 500, 1000, 2000 chars in, plus the
    end of the context) and return the first doc whose text contains ANY
    of those probes."""
    if not ctx_text:
        return None
    cleaned = _RAGAS_TAG_RE.sub("", ctx_text).strip()
    cleaned = _PYMUPDF_PIC_RE.sub("", cleaned).strip()
    if not cleaned:
        return None
    n = len(cleaned)
    offsets = [o for o in (50, 200, 500, 1000, 2000) if o + 200 <= n]
    if not offsets:
        offsets = [0]
    if n > 400:
        offsets.append(n - 200)
    for off in offsets:
        probe = cleaned[off:off + 200].strip()
        if len(probe) < 60:
            continue
        for d in full_docs:
            if probe in d.page_content:
                return d.metadata["doc_id"]
    return None


# ── Corpus loading ──────────────────────────────────────────────────

def _to_langchain_docs(rows: list[dict]) -> list[Document]:
    """Convert our documents.jsonl rows → LangChain Documents.
    Keeps WB metadata (doc_id, title, date, docty) on each Document."""
    out: list[Document] = []
    for r in rows:
        text = _sanitize((r.get("text") or "").strip())
        if not text:
            continue
        out.append(Document(
            page_content=text,
            metadata={
                "doc_id": r.get("doc_id"),
                "title": r.get("title"),
                "date": r.get("date"),
                "docty": r.get("docty"),
                "guid": r.get("guid"),
            },
        ))
    return out


# ── Batch tracking ──────────────────────────────────────────────────


def _select_unused_docs(
    rows: list[dict], max_docs: int, used_doc_ids: set[str], seed: int,
) -> list[dict]:
    """Randomly sample up to max_docs from rows whose doc_id is NOT in used_doc_ids.
    This is what enables batched generation: each batch picks fresh docs."""
    unused = [r for r in rows if r.get("doc_id") not in used_doc_ids]
    rng = random.Random(seed)
    if max_docs >= len(unused):
        return unused
    return rng.sample(unused, max_docs)


# ── Pipeline ────────────────────────────────────────────────────────

def generate(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_docs: int = DEFAULT_MAX_DOCS,
    testset_size: int = DEFAULT_TESTSET_SIZE,
    seed: int = 42,
    batch_id: str | None = None,
    provider: str | None = None,
    model_override: str | None = None,
) -> list[dict]:
    """Run RAGAS in-scope generation as a single BATCH.

    Each batch picks `max_docs` docs from the corpus EXCLUDING any doc
    already used in a previous batch (tracked in `_docs_used.jsonl`).
    Outputs are appended to:
      - `<output_dir>/prompts.jsonl`     all questions across all batches
                                          (each row tagged with batch_id)
      - `<output_dir>/_docs_used.jsonl`  registry of (doc_id, batch_id, ts)

    Re-running with the same batch_id is idempotent: docs already attributed
    to that batch_id are excluded along with all other used docs, so the
    sampling pool is the unused-and-not-this-batch set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if batch_id is None:
        batch_id = "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch: {batch_id}")

    used_registry_path = output_dir / "_docs_used.jsonl"
    prompts_path = output_dir / "prompts.jsonl"

    # 1. Load corpus, exclude already-used docs, sample
    used_records = read_jsonl(used_registry_path)
    used_doc_ids = {r["doc_id"] for r in used_records}
    print(f"Used-doc registry: {len(used_doc_ids)} docs already used in prior batches")

    corpus_rows = load_corpus(corpus_dir)
    sampled_rows = _select_unused_docs(corpus_rows, max_docs, used_doc_ids, seed)
    docs = _to_langchain_docs(sampled_rows)
    print(f"Sampled {len(docs)} unused docs (target max_docs={max_docs}, "
          f"corpus has {len(corpus_rows)} total, {len(corpus_rows) - len(used_doc_ids)} unused)")
    if not docs:
        print("Nothing to generate — every doc has been used. Exit.")
        return []

    # 2. Resolve provider config (in Python — no shell escaping involved)
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    # Apply HeadlineSplitter robustness patch BEFORE TestsetGenerator runs
    _patch_headline_splitter()

    if provider:
        api_key, base_url, model = _resolve_provider(provider)
        print(f"Provider:   {provider}")
    else:
        api_key, base_url, model = GENERATOR_API_KEY, GENERATOR_BASE_URL, GENERATOR_MODEL
    if model_override:
        model = model_override
        print(f"Model override: {model}")
    if not api_key:
        raise ValueError(
            "No API key resolved. Either set --provider or ensure the relevant "
            "env var (OPENAI_API_KEY / OPENROUTER_API_KEY / GENERATOR_API_KEY) "
            "is set in .env."
        )

    # timeout=60 is REQUIRED. Without it the OpenAI client hangs forever on
    # half-closed Cloudfront sockets (CLOSE_WAIT) and the entire pipeline
    # stalls silently — see project memory `feedback_long_runs_caffeinate.md`.
    # max_retries=3 covers transient 5xx / 429.
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 3,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    llm = llm_factory(model, provider="openai", client=client)
    print(f"LLM:        {model} @ {base_url or 'openai-default'} (timeout=60s, retries=3)")

    # 3. Embeddings
    if EMBEDDING_PROVIDER == "hf":
        from ragas.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
        print(f"Embeddings: HF local — {EMBEDDING_MODEL}")
    elif EMBEDDING_PROVIDER == "openai":
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        emb_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=60.0, max_retries=3)
        embedding_model = RagasOpenAIEmbeddings(client=emb_client)
        print(f"Embeddings: OpenAI — {EMBEDDING_MODEL}")
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")

    # 4. Generate using RAGAS' canonical pipeline.
    # Our corpus pipeline (pymupdf4llm) now emits Markdown with `##` section
    # headers, so HeadlineSplitter + HeadlinesExtractor + KeyphrasesExtractor +
    # NERExtractor + ThemesExtractor all work correctly. This is RAGAS' full
    # KG-based test-set synthesis, not the bypass path we used for plain-text
    # corpora previously.
    #
    # RunConfig is REQUIRED at scale: the OpenAI client `timeout=60` we set
    # on the bare client does NOT propagate through RAGAS' internal LLM
    # wrapper (llm_factory creates its own httpx-based client). RAGAS only
    # honours timeouts via RunConfig. Without it, async tasks block on dead
    # Cloudfront sockets in CLOSE_WAIT and the entire pipeline hangs (we
    # observed this at 73 min and 60 min in 300-doc runs). RunConfig fields:
    #   - timeout:      per-LLM-call wall clock cap
    #   - max_retries:  per-task retry on timeout/error
    #   - max_workers:  concurrent async tasks (lower = fewer parallel
    #                   sockets to drop)
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=60, max_retries=3, max_workers=8)
    generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
    print(f"\nGenerating {testset_size} test cases from {len(docs)} docs "
          f"(canonical path: HeadlineSplitter + extractors + KG)...")
    print(f"RunConfig: timeout=60s retries=3 workers=8")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=testset_size,
        raise_exceptions=False,
        run_config=run_config,
    )

    # 6. Convert RAGAS testset → in-scope schema
    df = testset.to_pandas()
    prompts: list[dict] = []
    raw_rows: list[dict] = []
    misses = 0

    # Build a lookup of doc_id → title for the doc_title field
    title_by_doc_id = {d.metadata["doc_id"]: d.metadata.get("title", "") for d in docs}

    for i, r in df.iterrows():
        question = r.get("user_input") or r.get("question") or ""
        reference_answer = (
            r.get("reference") or r.get("ground_truth")
            or r.get("answer") or r.get("reference_answer") or ""
        )
        contexts = (
            r.get("reference_contexts") or r.get("retrieved_contexts")
            or r.get("contexts") or []
        )
        if not isinstance(contexts, list):
            contexts = [contexts] if contexts is not None else []

        doc_ids: set[str] = set()
        for c in contexts:
            doc_id = _match_doc_id(_context_text(c), docs)
            if doc_id:
                doc_ids.add(doc_id)
            else:
                misses += 1

        # Use the first matched doc as the canonical source for the prompt row
        primary_doc_id = next(iter(sorted(doc_ids)), None)
        primary_doc_title = title_by_doc_id.get(primary_doc_id, "") if primary_doc_id else ""

        # Globally unique prompt_id across batches: prefix with batch_id so
        # we never collide with rows from a previous batch's `in_scope_NNNN`.
        prompt_id = f"{batch_id}__in_scope_{i:04d}"

        prompts.append({
            "prompt_id": prompt_id,
            "text": question,
            "source": "ragas",
            "category": "in_domain",  # used by both 1-B in-scope FRR and 2-B hallucination scorers
            "expected_behavior": "comply",
            "doc_id": primary_doc_id,
            "doc_title": primary_doc_title,
            "all_doc_ids": sorted(doc_ids),
            "reference_answer": reference_answer,
            # Needed by in_domain.quality's derivability filter AND by 2-B
            # hallucination's Faithfulness scorer. Stored as a list of chunk
            # strings (the same shape RAGAS returns).
            "reference_contexts": [_context_text(c) for c in contexts],
            "synthesizer": r.get("synthesizer_name"),
            "batch_id": batch_id,
            "generator_version": GENERATOR_VERSION,
            "generator_model": GENERATOR_MODEL,
        })

        # Sidecar with the full RAGAS row for debugging
        raw_rows.append({
            "prompt_id": prompt_id,
            **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()},
            "matched_doc_ids": sorted(doc_ids),
            "batch_id": batch_id,
        })

    print(f"Source-doc recovery: {len(prompts) - sum(1 for p in prompts if not p['doc_id'])}/{len(prompts)} matched, {misses} context misses")

    # 7. Append outputs (cross-batch accumulation) + register used docs
    slug = _slug(GENERATOR_MODEL)
    raw_path = output_dir / f"raw_{slug}.jsonl"

    # Prompts: append-mode so rows from earlier batches aren't overwritten.
    # Override generator_model with what we actually used (in case provider
    # was set and overrode the .env default).
    for r in prompts:
        r["generator_model"] = model
    with prompts_path.open("a", encoding="utf-8") as f:
        for r in prompts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with raw_path.open("a", encoding="utf-8") as f:
        for r in raw_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Register every doc we *fed* to RAGAS this batch (whether or not it
    # ended up contributing a question — we still don't want to re-feed it).
    ts = datetime.now().isoformat()
    for d in docs:
        append_jsonl(
            {"doc_id": d.metadata["doc_id"], "batch_id": batch_id, "timestamp": ts},
            used_registry_path,
        )

    print(f"\nSaved {len(prompts)} test cases (batch '{batch_id}'):")
    print(f"  prompts (appended): {prompts_path}")
    print(f"  raw (appended):     {raw_path}")
    print(f"  used-docs registry: {used_registry_path} (+{len(docs)} entries)")
    return prompts


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="In-domain RAGAS question generator (feeds both "
                    "1-B in-scope FRR and 2-B hallucination Faithfulness)")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                   help=f"Directory containing documents.jsonl (default: {DEFAULT_CORPUS_DIR})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                   help="Max number of NEW (unused) docs to feed RAGAS this batch")
    p.add_argument("--testset-size", type=int, default=DEFAULT_TESTSET_SIZE,
                   help="Number of test cases for RAGAS to generate")
    p.add_argument("--batch-id", default=None,
                   help="Identifier for this batch (default: auto timestamp). "
                        "Subsequent batches sample from the corpus EXCLUDING all "
                        "doc_ids already used in any prior batch (registry: "
                        "<output-dir>/_docs_used.jsonl).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--provider", default=None,
                   choices=list(PROVIDER_CONFIGS),
                   help="Resolves auth+URL+model in Python (avoids shell-escape "
                        "bugs with .env). If omitted, falls back to GENERATOR_* "
                        "env vars.")
    p.add_argument("--model", default=None,
                   help="Explicit model override (e.g. 'gpt-4o', 'gpt-4o-mini'). "
                        "Overrides the model selected by --provider.")
    args = p.parse_args()

    generate(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        testset_size=args.testset_size,
        seed=args.seed,
        batch_id=args.batch_id,
        provider=args.provider,
        model_override=args.model,
    )


if __name__ == "__main__":
    main()
