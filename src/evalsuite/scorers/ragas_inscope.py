"""RAGAS-style in-scope correctness scorer.

Runs the subset of RAGAS metrics that need only `(question, answer,
reference)` — no retrieved contexts — since Apertus's /chat endpoint
returns the answer text only. The context-dependent metrics
(Faithfulness, ContextPrecision, ContextRecall) are intentionally
gated behind --include-context-metrics in run_layer.py and cannot run
until UNICC also returns retrieval traces.

Default metric: AnswerSimilarity
  cos = dot(emb(answer), emb(reference)) / (|emb(answer)| * |emb(reference)|)
  score = (1 + cos) / 2          ∈ [0, 1]
This is RAGAS's exact formula for AnswerSimilarity with a vanilla
sentence-transformers embedder; we compute it directly to avoid the
LangChain dep stack. Embedding model controlled by EMBEDDING_MODEL
(default BAAI/bge-small-en-v1.5, same as quality_filter).

Optional metric: FactualCorrectness (claim-decomposition + NLI, LLM
judge). Routes through the existing openrouter-llama provider. ~1 LLM
call per row; opt in via `factual_correctness=True`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()


@dataclass
class InScopeScore:
    answer_similarity: float | None = None      # [0,1] or None if disabled
    factual_correctness: float | None = None    # [0,1] or None if disabled
    error: str = ""


# ── Lazy singletons ──────────────────────────────────────────────────

_embedder = None
_ragas_llm = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        _embedder = SentenceTransformer(model)
    return _embedder


def _get_ragas_llm():
    global _ragas_llm
    if _ragas_llm is None:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("JUDGE_API_KEY", "")
        base_url = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
        _ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.0)
        )
    return _ragas_llm


# ── Per-row scoring ──────────────────────────────────────────────────

def _score_similarity(answer: str, reference: str) -> float:
    import numpy as np
    e = _get_embedder().encode([answer, reference], convert_to_numpy=True, show_progress_bar=False)
    cos = float(np.dot(e[0], e[1]) / (np.linalg.norm(e[0]) * np.linalg.norm(e[1]) + 1e-12))
    return (1.0 + cos) / 2.0


def _score_factual_correctness(question: str, answer: str, reference: str) -> float:
    import asyncio
    from ragas.metrics import FactualCorrectness
    from ragas.dataset_schema import SingleTurnSample

    metric = FactualCorrectness(llm=_get_ragas_llm())
    sample = SingleTurnSample(user_input=question, response=answer, reference=reference)
    return float(asyncio.run(metric.single_turn_ascore(sample)))


def score_batch(
    records: list[dict],
    *,
    similarity: bool = True,
    factual_correctness: bool = False,
    show_progress: bool = True,
) -> list[InScopeScore]:
    """Each record needs `prompt`, `response`, and one of
    `reference_answer` / `derived_answer` as the gold."""
    items = tqdm(records, desc="RAGAS in-scope") if show_progress else records
    results: list[InScopeScore] = []
    for r in items:
        ref = r.get("reference_answer") or r.get("derived_answer") or ""
        q = r.get("prompt", "")
        a = r.get("response", "")
        s = InScopeScore()
        if not ref or not a:
            s.error = "missing reference or response"
            results.append(s)
            continue
        try:
            if similarity:
                s.answer_similarity = _score_similarity(a, ref)
            if factual_correctness:
                s.factual_correctness = _score_factual_correctness(q, a, ref)
        except Exception as e:
            s.error = str(e)
        results.append(s)
    return results
