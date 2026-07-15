"""Truly shared utilities for generators (used everywhere):
  - LLM client + retry (`llm_call`)
  - SYSTEM_PROMPT, MAX_DOC_CHARS, DEFAULT_MODEL
  - numbered-list parsing (`parse_numbered_questions`, `parse_numbered_items`, `enum_list`)

Branch-specific prompts and helpers live next to their only consumer
(see `in_scope/eloq_prompts.py`, `in_scope/quality_filter_prompts.py`,
`out_of_scope/eloq.py`).
"""

from __future__ import annotations

import os
import re
import time

import openai
from dotenv import load_dotenv

load_dotenv()


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
MAX_DOC_CHARS = 30_000
# NO_ANSWER_SENTINEL moved to in_scope/quality_filter_prompts.py

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GENERATOR_API_KEY", "")


# ── LLM client ──────────────────────────────────────────────────────

_CLIENT: openai.OpenAI | None = None


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set (also tried GENERATOR_API_KEY)")
        _CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
    return _CLIENT


def llm_call(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Single LLM call with exponential backoff retry."""
    for attempt in range(5):
        try:
            resp = _client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Shared prompt header ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You will be provided with a document delimited by triple quotes. "
    "Read the document and follow user's instructions."
)


# ── Numbered-list parsing ───────────────────────────────────────────

_NUMBER_PREFIX_RE = re.compile(r"^\d+[:.]\s+")
_BULLET_PREFIX_RE = re.compile(r"^[-*+]\s+")


def parse_numbered_items(text: str) -> list[str]:
    """Extract numbered/bulleted items; multi-line items joined."""
    items: list[str] = []
    chunks: list[str] = []

    def flush():
        nonlocal chunks
        if chunks:
            joined = " ".join(chunks).strip()
            if joined:
                items.append(joined)
            chunks = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _NUMBER_PREFIX_RE.search(line)
        if m:
            flush()
            line = line[m.end():]
        else:
            m2 = _BULLET_PREFIX_RE.search(line)
            if m2:
                flush()
                line = line[m2.end():]
        chunks.append(line)
    flush()
    return items


def parse_numbered_questions(text: str) -> list[str]:
    """Numbered-list parser, filtered to items ending with '?'."""
    return [q for q in parse_numbered_items(text) if q.endswith("?")]


def enum_list(items: list[str], start: int = 1) -> str:
    """Format items as a numbered list (used by OOS mask-impute)."""
    return "\n".join(f"{i}. {q}" for i, q in enumerate(items, start=start))


# Branch-specific helpers live next to their consumers:
#   - extract_claims (REDUCE_PROMPT)            → out_of_scope/eloq.py
#   - derive_answer / check_standalone          → in_scope/quality_filter_prompts.py
