"""Prompts + helpers for the in-scope quality filter.

Two binary judges:
  - derive_answer + is_extractable  → answerable from source doc?
  - check_standalone                → subject identifiable WITHOUT source doc?

Both are LLM calls returning a binary verdict. Living next to the only
consumer (`quality_filter.py`) rather than in `_common.py`.
"""

from __future__ import annotations

from evalsuite.generators._common import DEFAULT_MODEL, MAX_DOC_CHARS, llm_call


NO_ANSWER_SENTINEL = "NO_ANSWER_IN_DOCUMENT"


# ── Answer extraction ───────────────────────────────────────────────

ANSWER_EXTRACTION_SYSTEM = (
    "You are a careful reading-comprehension assistant. "
    "Given a document and a question, extract a concise answer using only "
    "information present in the document. If the document does not contain "
    f"an answer, respond with exactly: {NO_ANSWER_SENTINEL}."
)

ANSWER_EXTRACTION_USER = (
    "Document:\n\"\"\"\n{doc}\n\"\"\"\n\n"
    "Question: {question}\n\n"
    "Answer (one sentence, drawn only from the document):"
)


def derive_answer(question: str, doc: str, model: str = DEFAULT_MODEL) -> str:
    """LLM call: extract the answer from the doc, or NO_ANSWER sentinel."""
    return llm_call(
        [
            {"role": "system", "content": ANSWER_EXTRACTION_SYSTEM},
            {"role": "user", "content": ANSWER_EXTRACTION_USER.format(
                doc=doc[:MAX_DOC_CHARS], question=question)},
        ],
        model=model,
        temperature=0.3,
        max_tokens=256,
    )


def is_extractable(derived_answer: str) -> bool:
    """Binary verdict on derive_answer's output."""
    return NO_ANSWER_SENTINEL not in derived_answer


# ── Standalone check ────────────────────────────────────────────────

STANDALONE_SYSTEM = (
    "You evaluate whether a question is fully self-contained. You do NOT "
    "have access to any source document — only the question itself."
)

STANDALONE_USER = (
    "Question: {question}\n\n"
    "Is this question too vague to identify what specific subject is being "
    "asked about? Pass it unless it is clearly underspecified — i.e., a "
    "search engine over a large document corpus would not be able to tell "
    "which topic, entity, or document the question concerns.\n\n"
    "Acronyms, partial names, and common entity references are fine as long "
    "as the subject is identifiable.\n\n"
    "Reply with exactly one word: OK (specific enough to identify a subject) "
    "or VAGUE (too underspecified)."
)


def check_standalone(question: str, model: str = DEFAULT_MODEL) -> bool:
    """LLM call (sees Q only, no doc): True iff subject is identifiable."""
    out = llm_call(
        [
            {"role": "system", "content": STANDALONE_SYSTEM},
            {"role": "user", "content": STANDALONE_USER.format(question=question)},
        ],
        model=model,
        temperature=0.0,
        max_tokens=8,
    )
    return out.strip().upper().startswith("OK")


# ── Single-aspect check ─────────────────────────────────────────────
#
# Catches the residual failure mode in annotator-B's review of v3:
# compound questions like
#   "What role does the World Bank aim to play in reducing poverty
#    AND supporting economic growth in developing countries?"
# which pass extractable + standalone but ask about ≥2 distinct
# aspects, so the gold answer can't be a single specific fact.

SINGLE_ASPECT_SYSTEM = (
    "You evaluate whether a question targets exactly ONE specific aspect, "
    "fact, decision, number, event, or finding. You do NOT have access to "
    "any source document — only the question itself."
)

SINGLE_ASPECT_USER = (
    "Question: {question}\n\n"
    "Does this question target exactly ONE specific aspect (one fact, "
    "one decision, one number, one event, one finding)?\n\n"
    "Mark VAGUE if the question:\n"
    "  - is a compound question joining two distinct topics with 'and' "
    "    (e.g. 'poverty and economic growth', 'AI's impact on inequality "
    "    and labor displacement')\n"
    "  - asks broadly 'what role does X play' or 'how is X responding' "
    "    without pinning a specific decision/fact\n"
    "  - is a multi-part question (would naturally require multiple "
    "    separate answers)\n\n"
    "Mark OK if the question pins a single, specific fact that can be "
    "answered in one sentence (a date, an amount, one decision, one "
    "named action).\n\n"
    "Reply with exactly one word: OK or VAGUE."
)


def check_single_aspect(question: str, model: str = DEFAULT_MODEL) -> bool:
    """LLM call (sees Q only): True iff the question targets ONE specific aspect."""
    out = llm_call(
        [
            {"role": "system", "content": SINGLE_ASPECT_SYSTEM},
            {"role": "user", "content": SINGLE_ASPECT_USER.format(question=question)},
        ],
        model=model,
        temperature=0.0,
        max_tokens=8,
    )
    return out.strip().upper().startswith("OK")
