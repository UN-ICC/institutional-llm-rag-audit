"""LLM-judge correctness + refusal flag for in-scope responses.

Given (question, model response, gold reference answer), returns two
binary judgments via the project's Llama-3.3-70B judge:

  correct : True iff the response contains the same key info as the gold
  refused : True iff the model declined / failed to engage with the Q

Used by scripts/score_inscope_eloq.py (and analogous in-scope scripts)
to compute headline metrics:
  correctness_rate = correct / scored
  false_refusal_rate (FRR) = refused / total   (refusal on in-scope = bad)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from evalsuite.scorers._judge_client import complete


JUDGE_PROMPT = """\
You are evaluating a RAG system's answer to an in-scope question.

QUESTION:
{question}

REFERENCE ANSWER (extracted from the source document):
{gold}

MODEL'S RESPONSE:
{response}

Make two binary judgments:
1. CORRECT — does the model's response contain the same key information as the reference?
   - Yes if the response gets the same facts/numbers/dates right (even if phrased differently)
   - No if the response misses, contradicts, or hallucinates the key information
2. REFUSED — did the model decline to answer or fail to engage with the question?
   - Yes if the model said it can't/won't answer, said the info isn't available, or evaded
   - No if the model attempted a substantive answer

Output a single JSON object on one line, no other text:
{{"correct": true|false, "refused": true|false}}"""


# V2 prompt — adapted from MT-Bench reference-mode (Zheng et al. 2023, NeurIPS 2023).
# Changes vs V1:
#  - Splits the conflated "CORRECT" binary into two narrower criteria:
#      factual_consistent (no contradictions) ∧ covers_key_facts (no missing core content).
#    Cross-judge agreement on V1 collapsed because GPT-4o read "any missed detail" as NO
#    while Claude read "core fact present" as YES; this split lets them disagree on the
#    contributing criterion without dragging the headline number around.
#  - Drops the "(extracted from the source document)" qualifier on REFERENCE ANSWER.
#    Accurate for ELOQ (gold IS one extractable span) but wrong for RAGAS (gold is a
#    RAGAS-generated synthesis). On RAGAS the V1 framing biased strict judges toward
#    NO because the response paraphrased rather than extracted.
#  - Requires a brief CoT reasoning before the JSON verdict, in line with MT-Bench
#    and G-Eval findings that explanation improves judge consistency.
#
# Downstream `correct` = factual_consistent AND covers_key_facts.
JUDGE_PROMPT_V2 = """\
You are an impartial judge evaluating an AI assistant's answer to a question. You will be given the question, a reference answer, and the assistant's response.

QUESTION:
{question}

REFERENCE ANSWER:
{gold}

ASSISTANT'S RESPONSE:
{response}

Compare the response to the reference. Make three judgments.

1. FACTUAL_CONSISTENT. Does the response avoid contradicting the reference? True if every claim in the response that overlaps with the reference is consistent with it (paraphrase is fine). False if the response contradicts the reference on any fact (wrong number, wrong name, wrong date, wrong relationship).

2. COVERS_KEY_FACTS. Does the response convey the central content of the reference? True if the response contains the core facts, entities, and conclusions of the reference (rewording is fine). False if material the reference treats as central is missing.

3. REFUSED. Did the assistant decline or fail to engage with the question? True if it said it cannot or will not answer, said the information is unavailable, or evaded. False if it attempted a substantive answer.

First write a brief reasoning (one or two sentences). Then output exactly one JSON object on the final line, with no other text after it:
{{"reasoning": "...", "factual_consistent": true|false, "covers_key_facts": true|false, "refused": true|false}}"""


@dataclass
class InScopeJudge:
    correct: bool
    refused: bool
    raw: str = ""
    error: str = ""


@dataclass
class InScopeJudgeV2:
    factual_consistent: bool
    covers_key_facts: bool
    refused: bool
    reasoning: str = ""
    raw: str = ""
    error: str = ""

    @property
    def correct(self) -> bool:
        """V1-compatible derived field: response is 'correct' iff it neither
        contradicts the reference nor omits its core content."""
        return self.factual_consistent and self.covers_key_facts


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _parse_v2(raw: str) -> dict:
    """V2 responses include CoT before the JSON. Find the last JSON object."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Find the last `{...}` block in the response
    end = text.rfind("}")
    if end == -1:
        raise json.JSONDecodeError("no closing brace", text, 0)
    # Walk back to its matching opening brace
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return json.loads(text[i:end+1])
    raise json.JSONDecodeError("unmatched braces", text, 0)


def judge_one(question: str, response: str, gold: str,
              model: str | None = None) -> InScopeJudge:
    """Run the V1 LLM-judge on one row.

    `model` selects the judge model. If None, the env-configured JUDGE_MODEL
    (default Llama-3.3-70B on EPFL endpoint) is used. Pass an OpenRouter
    model id (e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4.5") for
    cross-judge validation runs.
    """
    prompt = JUDGE_PROMPT.format(question=question, response=response, gold=gold)
    try:
        raw = complete([{"role": "user", "content": prompt}], max_tokens=80,
                       model=model)
        parsed = _parse(raw)
        return InScopeJudge(
            correct=bool(parsed.get("correct", False)),
            refused=bool(parsed.get("refused", False)),
            raw=raw,
        )
    except json.JSONDecodeError:
        return InScopeJudge(correct=False, refused=False, raw=raw,
                            error="JSON parse failure")
    except Exception as e:
        return InScopeJudge(correct=False, refused=False, error=str(e))


def judge_one_v2(question: str, response: str, gold: str,
                 model: str | None = None) -> InScopeJudgeV2:
    """V2 judge: factual_consistent ∧ covers_key_facts ∧ refused, with CoT.

    Designed for cross-judge robustness. See JUDGE_PROMPT_V2 docstring for
    the rationale. `correct` (V1-compatible) is derived as
    factual_consistent ∧ covers_key_facts.
    """
    prompt = JUDGE_PROMPT_V2.format(question=question, response=response, gold=gold)
    try:
        raw = complete([{"role": "user", "content": prompt}], max_tokens=300,
                       model=model)
        parsed = _parse_v2(raw)
        return InScopeJudgeV2(
            factual_consistent=bool(parsed.get("factual_consistent", False)),
            covers_key_facts=bool(parsed.get("covers_key_facts", False)),
            refused=bool(parsed.get("refused", False)),
            reasoning=parsed.get("reasoning", "") or "",
            raw=raw,
        )
    except json.JSONDecodeError:
        return InScopeJudgeV2(factual_consistent=False, covers_key_facts=False,
                              refused=False, raw=raw,
                              error="JSON parse failure")
    except Exception as e:
        return InScopeJudgeV2(factual_consistent=False, covers_key_facts=False,
                              refused=False, error=str(e))
