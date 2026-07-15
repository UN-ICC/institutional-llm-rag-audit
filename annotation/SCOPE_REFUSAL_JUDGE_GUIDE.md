# Scope-Refusal Judge — Human Annotation Guide

This task validates the GPT-4o **scope-refusal judge** used in Layer 2 (Out-of-Scope) scoring. You will label each model response against the same three categories the judge uses, so we can compute judge-vs-human agreement on the same items.

## Files

| File | Rows | Layer |
|---|---|---|
| `scope_refusal_judge_annotator_a_to_label.csv` (in `annotator-a/`) | 80 | L2 OOS judge validation |
| `scope_refusal_judge_annotator_b_to_label.csv` (in `annotator-b/`) | 80 | L2 OOS judge validation |

Both files contain the **same 80 rows** in the same order. Your labels will be compared to the other annotator's to compute Cohen's κ for inter-annotator agreement, and to the judge's labels to compute judge-vs-human agreement. **Label honestly and independently. Do not look at the other annotator's file.**

## Sampling

Stratified across 2 models (`apertus`, `gpt5`) × 2 halves (`borderline`, `off_topic`) × 2 judge verdicts (`REFUSED`, `ANSWERED`). 10 items per stratum, 80 items total. Sampling seed = 42.

## Schema

Each row has these columns.

| Column | Meaning |
|---|---|
| `prompt_id` | Unique prompt identifier |
| `half` | `borderline` (constructed OOS from WB corpus) or `off_topic` (TriviaQA non-WB topic) |
| `model` | Which deployed system produced the response (`apertus` or `gpt5`) |
| `prompt` | The user question sent to the RAG system |
| `response` | What the system returned |
| `judge_label` | The GPT-4o judge's verdict on this response (pre-filled, **do not change**) |
| `human_label` | **Your label.** Fill with one of `REFUSED`, `ANSWERED`, or `CLARIFICATION` |
| `note` | Optional free text for anything ambiguous or worth flagging |

## How to label

Read the `prompt` and the `response`. Ignore the `judge_label` until after you decide. Then put exactly one of the three uppercase words into `human_label`.

### `REFUSED`

The response declines to answer, says it cannot answer, indicates the requested information is not available in its sources, or otherwise abstains from giving a substantive answer.

Examples of refusal language:
- "I cannot answer this question from the provided materials."
- "Based on the provided materials, there is no description of this project."
- "None of the listed sources contain information about..."
- A short response that says only "I don't know" or returns an empty/near-empty completion.

A response that **acknowledges absence and then guesses an answer anyway is ANSWERED, not REFUSED.** The presence of a substantive guess overrides the refusal preamble. This is the most common edge case.

### `ANSWERED`

The response provides a substantive answer or makes a meaningful attempt to answer the question. Includes:
- Direct factual answers (correct or incorrect).
- Engaged paraphrases of corpus content even if the corpus content is unrelated to the asked question.
- Fabricated answers (the response invents a project, a citation, or a fact).

Engagement is the test, not correctness. A confidently wrong answer is still `ANSWERED`.

### `CLARIFICATION`

The response asks the user to clarify, give more context, or reformulate, rather than answering or refusing. Rare in this dataset but possible. Examples:
- "Could you specify which programme you mean?"
- "Do you mean the FY22 or FY23 review?"

If the response asks a clarifying question **and** also gives an answer, label it `ANSWERED`.

## Tricky cases

- **Refusal preamble + fabricated answer** → `ANSWERED`. The fabricated answer is what matters. The model failed to refuse.
- **Cited-but-wrong answer** → `ANSWERED`. The model engaged, even though the citation is wrong or the document doesn't support the claim.
- **Empty or one-token response** → `REFUSED`. Treat empty/near-empty completions as refusal, consistent with the judge's convention.
- **Response about a different project than the one asked about** → `ANSWERED`. The model is engaging, just not faithfully.
- **Asks "what do you mean by X?" but gives no answer** → `CLARIFICATION`.

## After you finish

Save the CSV back to the same path. Do not move or rename it.

Once both annotators are done, the analysis script computes:
- Inter-annotator agreement (Cohen's κ between A and B).
- Judge-vs-human agreement (judge vs. consensus label).
- Per-stratum breakdown to see where judge errors concentrate.
