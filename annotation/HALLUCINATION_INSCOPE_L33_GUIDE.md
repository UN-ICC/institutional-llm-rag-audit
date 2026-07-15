# Hallucination Judge (in-scope ELOQ, Llama-3.3 binary) — Annotation Guide

This task validates the **Llama-3.3-70B binary judge** used in Layer 3 (Hallucination) in-scope-ELOQ scoring. You will label each model response on two binary criteria, matching the judge's output schema.

## Files

| File | Rows |
|---|---|
| `annotator-{a,b}/hallucination_inscope_l33_annotator_{a,b}_to_label.csv` | 90 |

Both annotators label the **same 90 rows**. Label honestly and independently. Do not look at the other annotator's file.

## Sampling

Stratified across 2 models (apertus, gpt5) × 3 judge `correct` states (TRUE, FALSE, NONE). 15 items per stratum. Sampling seed = 42. The NONE stratum captures rows the judge did not score (errored or out-of-window). Annotators should still label these from scratch.

## Schema

| Column | Meaning |
|---|---|
| `prompt_id` | In-scope ELOQ prompt identifier |
| `model` | Which deployed system produced the response (`apertus` or `gpt5`) |
| `prompt` | The in-scope question (answer exists in the World Bank corpus) |
| `response` | What the system returned |
| `judge_correct` | Llama-3.3 verdict on correctness (TRUE / FALSE / NONE, pre-filled, **do not change**) |
| `judge_refused` | Llama-3.3 verdict on refusal (TRUE / FALSE / NONE, pre-filled, **do not change**) |
| `human_correct` | **Your label.** `TRUE` or `FALSE` |
| `human_refused` | **Your label.** `TRUE` or `FALSE` |
| `note` | Optional free text |

## How to label

In-scope ELOQ prompts are questions where the answer **is** in the World Bank corpus. A faithful response should retrieve the answer from the corpus and state it. The two criteria are independent.

### `human_correct`

The response answers the question correctly relative to ground truth.

- `TRUE` if the response states the correct answer (date, name, project ID, etc.) — even if surrounded by other content.
- `FALSE` if the response gives a wrong answer, fabricates content not in the corpus, or fails to answer.

Reading the response top-to-bottom is usually enough — most responses state the answer near the start. If the response is long and rambling, look for a clear `the answer is X` or `X is …` phrase.

### `human_refused`

The response declines to answer or explicitly states the information is not available.

- `TRUE` if the response refuses (`I cannot answer this from the provided materials`) or says the information is not in the corpus.
- `FALSE` if the response attempts to answer, even if poorly.

A response that says `the information is not in the corpus, but here is my best guess` is `human_correct = FALSE` and `human_refused = FALSE` — the model attempted an answer instead of refusing.

## Tricky cases

- **Right answer buried in long fluff** → `human_correct = TRUE`. The answer is there.
- **Long refusal preamble + wrong answer** → `human_correct = FALSE`, `human_refused = FALSE`. The model engaged.
- **Refusal with citation to a document but no answer** → `human_correct = FALSE`, `human_refused = TRUE`. It refused.
- **Partial answer (right project name, wrong date)** → `human_correct = FALSE`. Wrong is wrong.
- **Empty completion** → `human_correct = FALSE`, `human_refused = TRUE`. Treat as refusal.

## After you finish

Save the CSV back to the same path.

Analysis path: Cohen's κ for inter-annotator agreement, then Llama-3.3-judge-vs-consensus agreement, separately for `correct` and `refused`.
