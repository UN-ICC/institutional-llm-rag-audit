## Evalsuite Red-Teaming Layout

End-to-end evaluation suite for running red-teaming style benchmarks (HarmBench, JailbreakBench, scope QA, hallucination, sycophancy, poisoning) against your model endpoint and aggregating metrics.

---

### Project Structure

```
.
├── README.md
├── requirements.txt
├── annotation/                  # human annotation configs/exports
├── data/                        # benchmark & probe datasets
├── results/                     # aggregated eval results
├── outputs/
│   └── runs/                    # per-run artifacts (responses, scores, summaries)
│
├── scripts/                     # standalone pipeline scripts
│   ├── score_harmbench.py           # HarmBench scoring
│   ├── score_jailbreakbench.py      # JailbreakBench scoring
│   ├── score_oos.py                 # out-of-scope refusal scoring
│   ├── score_inscope_eloq.py        # in-scope ELOQ hallucination scoring
│   ├── score_in_domain_ragas.py     # RAGAS faithfulness scoring
│   ├── score_political_sycophancy.py # political-framing sycophancy scoring
│   ├── score_multijudge.py          # cross-judge aggregation
│   ├── score_strongreject.py        # StrongREJECT scoring
│   ├── run_political_sycophancy_gpt5.py
│   ├── build_annotation_viewer.py   # + build_1a_viewers.sh, build_oos_viewers.sh, etc.
│   ├── plot_1a_asr.py               # + plot_1a_categories.py, plot_inscope_eloq.py
│   ├── sample_for_annotation.py     # + sample_1a_for_annotation.py, sample_inscope_eloq_for_annotation.py
│   ├── compute_1a_kappa.py          # inter-annotator agreement
│   ├── compute_engaged_incorrect.py
│   └── ...                          # promote/regen/filter helpers for RAGAS & OOS pipelines
│
└── src/
    └── evalsuite/                # core evaluation package
        ├── __init__.py
        ├── __main__.py            # `python -m evalsuite ...` entrypoint
        ├── cli.py
        ├── _io.py                 # JSONL/CSV read-write helpers
        ├── types.py                # core dataclasses (TestCase, RunResult, etc.)
        ├── clients/                # RAG endpoint + HTTP client wrappers
        ├── corpus/                 # RAG corpus loading/handling
        ├── datasets/                # dataset loaders (HarmBench, JailbreakBench, etc.)
        ├── generators/              # probe/dataset generators per layer
        ├── runners/                 # evaluation loop + report aggregation
        └── scorers/                 # scoring logic per layer
```

---

### Installing Dependencies

```bash
pip install -r requirements.txt
```

---

### Setup: input data

**Note on data access.** The public benchmark datasets (HarmBench, JailbreakBench, StrongREJECT) are freely available from their original sources, as described below. The World Bank document corpus used for Layers 1-B, 2-B, and 2-C is not published in this repository. Constructed scenarios, model outputs, and generated results (`results/`, `outputs/runs/`) are available on request only. To request access to the corpus and/or run outputs, contact UNICC AI Hub: aihub@unicc.org.

#### World Bank corpus (Layers 1-B, 2-B, 2-C)

The corpus is consumed by the scope-refusal, hallucination, and sycophancy
generators. Pick **one** of the two paths below.

**Option A — supplied ZIP (1,628 docs, recommended)**
1. Place `worldbank.zip` (≈707 MB) at `data/worldbank.zip`
2. Extract:
   ```bash
   python -m evalsuite.corpus.extract
   # → data/worldbank_extracted/documents.jsonl  (1,628 docs)
   ```

**Option B — fetch from the World Bank API (Board Summaries only, ~187 docs)**
```bash
python -m evalsuite.datasets.download_world_bank_corpus
# → data/worldbank/{pdfs,texts,metadata.jsonl}
```
Filters to `docty=Board Summary, lang=English, 2020-01-01 → 2025-09-29`.
This is the corpus the Layer 2-B RAGAS run used.

#### Public benchmarks (Layer 1-A)

| Dataset | File the loader expects | How to obtain |
|---|---|---|
| HarmBench | `data/benchmarks/harmbench_behaviors_text_test.csv` | `huggingface-cli download walledai/HarmBench --repo-type dataset --include "standard/behaviors.csv" --local-dir data/benchmarks/` (then rename), **or** download `data/behavior_datasets/harmbench_behaviors_text_test.csv` from https://github.com/centerforaisafety/HarmBench |
| JailbreakBench | `data/benchmarks/jailbreakbench_{harmful,benign}.csv` | `python scripts/download_jailbreakbench.py` (uses `JailbreakBench/JBB-Behaviors` on HF) |
| StrongREJECT | `data/benchmarks/strongreject_dataset.csv` | `huggingface-cli download walledai/StrongREJECT --repo-type dataset --local-dir data/benchmarks/`, **or** download `data/strongreject_dataset.csv` from https://github.com/alexandrasouly/strongreject |

`data/benchmarks/SOURCES.md` tracks the canonical citation for each dataset.

---

### Running Prompts Through Apertus (RAG endpoint)

To send our generated prompt datasets to UNICC's Apertus RAG endpoint and capture the responses, use the existing run-phase entry point of `run_layer`:

```bash
python -m evalsuite.runners.run_layer --layer <code> --phase run [--sample]
```

The runner is **resumable** (checkpoints to `outputs/runs/<run-id>/responses.jsonl` after every response), uses the same `RAGClient` (`evalsuite.clients.rag_client`) that talks to UNICC's `/chat` API, and writes both `responses.jsonl` and a CSV mirror.

#### One-time setup

```bash
pip install -r requirements.txt
echo 'CHAT_ENDPOINT=https://<UNICC-supplied-host>/chat' >> .env
```

#### Layer codes

| Code | Loader | Final size | Sample size |
|---|---|---|---|
| `1a` | HarmBench + JailbreakBench + StrongREJECT (combined) | ~733 | — |
| `1b-in-scope-eloq` | `data/in-scope-eloq-{final,sample}.jsonl` | 189 | 50 |
| `1b-in-domain-ragas` | `data/in-domain-ragas-{final,sample}.jsonl` | (regenerating) | 50 |
| `1b-oos` | `data/out-of-scope-eloq-{final,sample}.jsonl` | (regenerating) | 50 |

The `1b-in-domain-ragas` dataset is the unified RAGAS dataset — same prompts feed both the 1-B in-scope FRR scorer and the 2-B hallucination Faithfulness scorer (the layer code just selects which scorer runs).

#### Run the three annotation samples (50 prompts each, ~5 min total)

```bash
python -m evalsuite.runners.run_layer --layer 1b-in-scope-eloq  --phase run --sample
python -m evalsuite.runners.run_layer --layer 1b-in-domain-ragas --phase run --sample
python -m evalsuite.runners.run_layer --layer 1b-oos            --phase run --sample
```

Outputs land in `outputs/runs/<YYYY-MM-DD>_<layer>_sample/responses.jsonl`. Drop `--sample` to run the full final datasets.

Adding a new layer is one entry in `LAYER_LOADERS` in `runners/run_layer.py` plus a loader function in `datasets/loaders.py` — no other changes needed.

---

### Basic Usage (`python -m evalsuite`)

At a high level, you:

1. **Configure** your model endpoint and dataset in `configs/local.yaml` or `configs/prod.yaml`.
2. **Run** the evaluation loop with the CLI.
3. **Inspect** the run directory under `outputs/runs/`.

Example (placeholder; adjust once CLI is implemented):

```bash
python -m evalsuite \
  --config configs/local.yaml \
  --benchmark la_harmbench.csv \
  --run-id 2026-03-83_1a_harmbench
```

This will:

- load test cases from `data/benchmarks/…` via `datasets/loaders.py`
- call your RAG / model endpoint through `clients/rag_client.py` or `clients/http_client.py`
- score responses with the modules in `scorers/`
- write `responses.csv`, `scores.csv`, and `summary.csv` under `outputs/runs/<run-id>/`

---

### Datasets & Formats

- **CSV benchmarks** in `data/benchmarks/` are the canonical inputs for canned suites (HarmBench, JailbreakBench-style, etc.).
- **JSONL formats** are handled via `datasets/io.py` (for custom or generated test cases).

You can extend or replace loaders in `datasets/loaders.py` to plug in new benchmarks.

---

### Extending the Suite

- **New client** → add a module under `clients/` and wire it into the CLI.
- **New scorer** → add a file under `scorers/` and register it in `run_layer.py`.
- **New generator** → add a module under `generators/` that writes JSONL/CSV into `data/benchmarks/`.

The goal is to keep the core flow—*load test cases → call model → score → report*—simple while allowing you to swap in your own infrastructure as needed.
