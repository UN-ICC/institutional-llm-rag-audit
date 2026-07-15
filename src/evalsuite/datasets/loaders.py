"""Benchmark dataset loaders for red-teaming evaluation.

Loads prompts from official benchmark datasets and normalizes them
to a list of TestCase objects.

Benchmark sources:
  - HarmBench (Mazeika et al., 2024)
    https://github.com/centerforaisafety/HarmBench
  - JailbreakBench (Chao et al., 2024)
    https://github.com/JailbreakBench/jailbreakbench
    Downloaded via: python -m evalsuite.datasets.download_jailbreakbench
  - StrongREJECT (Souly et al., 2024)
    https://github.com/alexandrasouly/strongreject

All benchmark data files are expected under data/benchmarks/.
"""

from __future__ import annotations

import csv
from pathlib import Path

from evalsuite.types import TestCase


DATA_DIR = Path("data/benchmarks")


# ── HarmBench ────────────────────────────────────────────────────────

def load_harmbench(path: str | Path | None = None) -> list[TestCase]:
    """Load HarmBench text behaviors (test split).

    Includes all behaviors: standard, contextual, and copyright.
    The RAG should refuse all of these since none are related to
    World Bank documents.

    For contextual behaviors, prepends ContextString to prompt text.

    Source: https://github.com/centerforaisafety/HarmBench
    Citation: Mazeika et al., "HarmBench: A Standardized Evaluation
              Framework for Automated Red Teaming and Refusal Training", 2024.
    """
    path = Path(path) if path else DATA_DIR / "harmbench_behaviors_text_test.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return []

    cases = []
    idx = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            func_cat = row.get("FunctionalCategory", "")

            behavior = row["Behavior"].strip()
            context_string = row.get("ContextString", "").strip()
            behavior_id = row.get("BehaviorID", "").strip()

            if func_cat == "contextual" and context_string:
                prompt = f"{context_string}\n\n{behavior}"
            else:
                prompt = behavior

            metadata = {}
            if behavior_id:
                metadata["behavior_id"] = behavior_id
            if context_string:
                metadata["context_string"] = context_string
            metadata["behavior"] = behavior
            metadata["functional_category"] = func_cat

            cases.append(TestCase(
                id=f"harmbench_{idx:03d}",
                prompt=prompt,
                source="harmbench",
                category=row.get("SemanticCategory", ""),
                expected_behavior="refuse",
                metadata=metadata,
            ))
            idx += 1
    return cases


# ── JailbreakBench ───────────────────────────────────────────────────

def load_jailbreakbench(path: str | Path | None = None) -> list[TestCase]:
    """Load JailbreakBench harmful behaviors.

    Requires downloading first:
      python -m evalsuite.datasets.download_jailbreakbench

    Source: https://github.com/JailbreakBench/jailbreakbench
    Citation: Chao et al., "JailbreakBench: An Open Robustness Benchmark
              for Jailbreaking Large Language Models", 2024.
    """
    path = Path(path) if path else DATA_DIR / "jailbreakbench_harmful.csv"
    if not path.exists():
        print(f"  [skip] {path} not found (run: python -m evalsuite.datasets.download_jailbreakbench)")
        return []

    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            metadata = {}
            target = row.get("Target", "").strip()
            if target:
                metadata["target"] = target
            behavior_id = row.get("Index", row.get("BehaviorID", "")).strip()
            if behavior_id:
                metadata["behavior_id"] = behavior_id

            cases.append(TestCase(
                id=f"jailbreakbench_{i:03d}",
                prompt=row.get("Goal", "").strip(),
                source="jailbreakbench",
                category=row.get("Category", ""),
                expected_behavior="refuse",
                metadata=metadata,
            ))
    return cases


# ── StrongREJECT ─────────────────────────────────────────────────────

def load_strongreject(path: str | Path | None = None) -> list[TestCase]:
    """Load StrongREJECT forbidden prompts.

    Source: https://github.com/alexandrasouly/strongreject
    Citation: Souly et al., "A StrongREJECT for Empty Jailbreaks", 2024.
    """
    path = Path(path) if path else DATA_DIR / "strongreject_dataset.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return []

    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            cases.append(TestCase(
                id=f"strongreject_{i:03d}",
                prompt=row["forbidden_prompt"].strip(),
                source="strongreject",
                category=row.get("category", ""),
                expected_behavior="refuse",
            ))
    return cases


# ── Unified loader ───────────────────────────────────────────────────

def load_all_jailbreaking(data_dir: str | Path | None = None) -> list[TestCase]:
    """Load all jailbreaking benchmark sources.

    Returns combined list: HarmBench + JailbreakBench + StrongREJECT.
    (~733 prompts total)
    """
    if data_dir:
        global DATA_DIR
        DATA_DIR = Path(data_dir)

    all_cases: list[TestCase] = []

    print("Loading HarmBench...")
    hb = load_harmbench()
    print(f"  -> {len(hb)} prompts")
    all_cases.extend(hb)

    print("Loading JailbreakBench...")
    jbb = load_jailbreakbench()
    print(f"  -> {len(jbb)} prompts")
    all_cases.extend(jbb)

    print("Loading StrongREJECT...")
    sr = load_strongreject()
    print(f"  -> {len(sr)} prompts")
    all_cases.extend(sr)


# ── Layer 1-B (scope refusal) — JSONL inputs ─────────────────────────

def _load_jsonl_as_testcases(
    path: str | Path,
    source: str,
    expected_behavior: str,
) -> list[TestCase]:
    """Generic loader: JSONL with prompt_id + text → TestCase list.

    Used by the in-scope ELOQ, in-scope RAGAS, and OOS datasets — they
    all share the same prompt_id/text/doc_id/category schema. Extra
    fields (doc_id, doc_title, derived_answer, etc.) flow through via
    the metadata dict so downstream evaluators keep them.
    """
    import json
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {p} not found")
        return []
    cases: list[TestCase] = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cases.append(TestCase(
            id=r["prompt_id"],
            prompt=r.get("text") or r.get("prompt") or "",
            source=source,
            category=r.get("category", ""),
            expected_behavior=expected_behavior,
            metadata={k: v for k, v in r.items()
                      if k not in {"prompt_id", "text", "category"}},
        ))
    return cases


def load_in_scope_eloq(path: str | Path | None = None,
                       sample: bool = False) -> list[TestCase]:
    """Load in-scope ELOQ prompts (Layer 1-B, FRR-control half).
    Defaults to the full final dataset; pass sample=True for the 50-row
    annotation sample."""
    if path is None:
        fname = "in-scope-eloq-sample.jsonl" if sample else "in-scope-eloq-final.jsonl"
        path = Path("data") / fname
    return _load_jsonl_as_testcases(path, source="eloq", expected_behavior="comply")


def load_in_domain_ragas(path: str | Path | None = None,
                         sample: bool = False) -> list[TestCase]:
    """Load in-domain RAGAS prompts (feeds both Layer 1-B in-scope FRR and
    Layer 2-B hallucination Faithfulness — see generators/in_domain/ragas.py
    docstring for the unification rationale)."""
    if path is None:
        fname = "in-domain-ragas-sample.jsonl" if sample else "in-domain-ragas-final.jsonl"
        path = Path("data") / fname
    return _load_jsonl_as_testcases(path, source="ragas", expected_behavior="comply")


def load_oos(path: str | Path | None = None,
             sample: bool = False) -> list[TestCase]:
    """Load out-of-scope ELOQ prompts (Layer 1-B borderline OOS half —
    topic-relevant questions whose specific fact isn't in the corpus)."""
    if path is None:
        fname = "out-of-scope-eloq-sample.jsonl" if sample else "out-of-scope-eloq-final.jsonl"
        path = Path("data") / fname
    return _load_jsonl_as_testcases(path, source="eloq-oos", expected_behavior="refuse")


def load_oos_offtopic(path: str | Path | None = None,
                      sample: bool = False) -> list[TestCase]:
    """Load off-topic OOS prompts (Layer 1-B off-topic half — questions
    wholly outside the WB domain, sampled from TriviaQA).

    Complements load_oos() which produces borderline OOS (topic-relevant,
    fact missing). Off-topic probes gross-domain refusal vs ELOQ's
    fine-grained scope discipline.
    """
    if path is None:
        fname = ("out-of-scope-offtopic-sample.jsonl" if sample
                 else "out-of-scope-offtopic-final.jsonl")
        path = Path("data") / fname
    return _load_jsonl_as_testcases(path, source="triviaqa-offtopic",
                                    expected_behavior="refuse")


def load_political_sycophancy(path: str | Path | None = None,
                              sample: bool = False) -> list[TestCase]:
    """Load political-sycophancy prompts (Layer 2-C: political-bias /
    false-premise sycophancy).

    Each evaluation item explodes into FOUR prompts (variants):
      - clean:  true premise, neutral framing — baseline
      - plain:  false premise, neutral framing — measures premise-sycophancy
      - load_a: false premise, pole-A (right-of-center) rhetorical framing
      - load_b: false premise, pole-B (left-of-center) rhetorical framing

    The four variants share an item_id and an invariant interrogative_clause.
    Generator: political-sycophancy v8 pipeline (Opus B+C, Haiku A/A2/D2/D3,
    Sonnet/Opus D1). Source corpus: World Bank analytical reports (CEM, PER,
    PA, SCD, WDR, CPF). Axes: 6 MARPOR v5 paired political-bias dimensions.

    Expected behavior: "comply" — the model should answer the question while
    correcting the false premise in non-CLEAN variants. The sycophancy scorer
    measures (a) premise-correction rate and (b) asymmetry between LOAD_A
    and LOAD_B answers (the political-bias signal).
    """
    if path is None:
        fname = ("political-sycophancy-sample.jsonl" if sample
                 else "political-sycophancy-final.jsonl")
        path = Path("data") / fname
    return _load_jsonl_as_testcases(path,
                                    source="political-sycophancy-v8",
                                    expected_behavior="comply")


def test_cases_to_csv(cases: list[TestCase], path: str | Path) -> None:
    """Export test cases to CSV in UNICC's expected format.

    Columns: system_prompt, prompt (matching runner.py expectations).
    """
    from evalsuite.datasets.io import save_csv

    rows = []
    for tc in cases:
        rows.append({
            "id": tc.id,
            "system_prompt": tc.system_prompt,
            "prompt": tc.prompt,
            "source": tc.source,
            "category": tc.category,
            "expected_behavior": tc.expected_behavior,
        })
    save_csv(rows, path)
    print(f"Exported {len(rows)} test cases to {path}")
