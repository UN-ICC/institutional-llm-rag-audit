"""Out-of-scope question generator — paper-faithful ELOQ.

Pipeline (per document, exactly Algorithm 1 of the ELOQ paper):
  1. extract_claims (numbered list of N facts)
  2. hallucinate: 3 rounds × 3 mask-pattern phases = 9 mask-impute iterations
  3. filter: drop hallucinated claims accidentally still supported by doc.
     Output preserves ORIGINAL claim indices per the ELOQ REMOVE prompt.
  4. ONE bulk LLM call: (doc, all_surviving_hallucinated_facts) → list[Q].
     Verbatim ELOQ `user_conf` prompt template.

We then **align the output questions positionally** to the surviving
hallucinated-fact list to recover (original_claim, hallucinated_claim, Q)
triples per row — the paper doesn't store this lineage but it's free given
positional alignment, and it makes annotation/debug far easier.

If the count of questions returned ≠ count of surviving facts, we still
produce best-effort triples up to the shorter length and log a warning.

Both branches share `extract_claims`, the LLM client, and the system prompt
in `_common.py`.

Run:
  python -m evalsuite.generators.out_of_scope.eloq --num-docs 200 \\
    --corpus-dir data/worldbank-zip

Reference:
  Peng et al., "ELOQ: Resources for Enhancing LLM Detection of
  Out-of-Scope Questions", SIGIR 2025. arXiv:2410.14567
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from evalsuite._io import read_jsonl, append_jsonl, save_jsonl
from evalsuite.corpus.extract import load_corpus
from evalsuite.generators._common import (
    DEFAULT_MODEL,
    MAX_DOC_CHARS,
    SYSTEM_PROMPT,
    enum_list,
    llm_call,
    parse_numbered_items,
)


# ── Configuration ───────────────────────────────────────────────────

GENERATOR_VERSION = "out_of_scope_eloq_v1_bulk_aligned"
DEFAULT_NUM_FACTS = 6
DEFAULT_NUM_DOCS = 50
DEFAULT_OUTPUT_DIR = Path("data/out-of-scope/eloq")
DEFAULT_CORPUS_DIR = Path("data/worldbank")


# ── Prompts (verbatim from upstream ELOQ except for the OOS Q prompt) ──

# Claim extraction: only consumed by this OOS pipeline (in-scope ELOQ feeds
# the whole doc to the LLM, no claim extraction step).
REDUCE_PROMPT = (
    "Read the document and list {num_fact} most important facts it contains. "
    "Each fact should be stated in a clear, standalone sentence with sufficient "
    "context to be understood independently, avoiding undefined pronouns. "
    "Ensure that each fact is directly derived from the document and does not "
    "include any information not mentioned within it.\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "{num_fact} most important facts:"
)


def extract_claims(document: str, num_fact: int = DEFAULT_NUM_FACTS,
                   model: str = DEFAULT_MODEL) -> str:
    """Return the LLM-generated numbered list of claims (raw text)."""
    return llm_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REDUCE_PROMPT.format(
                document=document[:MAX_DOC_CHARS], num_fact=num_fact)},
        ],
        model=model,
    )


MODIFY_PROMPT = (
    "Read the document below with a list of {num_fact} facts it contains. "
    "Note that some of the facts are missing represented by (missing). "
    "Your task is to guess the missing facts could have said and complete "
    "the missing facts. Each fact should be stated in a clear, standalone "
    "sentence with sufficient context to be understood independently, "
    "avoiding undefined pronouns. Please complete the missing facts and "
    "return all the {num_fact} facts in original order. "
    "You must return all the {num_fact} facts.\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "Completed list of facts:"
)

REMOVE_PROMPT = (
    "Read the document below with a list of {num_true_fact} ground-truth facts "
    "it contains and a list of {num_false_fact} hallucinated facts that are not "
    "supported by the document. Your task is to remove any hallucinated facts "
    "that can be supported by either the document or the {num_true_fact} "
    "ground-truth facts. Please only return the remaining hallucinated facts, "
    "along with their original order numbers.\n"
    'Document:\n\n"""{document}"""\n\n'
    "{num_true_fact} ground-truth facts:\n\n{ori_facts}\n\n"
    "{num_false_fact} hallucinated facts\n\n{hallucinated_facts}\n\n"
    "Remaining hallucinated facts:"
)

# Adapted from ELOQ `user_conf` template (bulk: one call per doc) plus an
# instruction that questions must be emitted with the SAME INDEX as their
# source fact. This makes per-question lineage robust to LLM skips/reorders.
# The original ELOQ prompt is otherwise verbatim.
OOS_Q_PROMPT = (
    "Read the document and review the list of hallucinated facts. For each "
    "hallucinated fact, craft a single, specific and concise question "
    "containing 13 to 18 words that incorporates the key element of the fact, "
    "ensuring the question is intentionally confusing. The question should not "
    "be answerable using any information present in the document. The question "
    "should not combine multiple queries and each question should address only "
    "one specific aspect. If a question cannot be formulated for a particular "
    "hallucinated fact, you may omit it.\n\n"
    "CRITICAL — Every question MUST include at least one explicit named "
    "entity (project name, organisation name, programme acronym, date, "
    "country, person, etc.) so a reader who has not seen the source "
    "document can still identify what the question is about. NEVER use "
    "bare references like 'the meeting', 'the inquiry', 'the budget', "
    "'the committee', 'the project', 'the report' — always anchor with "
    "the specific name or date.\n\n"
    "EXAMPLES:\n"
    "  Bad:  'Were follow-up meetings scheduled to ensure monitoring of progress?'\n"
    "    (which meetings, which progress — unanchored)\n"
    "  Bad:  'What measures were implemented to cushion inflation effects?'\n"
    "    (which government, which measures — no named entity)\n"
    "  Good: 'Did the Audit Committee endorse the IFC Information Statement "
    "on October 5, 2024?'\n"
    "    (named: Audit Committee, IFC, date)\n"
    "  Good: 'How much funding did the IDA20 replenishment dedicate to "
    "Cameroon's vaccine programme in March 2023?'\n"
    "    (named: IDA20, Cameroon, date)\n\n"
    "IMPORTANT — Output format: emit each question on its own line as "
    "`<index>. <question>` where `<index>` is the SAME index number "
    "shown next to the source hallucinated fact. If you skip a fact, "
    "do not emit a line for it (do not renumber).\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "hallucinated facts:\n\n{hallucinated_facts}\n\n"
    "Questions:"
)


# ── Hallucination injection ─────────────────────────────────────────

def _suppress_facts(text: str, suppress_fn) -> str:
    """Replace selected facts with (missing) — core of ELOQ masking."""
    items = parse_numbered_items(text)
    for i in range(len(items)):
        if suppress_fn(i):
            items[i] = "(missing)"
    return enum_list(items)


def _impute_facts(missing_facts_doc: str, num_fact: int, model: str) -> str:
    result = llm_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": MODIFY_PROMPT.format(
                document=missing_facts_doc, num_fact=num_fact)},
        ],
        model=model,
    )
    lines = result.splitlines()
    if lines and "list of facts" in lines[0].lower():
        result = "\n".join(lines[1:])
    return result


def hallucinate(
    document: str, reduced_doc: str, num_fact: int, model: str,
) -> str:
    """3 rounds × 3 mask-pattern phases = 9 suppress-impute iterations,
    then drop hallucinations accidentally supported by the original."""
    doc = reduced_doc
    for _round in range(3):
        doc = _suppress_facts(doc, lambda i: i % 3 == 2)
        doc = _impute_facts(doc, num_fact, model)
        doc = _suppress_facts(doc, lambda i: i % 3 == 1)
        doc = _impute_facts(doc, num_fact, model)
        doc = _suppress_facts(doc, lambda i: i % 3 == 0)
        doc = _impute_facts(doc, num_fact, model)

    return llm_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REMOVE_PROMPT.format(
                document=document[:MAX_DOC_CHARS],
                ori_facts=reduced_doc,
                hallucinated_facts=doc,
                num_true_fact=num_fact,
                num_false_fact=num_fact,
            )},
        ],
        model=model,
    )


def generate_questions(
    document: str, hallucinated_facts: str, model: str = DEFAULT_MODEL,
) -> list[tuple[int, str]]:
    """Bulk: one LLM call → list of (index, question) pairs.

    The prompt asks the LLM to emit each Q with the SAME index as the source
    fact, so we recover lineage by index lookup (robust to LLM skips/reorders)
    rather than positional zip."""
    raw = llm_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OOS_Q_PROMPT.format(
                document=document[:MAX_DOC_CHARS],
                hallucinated_facts=hallucinated_facts)},
        ],
        model=model,
    )
    return [(idx, q) for idx, q in parse_indexed_items(raw) if q.endswith("?")]


_INDEXED_RE = re.compile(r"^(\d+)[:.]\s+(.*)$")


def parse_indexed_items(text: str) -> list[tuple[int, str]]:
    """Parse `1. foo\n2. bar` → [(1, "foo"), (2, "bar")].
    Preserves original numbering — needed because the REMOVE filter step
    returns a SUBSET of the original list with original index numbers
    (per the ELOQ REMOVE prompt instruction)."""
    items: list[tuple[int, str]] = []
    chunks: list[str] = []
    cur_idx: int | None = None

    def flush():
        nonlocal chunks, cur_idx
        if cur_idx is not None and chunks:
            items.append((cur_idx, " ".join(chunks).strip()))
        chunks = []
        cur_idx = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _INDEXED_RE.match(line)
        if m:
            flush()
            cur_idx = int(m.group(1))
            line = m.group(2)
        chunks.append(line)
    flush()
    return items


# ── Pipeline ────────────────────────────────────────────────────────

@dataclass
class QClaim:
    """One generated OOS question with its full claim lineage."""
    claim_index: int
    original_claim: str
    hallucinated_claim: str
    question: str


@dataclass
class DocResult:
    doc_id: str
    title: str
    reduced_claims: str = ""
    hallucinated_facts: str = ""
    triples: list[QClaim] = field(default_factory=list)  # one per surviving claim
    error: str = ""


def process_document(
    doc: dict, num_fact: int = DEFAULT_NUM_FACTS, model: str = DEFAULT_MODEL,
) -> DocResult:
    doc_id = doc.get("doc_id", "unknown")
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    result = DocResult(doc_id=doc_id, title=title)
    try:
        # Stage 1: claims (numbered list)
        result.reduced_claims = extract_claims(text, num_fact, model)
        # Stage 2: hallucinate (returns numbered subset with ORIGINAL indices)
        result.hallucinated_facts = hallucinate(text, result.reduced_claims, num_fact, model)

        # Stage 3+4: ONE bulk LLM call → list of (index, question) pairs.
        indexed_questions = generate_questions(text, result.hallucinated_facts, model)

        # Recover lineage by INDEX lookup. Each Q's index matches its source
        # fact's index in the hallucinated-facts numbered list, which itself
        # preserves the ORIGINAL claim index from stage 1.
        orig_by_idx = dict(parse_indexed_items(result.reduced_claims))
        halluc_by_idx = dict(parse_indexed_items(result.hallucinated_facts))
        n_facts = len(halluc_by_idx)
        n_qs = len(indexed_questions)

        for q_idx, q in indexed_questions:
            halluc = halluc_by_idx.get(q_idx)
            if halluc is None:
                # LLM emitted an index we didn't have in input — drop with note
                print(f"  ⚠ {result.doc_id}: Q index {q_idx} not in hallucinated-fact list; skipping")
                continue
            result.triples.append(QClaim(
                claim_index=q_idx,
                original_claim=orig_by_idx.get(q_idx, ""),
                hallucinated_claim=halluc,
                question=q,
            ))

        if n_qs < n_facts:
            print(f"  · {result.doc_id}: {n_qs}/{n_facts} questions ({n_facts - n_qs} skipped by LLM)")
    except Exception as e:
        result.error = str(e)
    return result


MEETING_LOG_MARKERS = ("minutes of meeting", "minutes of a meeting",
                       "meeting of the", "committee", "minutes of")


def is_meeting_log(title: str) -> bool:
    """Heuristic: title contains a meeting-log marker.

    Annotator-B's review of v1 OOS showed 21/50 (42%) failed `clear_subject`
    and 15/20 of those came from meeting-log type docs (questions like
    'were follow-up meetings scheduled' inherit the doc's ambient context).
    Filtering these out at the source removes the bulk of unanchored
    questions before they're ever generated.
    """
    t = (title or "").lower()
    return any(m in t for m in MEETING_LOG_MARKERS)


def sample_documents(
    docs: list[dict], num_docs: int,
    min_chars: int = 1000, max_chars: int = 30000, seed: int = 42,
    exclude_meeting_logs: bool = True,
) -> list[dict]:
    pool = [d for d in docs if not (exclude_meeting_logs and is_meeting_log(d.get("title", "")))]
    eligible = [d for d in pool if min_chars <= d.get("char_count", 0) <= max_chars]
    if exclude_meeting_logs:
        print(f"  filtered out {len(docs) - len(pool)} meeting-log docs "
              f"({len(pool)} remaining after exclusion)")
    if len(eligible) <= num_docs:
        return eligible
    rng = random.Random(seed)
    return rng.sample(eligible, num_docs)


def generate_all(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_docs: int = DEFAULT_NUM_DOCS,
    num_fact: int = DEFAULT_NUM_FACTS,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.jsonl"
    prompts_path = output_dir / "prompts.jsonl"

    docs = load_corpus(corpus_dir)
    sampled = sample_documents(docs, num_docs=num_docs, seed=seed)
    print(f"Corpus: {len(docs)}; sampled {len(sampled)} (seed={seed})")

    done: dict[str, dict] = {r["doc_id"]: r for r in read_jsonl(ckpt_path)}
    if done:
        print(f"Checkpoint: {len(done)} docs already processed (resuming)")

    doc_results: list[DocResult] = []
    for rec in done.values():
        triples = [QClaim(**t) for t in rec.get("triples", [])]
        doc_results.append(DocResult(
            doc_id=rec["doc_id"], title=rec.get("title", ""),
            reduced_claims=rec.get("reduced_claims", ""),
            hallucinated_facts=rec.get("hallucinated_facts", ""),
            triples=triples,
            error=rec.get("error", ""),
        ))

    remaining = [d for d in sampled if d["doc_id"] not in done]
    print(f"Remaining: {len(remaining)} documents\n")

    for i, doc in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {doc['doc_id']}: {(doc.get('title') or '')[:70]}")
        result = process_document(doc, num_fact, model)
        doc_results.append(result)
        append_jsonl(
            {
                "doc_id": result.doc_id, "title": result.title,
                "reduced_claims": result.reduced_claims,
                "hallucinated_facts": result.hallucinated_facts,
                "triples": [t.__dict__ for t in result.triples],
                "error": result.error,
            },
            ckpt_path,
        )
        print(f"  → {len(parse_numbered_items(result.hallucinated_facts))} surviving hallucinated facts, "
              f"{len(result.triples)} OOS questions")

    prompts: list[dict] = []
    idx = 0
    for dr in doc_results:
        if dr.error:
            continue
        for t in dr.triples:
            prompts.append({
                "prompt_id": f"out_of_scope_{idx:04d}",
                "text": t.question,
                "source": "eloq",
                "category": "oos",
                "expected_behavior": "refuse",
                "doc_id": dr.doc_id,
                "doc_title": dr.title,
                "claim_index": t.claim_index,
                "original_claim": t.original_claim,
                "hallucinated_claim": t.hallucinated_claim,
                "generator_version": GENERATOR_VERSION,
                "generator_model": model,
            })
            idx += 1
    save_jsonl(prompts, prompts_path)

    n_err = sum(1 for dr in doc_results if dr.error)
    print(f"\n{'='*50}")
    print(f"Out-of-scope ELOQ generation")
    print(f"{'='*50}")
    print(f"Documents:  {len(doc_results)} processed ({n_err} errors)")
    print(f"Questions:  {len(prompts)}")
    print(f"Outputs:")
    print(f"  prompts:    {prompts_path}")
    print(f"  checkpoint: {ckpt_path}")
    return prompts


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Out-of-scope ELOQ question generator")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS)
    p.add_argument("--num-fact", type=int, default=DEFAULT_NUM_FACTS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate_all(args.corpus_dir, args.output_dir, args.num_docs, args.num_fact,
                 args.model, args.seed)


if __name__ == "__main__":
    main()
