"""World Bank corpus pipeline: fetch → download → extract → clean.

Fetches metadata from the World Bank WDS API, downloads matching PDFs,
extracts and cleans text. Resumable: re-runs skip docs whose row already
exists in documents.jsonl.

Pipeline:
  - pymupdf4llm extraction (multi-column merge, hyphenation, tables) with
    pdfplumber fallback for PDFs pymupdf can't open
  - WB-specific boilerplate ("Public Disclosure Authorized") removal
  - Per-document and per-page JSONL records (Markdown text)
  - Quality diagnostics (low-quality pages, extraction method per doc)
  - Retry with backoff on API and PDF downloads

Output (under <output_dir>, default data/worldbank/):
  pdfs/<guid>.pdf       cached PDF downloads (skip-if-exists)
  documents.jsonl       one row per doc: WB metadata + cleaned text + stats
  pages.jsonl           one row per page: cleaned + raw text
  metadata.json         run-level extraction statistics

Run:
  python -m evalsuite.corpus.extract                            # default filters
  python -m evalsuite.corpus.extract --doc-type "Board Summary" # restrict to one type
  python -m evalsuite.corpus.extract --limit 100                # smoke test
"""

from __future__ import annotations

from evalsuite._io import read_jsonl, append_jsonl

import argparse
import hashlib
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf4llm
import pdfplumber
import requests
from tqdm import tqdm


# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("data/worldbank")          # API mode default
DEFAULT_ZIP_OUTPUT_DIR = Path("data/worldbank-zip")  # zip mode default
DEFAULT_ZIP_PATH = Path("data/worldbank.zip")

WB_API_BASE = "https://search.worldbank.org/api/v3/wds"
WB_API_FIELDS = "pdfurl,display_title,docdt,docna,repnme,guid,docty"
DEFAULT_DATE_START = "2020-01-01"
DEFAULT_DATE_END = "2025-09-29"
DEFAULT_LIMIT = 2000
DEFAULT_PAGE_SIZE = 50

MIN_CHARS_THRESHOLD = 100   # fall back to pdfplumber if PyMuPDF yields fewer chars
LOW_QUALITY_THRESHOLD = 50  # flag pages with fewer chars after cleaning

HTTP_RETRIES = 4
HTTP_BACKOFF_BASE = 2  # seconds; doubled each attempt


# ── Text cleaning ───────────────────────────────────────────────────

# pymupdf4llm handles multi-column merging, hyphenation, soft line wraps,
# tables, and headers/footers. We only post-process to strip WB-specific
# boilerplate that no general-purpose PDF library would know about.
BOILERPLATE_PATTERNS = [
    re.compile(r"^Public Disclosure Authorized\s*$", re.MULTILINE),
    re.compile(r"^Disclosure Authorized\s*$", re.MULTILINE),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*-?\s*\d{1,3}\s*-?\s*$", re.MULTILINE),  # bare page numbers
    re.compile(r"^\s*FOR OFFICIAL USE ONLY\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*RESTRICTED DISTRIBUTION\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*CONFIDENTIAL\s*$", re.MULTILINE | re.IGNORECASE),
]
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Repeated-line detection: lines that appear on ≥ this fraction of pages are
# treated as running headers/footers and stripped corpus-wide for that doc.
REPEATED_LINE_PAGE_FRAC = 0.30
REPEATED_LINE_MIN_PAGES = 4  # don't trigger on tiny docs


def clean_page_text(text: str) -> str:
    """Strip World Bank-specific boilerplate. General PDF cleanup
    (hyphenation, multi-column, whitespace, tables) is delegated to
    pymupdf4llm and not duplicated here."""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Detect lines that recur on many pages (running headers/footers,
    doc-id stamps, copyright notices, etc.) and remove them everywhere.

    Triggered only when there are enough pages (REPEATED_LINE_MIN_PAGES)
    to make the heuristic reliable. A line counts as "repeated" if its
    normalized form appears on ≥REPEATED_LINE_PAGE_FRAC of pages and is
    short enough to plausibly be a header/footer (≤120 chars)."""
    if len(pages) < REPEATED_LINE_MIN_PAGES:
        return pages
    counts: Counter = Counter()
    for p in pages:
        for line in {ln.strip() for ln in p.splitlines() if 3 <= len(ln.strip()) <= 120}:
            counts[line] += 1
    threshold = max(2, int(len(pages) * REPEATED_LINE_PAGE_FRAC))
    repeated = {ln for ln, c in counts.items() if c >= threshold}
    if not repeated:
        return pages
    out = []
    for p in pages:
        kept = [ln for ln in p.splitlines() if ln.strip() not in repeated]
        out.append("\n".join(kept))
    return out


# ── Paragraph segmentation (downstream-facing) ──────────────────────

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_RE = re.compile(r"^\s*(?:[-*+•·]|\d+[.)])\s+\S")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*-{3,}.*$")
_ACRONYM_LINE_RE = re.compile(r"^[A-Z][A-Z0-9/&. ]{2,}$")  # all-caps acronym list line
_HAS_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")  # rough proper noun
_HAS_NUMBER_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?(?:\s*(?:%|percent|million|billion|trillion))?\b")
_HAS_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def iter_paragraphs(md_text: str, *, min_chars: int = 80):
    """Yield {text, kind, score} per paragraph from pymupdf4llm Markdown.

    `kind` ∈ {prose, heading, list, table, short, other}. `score` is a
    coarse 0-3 substantiveness signal (entity + number + year). Downstream
    paragraph filtering in PIPELINE.md §3c can pull only kind=='prose' with
    score≥2 for high-recall, claim-worthy paragraphs."""
    blocks = re.split(r"\n\s*\n", md_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if any(_TABLE_RE.match(ln) or _TABLE_SEP_RE.match(ln) for ln in lines):
            kind = "table"
        elif all(_HEADING_RE.match(ln) for ln in lines):
            kind = "heading"
        elif sum(1 for ln in lines if _LIST_RE.match(ln)) >= max(2, len(lines) // 2):
            kind = "list"
        elif len(block) < min_chars:
            kind = "short"
        elif sum(1 for ln in lines if _ACRONYM_LINE_RE.match(ln)) >= max(2, len(lines) // 2):
            kind = "other"
        else:
            kind = "prose"
        score = (
            int(bool(_HAS_ENTITY_RE.search(block)))
            + int(bool(_HAS_NUMBER_RE.search(block)))
            + int(bool(_HAS_YEAR_RE.search(block)))
        )
        yield {"text": block, "kind": kind, "score": score}


# ── HTTP with retry/backoff ─────────────────────────────────────────

def _http_get(url: str, params: dict | None = None, timeout: int = 60) -> requests.Response:
    """GET with exponential backoff. Retries on connection errors and 5xx."""
    last_exc: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError(f"server error {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
            if attempt + 1 < HTTP_RETRIES:
                wait = HTTP_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ── PDF extraction (pymupdf4llm + pdfplumber fallback) ──────────────

def extract_text_pymupdf4llm(pdf_path: Path) -> list[str]:
    """Use pymupdf4llm with page_chunks=True. Returns per-page Markdown."""
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        show_progress=False,
    )
    # chunks is a list of dicts with keys: 'metadata', 'text', 'tables', ...
    return [(c.get("text") or "") for c in chunks]


def extract_text_pdfplumber(pdf_path: Path) -> list[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def extract_pdf(pdf_path: Path) -> tuple[list[str], str]:
    """Return (per-page text, method). Tries pymupdf4llm first (handles
    multi-column, tables, hyphenation natively); falls back to pdfplumber
    only for PDFs pymupdf can't open."""
    try:
        pages = extract_text_pymupdf4llm(pdf_path)
        if sum(len(p) for p in pages) >= MIN_CHARS_THRESHOLD:
            return pages, "pymupdf4llm"
    except Exception:
        pass
    try:
        pages = extract_text_pdfplumber(pdf_path)
        return pages, "pdfplumber"
    except Exception:
        return [], "failed"


# ── World Bank API metadata fetch ───────────────────────────────────

def fetch_metadata(
    doc_type: str | list[str] | None = None,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    limit: int = DEFAULT_LIMIT,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict]:
    """Page through the WB WDS API and return metadata records.

    By default does NOT restrict document type — pulls everything in the
    date range so downstream stratified sampling has variety. Pass a
    `doc_type` (string or list of strings) to restrict; lists are fetched
    as separate API queries and the results merged with the global `limit`
    applied across all types."""
    params_base = {
        "format": "json",
        "lang_exact": "English",
        "srt": "docdt",
        "order": "desc",
        "strdate": date_start,
        "enddate": date_end,
        "fl": WB_API_FIELDS,
    }

    if isinstance(doc_type, list):
        doc_types: list[str | None] = [d for d in doc_type if d]
    elif doc_type:
        doc_types = [doc_type]
    else:
        doc_types = [None]

    records: list[dict] = []
    seen_guids: set[str] = set()
    for dt in doc_types:
        per_type_params = params_base.copy()
        if dt:
            per_type_params["docty_exact"] = dt
        per_type_limit = limit if len(doc_types) == 1 else limit  # global cap; we break early below
        for offset in range(0, per_type_limit, page_size):
            if len(records) >= limit:
                break
            params = per_type_params.copy()
            params["os"] = offset
            params["rows"] = min(page_size, per_type_limit - offset)
            r = _http_get(WB_API_BASE, params=params)
            docs = r.json().get("documents", {})
            if not docs:
                break
            for _, doc in docs.items():
                guid = doc.get("guid")
                if not guid or not doc.get("pdfurl") or guid in seen_guids:
                    continue  # dedup across types and across pages
                seen_guids.add(guid)
                records.append({
                    "guid": guid,
                    "title": doc.get("display_title"),
                    "date": doc.get("docdt"),
                    "doc_name": doc.get("docna"),
                    "report_name": doc.get("repnme"),
                    "docty": doc.get("docty"),
                    "pdf_url": doc.get("pdfurl"),
                })
                if len(records) >= limit:
                    break
        if len(records) >= limit:
            break
    return records[:limit]


def download_pdf(url: str, dest: Path) -> None:
    if dest.exists():
        return
    r = _http_get(url)
    dest.write_bytes(r.content)


# ── JSONL helpers (local, to keep this module dep-free) ─────────────


def _processed_doc_ids(documents_path: Path) -> set[str]:
    return {row.get("doc_id") for row in read_jsonl(documents_path) if row.get("doc_id")}


# ── Pipeline ────────────────────────────────────────────────────────

def _empty_stats(total: int) -> dict:
    return {
        "total_pdfs": total,
        "successful": 0,
        "failed": 0,
        "method_pymupdf4llm": 0,
        "method_pdfplumber": 0,
        "total_pages": 0,
        "total_chars_raw": 0,
        "total_chars_cleaned": 0,
        "low_quality_pages": 0,
        "low_quality_docs": 0,
    }


def _process_one_pdf(
    pdf_path: Path,
    doc_id: str,
    extra_meta: dict,
) -> tuple[dict | None, list[dict], dict]:
    """Shared PDF→records pipeline. Both API and ZIP modes call this so
    extraction/cleaning/quality logic lives in exactly one place.

    Returns (doc_row, page_rows, stats_delta). doc_row is None if extraction
    failed (caller bumps `failed` from stats_delta)."""
    pages_raw, method = extract_pdf(pdf_path)
    if not pages_raw or method == "failed":
        return None, [], {"failed": 1}

    pages_cleaned = [clean_page_text(p) for p in pages_raw]
    pages_cleaned = strip_repeated_lines(pages_cleaned)
    num_low_quality = sum(1 for p in pages_cleaned if len(p) < LOW_QUALITY_THRESHOLD)
    full_text_raw = "\n\n".join(pages_raw)
    full_text = "\n\n".join(pages_cleaned)
    low_quality_doc = (
        len(pages_cleaned) > 0
        and num_low_quality / len(pages_cleaned) >= 0.5
    )

    doc_row = {
        "doc_id": doc_id,
        **extra_meta,
        "filename": pdf_path.name,
        "text": full_text,
        "text_raw": full_text_raw,
        "num_pages": len(pages_raw),
        "char_count": len(full_text),
        "char_count_raw": len(full_text_raw),
        "word_count": len(full_text.split()),
        "num_low_quality_pages": num_low_quality,
        "low_quality_doc": low_quality_doc,
        "method": method,
    }
    page_rows = [
        {
            "doc_id": doc_id,
            "page": page_num,
            "text": cleaned,
            "text_raw": raw,
            "is_low_quality": len(cleaned) < LOW_QUALITY_THRESHOLD,
        }
        for page_num, (raw, cleaned) in enumerate(zip(pages_raw, pages_cleaned))
    ]
    stats_delta = {
        "successful": 1,
        f"method_{method}": 1,
        "total_pages": len(pages_raw),
        "total_chars_raw": len(full_text_raw),
        "total_chars_cleaned": len(full_text),
        "low_quality_pages": num_low_quality,
        "low_quality_docs": 1 if low_quality_doc else 0,
    }
    return doc_row, page_rows, stats_delta


def _merge_stats(stats: dict, delta: dict) -> None:
    for k, v in delta.items():
        stats[k] = stats.get(k, 0) + v


def build_from_zip(
    output_dir: Path = DEFAULT_ZIP_OUTPUT_DIR,
    zip_path: Path = DEFAULT_ZIP_PATH,
    limit: int | None = None,
    pdfs_dir: Path | None = None,
) -> None:
    """Extract a corpus from a user-supplied ZIP of PDFs.

    Same pymupdf4llm + cleaning + JSONL pipeline as the API path, but
    sources PDFs from a local archive. Doc IDs are `wb_zip_<sha1-10>` of
    the PDF filename — stable across reruns / reorderings — and remain
    distinguishable from API-mode `wb_<guid>` IDs.

    If `pdfs_dir` is given, that directory is used directly (no unzip);
    `zip_path` is then optional and only recorded as provenance.
    """
    if pdfs_dir is not None:
        extract_dir = pdfs_dir
        if not extract_dir.exists():
            print(f"PDFs dir not found at {extract_dir}")
            return
    else:
        if not zip_path.exists():
            print(f"ZIP not found at {zip_path}")
            return

    documents_path = output_dir / "documents.jsonl"
    pages_path = output_dir / "pages.jsonl"
    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdfs_dir is None:
        extract_dir = output_dir / "_raw"
        if not extract_dir.exists():
            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            print("Unzip complete.")

    pdf_files = sorted(extract_dir.rglob("*.pdf")) + sorted(extract_dir.rglob("*.PDF"))
    if limit:
        pdf_files = pdf_files[:limit]
    print(f"Found {len(pdf_files)} PDF files (mode=zip)")

    done = _processed_doc_ids(documents_path)
    if done:
        print(f"Already processed: {len(done)} docs (resuming)")

    stats = _empty_stats(total=len(pdf_files))
    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs (zip)"):
        # Stable id keyed on filename, not list order — survives reordering
        # / partial reruns / mixing two zips.
        name_hash = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:10]
        doc_id = f"wb_zip_{name_hash}"
        if doc_id in done:
            continue
        doc_row, page_rows, delta = _process_one_pdf(
            pdf_path, doc_id,
            extra_meta={"title": pdf_path.stem, "source_zip": str(zip_path)},
        )
        _merge_stats(stats, delta)
        if doc_row is None:
            continue
        append_jsonl(doc_row, documents_path)
        for row in page_rows:
            append_jsonl(row, pages_path)

    with metadata_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtraction complete (zip mode):")
    print(f"  Successful: {stats['successful']}/{stats['total_pdfs']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Total pages: {stats['total_pages']} ({stats['low_quality_pages']} low-quality)")
    print(f"  Total chars: {stats['total_chars_raw']:,} raw → {stats['total_chars_cleaned']:,} cleaned")
    print(f"  Methods: pymupdf4llm={stats['method_pymupdf4llm']}, pdfplumber={stats['method_pdfplumber']}")
    print(f"\nOutputs:")
    print(f"  {documents_path}")
    print(f"  {pages_path}")
    print(f"  {metadata_path}")


def build(
    output_dir: Path,
    doc_type: str | list[str] | None = None,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    limit: int = DEFAULT_LIMIT,
) -> None:
    pdfs_dir = output_dir / "pdfs"
    documents_path = output_dir / "documents.jsonl"
    pages_path = output_dir / "pages.jsonl"
    metadata_path = output_dir / "metadata.json"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata from {WB_API_BASE}")
    print(f"  doc_type={doc_type or '(all types)'}  dates={date_start} → {date_end}  limit={limit}")
    records = fetch_metadata(
        doc_type=doc_type, date_start=date_start, date_end=date_end, limit=limit,
    )
    print(f"  → {len(records)} candidate records")

    done = _processed_doc_ids(documents_path)
    if done:
        print(f"Already processed: {len(done)} docs (resuming)")

    stats = _empty_stats(total=len(records))
    for rec in tqdm(records, desc="Extracting PDFs"):
        doc_id = f"wb_{rec['guid']}"
        if doc_id in done:
            continue
        pdf_path = pdfs_dir / f"{rec['guid']}.pdf"
        try:
            download_pdf(rec["pdf_url"], pdf_path)
        except Exception as e:
            stats["failed"] += 1
            tqdm.write(f"  download failed for {rec['guid']}: {e}")
            continue

        doc_row, page_rows, delta = _process_one_pdf(
            pdf_path, doc_id,
            extra_meta={
                "guid": rec["guid"],
                "title": rec.get("title"),
                "date": rec.get("date"),
                "doc_name": rec.get("doc_name"),
                "report_name": rec.get("report_name"),
                "docty": rec.get("docty"),
                "pdf_url": rec.get("pdf_url"),
            },
        )
        _merge_stats(stats, delta)
        if doc_row is None:
            continue
        append_jsonl(doc_row, documents_path)
        for row in page_rows:
            append_jsonl(row, pages_path)

    with metadata_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Successful: {stats['successful']}/{stats['total_pdfs']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Total pages: {stats['total_pages']} ({stats['low_quality_pages']} low-quality)")
    print(f"  Total chars: {stats['total_chars_raw']:,} raw → {stats['total_chars_cleaned']:,} cleaned")
    print(f"  Methods: pymupdf4llm={stats['method_pymupdf4llm']}, pdfplumber={stats['method_pdfplumber']}")
    print(f"\nOutputs:")
    print(f"  {documents_path}")
    print(f"  {pages_path}")
    print(f"  {metadata_path}")
    print(f"  {pdfs_dir}/")


# ── Loader (used by downstream generators) ──────────────────────────

def load_corpus(corpus_dir: str | Path | None = None) -> list[dict]:
    """Read documents.jsonl from <corpus_dir> (default data/worldbank/)."""
    base = Path(corpus_dir) if corpus_dir else DEFAULT_OUTPUT_DIR
    path = base / "documents.jsonl" if base.is_dir() else base
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}. "
            f"Run `python -m evalsuite.corpus.extract` first."
        )
    return read_jsonl(path)


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Build the World Bank corpus.")
    p.add_argument("--mode", choices=["api", "zip"], default="api",
                   help="api: fetch from WB WDS (default). zip: read from --zip path")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Output dir (default: {DEFAULT_OUTPUT_DIR} for api, "
                        f"{DEFAULT_ZIP_OUTPUT_DIR} for zip)")
    # API-mode args
    p.add_argument("--doc-type", default=None, action="append",
                   help='WB document-type filter, e.g. "Board Summary". '
                        'Repeat for multiple types (each fetched separately, '
                        'results merged). Default: no filter.')
    p.add_argument("--date-start", default=DEFAULT_DATE_START)
    p.add_argument("--date-end", default=DEFAULT_DATE_END)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Max records to process")
    # ZIP-mode args
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH,
                   help="Path to worldbank.zip (zip mode)")
    p.add_argument("--pdfs-dir", type=Path, default=None,
                   help="Skip unzip; read PDFs from this directory tree "
                        "(zip mode). Useful if the zip is already extracted.")
    args = p.parse_args()

    if args.mode == "api":
        build(
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            doc_type=args.doc_type,
            date_start=args.date_start,
            date_end=args.date_end,
            limit=args.limit,
        )
    else:
        build_from_zip(
            output_dir=args.output_dir or DEFAULT_ZIP_OUTPUT_DIR,
            zip_path=args.zip,
            limit=args.limit if args.limit != DEFAULT_LIMIT else None,
            pdfs_dir=args.pdfs_dir,
        )


if __name__ == "__main__":
    main()
