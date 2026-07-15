"""Shared JSONL helpers used across the evalsuite (corpus, generators, eval).

Standard line-delimited JSON. Conventions:
  - read_jsonl(path)        → [] if path missing, else list[dict]
  - append_jsonl(record, p) → append one row, creates parent dir if needed
  - save_jsonl(rows, path)  → write/overwrite the whole file
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(record: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
