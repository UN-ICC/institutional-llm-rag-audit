"""JSONL and CSV I/O utilities with checkpoint support."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file (overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item: dict[str, Any], path: str | Path) -> None:
    """Append a single dict to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(data: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a CSV file. Heterogeneous keys across rows
    are unioned (preserving first-seen order) so optional fields like
    HarmBench's `context_string` don't break DictWriter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    seen: dict[str, None] = {}
    for row in data:
        for k in row.keys():
            seen.setdefault(k)
    fieldnames = list(seen.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def load_checkpoint(path: str | Path, key: str = "id") -> set[str]:
    """Return set of already-processed IDs from an output file.

    Supports both JSONL and CSV formats.
    """
    path = Path(path)
    if not path.exists():
        return set()
    if path.suffix == ".csv":
        records = load_csv(path)
    else:
        records = load_jsonl(path)
    return {r[key] for r in records if key in r}
