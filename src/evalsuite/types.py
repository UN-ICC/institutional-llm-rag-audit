"""Core dataclasses for the evalsuite evaluation pipeline.

Scorer-specific result types live next to each scorer
(see scorers/harmbench.py, scorers/jailbreakbench.py, scorers/strongreject.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    """A single evaluation test case."""

    id: str
    prompt: str
    source: str  # e.g. "harmbench", "jailbreakbench", "strongreject"
    category: str = ""
    expected_behavior: str = "refuse"  # "refuse" or "comply"
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result from querying the RAG endpoint."""

    test_case: TestCase
    response: str
    processing_time: float = 0.0
    timestamp: str = ""
    model: str = ""
