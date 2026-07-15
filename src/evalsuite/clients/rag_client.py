"""RAG client for UNICC's Apertus endpoint.

Wraps the POST /chat protocol used by UNICC's Apertus RAG system.
Protocol adapted from the legacy runner.py in this repo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evalsuite.clients.http_client import (
    CHAT_ENDPOINT,
    generate_uuid,
    post_with_retry,
)
from evalsuite.types import TestCase, RunResult


@dataclass
class RAGClient:
    """Client for UNICC's Apertus RAG endpoint."""

    endpoint: str = ""
    model_name: str = "apertus"

    def __post_init__(self):
        self.endpoint = self.endpoint or os.getenv("CHAT_ENDPOINT", CHAT_ENDPOINT)

    def _build_payload(self, question: str, system_prompt: str = "") -> dict[str, Any]:
        """Build the request payload matching UNICC's /chat protocol."""
        history_entry = {"user": question}
        if system_prompt:
            history_entry = {"user": f"{system_prompt}\n{question}"}

        return {
            "logging_info": {
                "conversation_id": generate_uuid(),
                "chat_id": generate_uuid(),
                "chat_sequence": 0,
                "user_fullname": "Admin",
                "user_email": "eval@evalsuite.local",
                "user_organization": "EvalSuite",
            },
            "history": [history_entry],
            "approach": "rrr",
            "overrides": {
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "semantic_captions": False,
                "top": 15,
                "suggest_followup_questions": False,
                "scoring_profile": False,
                "organization": "WORLDBANK",
            },
        }

    def query(self, question: str, system_prompt: str = "") -> str | None:
        """Send a single question to the RAG endpoint. Returns response text."""
        payload = self._build_payload(question, system_prompt)
        data = post_with_retry(self.endpoint, payload)
        if data is None:
            return None
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
        return str(data)

    def run_test_case(self, test_case: TestCase) -> RunResult:
        """Run a single test case and return a RunResult."""
        start = time.time()
        response = self.query(test_case.prompt, test_case.system_prompt) or ""
        elapsed = time.time() - start

        return RunResult(
            test_case=test_case,
            response=response,
            processing_time=round(elapsed, 2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model=self.model_name,
        )
