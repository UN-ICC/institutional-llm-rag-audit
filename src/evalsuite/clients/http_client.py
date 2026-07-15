"""Generic HTTP client wrapper with retry logic."""

from __future__ import annotations

import os
import uuid
import time
from typing import Any

import requests


CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT", "http://localhost:5173/chat")


def generate_uuid() -> str:
    return str(uuid.uuid4())


def post_with_retry(
    url: str,
    payload: dict[str, Any],
    timeout: int = 200,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """POST JSON with retry on failure."""
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  [retry {attempt + 1}/{max_retries}] Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None
