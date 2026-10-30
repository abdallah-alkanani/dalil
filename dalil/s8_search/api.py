"""OpenRouter endpoints used at query time: query embeddings, the query planner,
source classification and the optional listwise rerank."""
from __future__ import annotations

import os
import time
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM = 4096
# Same instruction the corpus was embedded against; changing it moves every
# query vector relative to the stored documents.
EMBED_INSTRUCTION_AR = (
    "Instruct: Given a search query, retrieve relevant Arabic Islamic texts "
    "from Quran, Hadith, Tafsir, Fiqh, Aqeedah, and Seerah sources\n"
    "Query:"
)

# Chat models are read from the environment so a model OpenRouter retires can be
# swapped without a release. The previous pins (qwen/qwen3-8b for the rerank and
# source classifier, qwen3-30b-a3b-instruct-2507 on CoreWeave only for the planner)
# stopped being served and every call failed silently. Claude Haiku 5.5 is served
# by several providers and handles Arabic and JSON output; it has not been through
# the 2026-08-28 planner/critic benchmark, so re-run that before relying on it.
LISTWISE_MODEL = os.environ.get("DALIL_LISTWISE_MODEL", "anthropic/claude-haiku-5.5")
PLANNER_MODEL = os.environ.get("DALIL_PLANNER_MODEL", "anthropic/claude-haiku-5.5")
# Optional preferred provider for the planner; others are still allowed as fallbacks.
PLANNER_PROVIDER = os.environ.get("DALIL_PLANNER_PROVIDER", "")
PLANNER_TIMEOUT_SECONDS = 55


def request_with_retries(
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: int,
    retries: int = 3,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc
