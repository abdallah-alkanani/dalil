"""LLM transports used by the structure and audit stages.

Every transport module exposes the same two functions, so callers pick one by
name and nothing else changes:

    ask(prompt, *, model=..., timeout=..., ...) -> str
    ask_json(prompt, **kw) -> dict

Transports:
    openrouter    OpenRouter HTTP API (fastest; separate billing)
    ollama_cloud  ollama.com HTTP API
    ollama_cli    the signed-in local ``ollama`` CLI
    opencode_cli  the signed-in local ``opencode`` CLI, with its tools switched off
"""
from __future__ import annotations

import importlib
import json
import random
import re
from types import ModuleType

TRANSPORTS = ("openrouter", "ollama_cloud", "ollama_cli", "opencode_cli")
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """A model call failed or its reply could not be used."""


def get_transport(name: str) -> ModuleType:
    if name not in TRANSPORTS:
        raise ValueError(f"unknown transport {name!r}; choose from {', '.join(TRANSPORTS)}")
    return importlib.import_module(f"{__name__}.{name}")


def parse_json_reply(text: str, error: type[LLMError] = LLMError) -> dict:
    """Parse the JSON object out of a model reply.

    Models asked for bare JSON sometimes wrap it in a fence or add a sentence, so
    the outermost {...} is taken rather than trusting the whole reply.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise error(f"no JSON in reply: {text.strip()[:160]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise error(f"bad JSON: {exc}: {match.group(0)[:160]}") from exc


def backoff_seconds(attempt: int, retry_after: str | None = None, *,
                    base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter, honouring a server's Retry-After.

    Without a delay, retries are useless against rate limiting: instant retries
    spend every attempt inside the same rejected second.
    """
    try:
        delay = float(retry_after) if retry_after else 0.0
    except ValueError:
        delay = 0.0
    return delay or min(cap, base * (2 ** attempt)) * (0.5 + random.random())
