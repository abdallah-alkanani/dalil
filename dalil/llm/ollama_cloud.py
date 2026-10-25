"""Direct client for models hosted on ollama.com (DeepSeek by default).

The audit tools used to shell out to a coding-agent CLI, which starts a full
agent loop: it loads tools and runs shell commands to check its own work before
replying. A prompt reading "Reply with exactly: OK" took 4m33s that way and 1.9s
against the HTTP endpoint. So this module talks to the endpoint directly: one
request, one answer, no agent loop.

Needs ``OLLAMA_API_KEY`` in the environment or in ``.env``.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import RETRYABLE_HTTP, LLMError, backoff_seconds, parse_json_reply
from .env import api_key, load_env_file

ENDPOINT = "https://ollama.com/api/chat"
DEFAULT_MODEL = "deepseek-v4-flash:cloud"
PRO_MODEL = "deepseek-v4-pro:cloud"


class OllamaCloudError(LLMError):
    pass


def ask(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 300,
        temperature: float = 0.0, retries: int = 6) -> str:
    """Send one prompt, return the assistant's text.

    The endpoint rate-limits at high concurrency, so 429 and 5xx back off
    exponentially with jitter and honour Retry-After.
    """
    load_env_file()
    key = api_key("OLLAMA_API_KEY")
    if not key:
        raise OllamaCloudError("OLLAMA_API_KEY is not set (environment or .env)")
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            return (payload.get("message") or {}).get("content") or ""
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRYABLE_HTTP or attempt == retries:
                break
            time.sleep(backoff_seconds(attempt, exc.headers.get("Retry-After") if exc.headers else None))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds(attempt, base=1.5, cap=30.0))
    raise OllamaCloudError(f"{type(last).__name__}: {last}")


def ask_json(prompt: str, **kw) -> dict:
    """Send one prompt and parse the JSON object out of the reply."""
    return parse_json_reply(ask(prompt, **kw), OllamaCloudError)
