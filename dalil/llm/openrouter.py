"""OpenRouter transport. The fastest of the three (~1-4 s per call, measured),
and billed separately from Ollama, so it can run alongside the other transports.

The key comes from ``OPENROUTER_API_KEY`` (environment or ``.env``) or
``~/.config/openrouter.key``.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from . import RETRYABLE_HTTP, LLMError, backoff_seconds, parse_json_reply
from .env import api_key, load_env_file

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
# Provider preference, not a lock. Pinged with a trivial prompt on 2026-08-18,
# CoreWeave answered in 0.75 s where default routing took 3.02 s. Fallbacks stay
# on, so one provider having a bad minute costs speed rather than the run.
PROVIDER_ORDER = ["CoreWeave", "Baidu", "Parasail", "Alibaba", "Novita"]
# Generation ids of recent calls, newest last. OpenRouter stores completions
# server-side and /api/v1/generation?id=... fetches them back, but only by id:
# paid-for answers were once lost because a caller discarded them.
LAST_GENERATIONS: list[str] = []


class OpenRouterError(LLMError):
    pass


def ask(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 300,
        temperature: float = 0.0, retries: int = 4, max_tokens: int | None = None) -> str:
    """Send one prompt, return the assistant's text.

    Leave ``max_tokens`` unset for reasoning models: the cap counts reasoning
    tokens, and a large Arabic chunk once spent all 16,000 thinking and returned
    ``finish_reason: length`` with an empty string, which looks exactly like
    "the model found nothing".

    429/5xx back off with jitter and honour Retry-After. 402 (out of credit) is
    not retried.
    """
    load_env_file()
    key = api_key("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set (environment, .env or ~/.config/openrouter.key)")
    payload = {
        "model": model,
        "temperature": temperature,
        "provider": {"order": PROVIDER_ORDER, "allow_fallbacks": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "dalil"}

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                reply = json.loads(resp.read())
            choices = reply.get("choices") or []
            if not choices:
                raise OpenRouterError(f"no choices: {str(reply)[:200]}")
            if reply.get("id"):
                LAST_GENERATIONS.append(reply["id"])
                del LAST_GENERATIONS[:-5000]
            return (choices[0].get("message") or {}).get("content") or ""
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 402:
                raise OpenRouterError("402 Payment Required: account out of credit") from exc
            if exc.code not in RETRYABLE_HTTP or attempt == retries:
                try:
                    detail = exc.read()[:200].decode(errors="replace")
                except Exception:
                    detail = ""
                raise OpenRouterError(f"HTTP {exc.code}: {detail}") from exc
            time.sleep(backoff_seconds(attempt, exc.headers.get("Retry-After") if exc.headers else None))
        # http.client.HTTPException (IncompleteRead, ...) is not an OSError; without
        # it one truncated response ended a 114-chunk run on its last chunk.
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                http.client.HTTPException, OSError) as exc:
            last = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds(attempt, base=1.5, cap=30.0))
    raise OpenRouterError(f"{type(last).__name__}: {last}")


def ask_json(prompt: str, **kw) -> dict:
    """Send one prompt and parse the JSON object out of the reply."""
    return parse_json_reply(ask(prompt, **kw), OpenRouterError)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Send stdin (or --prompt) to OpenRouter.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--prompt", help="prompt text; omit to read stdin")
    args = ap.parse_args()
    text = args.prompt if args.prompt is not None else sys.stdin.read()
    started = time.time()
    print(ask(text, model=args.model, timeout=args.timeout))
    print(f"\n--- {time.time() - started:.1f}s ---", file=sys.stderr)
