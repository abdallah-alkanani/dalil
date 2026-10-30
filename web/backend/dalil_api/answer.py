"""The answer step: one streamed chat call over numbered passages.

The system prompt is the repository's answering prompt
(``prompts/answering_system_prompt.md``) followed by the OpenUI component
instructions in ``answer_components.md``, which ``web/frontend`` generates from
its component library (``npm run prompt``). The model writes OpenUI Lang.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

COMPONENTS_PATH = Path(__file__).parent / "answer_components.md"


class AnswerError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def system_prompt(prompt_path: Path) -> tuple[str, str]:
    """(prompt, version): the version is a hash of the prompt text, stored with each answer."""
    text = prompt_path.read_text(encoding="utf-8").rstrip() + "\n\n" + COMPONENTS_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def number_passages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The passages shown to the model, numbered from 1 in rank order."""
    return [{"n": i, **r} for i, r in enumerate(results, 1)]


def render_user_message(question: str, passages: list[dict[str, Any]]) -> str:
    blocks = []
    for p in passages:
        head = f"[{p['n']}] {p.get('citation') or p.get('parent_id')}"
        if p.get("page_number"):
            head += f" (ص {p['page_number']})"
        if p.get("grade"):
            head += f" (حكم الألباني: {p['grade']})"
        text = p.get("text") or ""
        if p.get("truncated"):
            text += " [النص مقطوع هنا]"
        for nb in p.get("neighbours") or []:
            where = "قبله" if (nb.get("offset") or 0) < 0 else "بعده"
            text += f"\n\n(نص متصل {where}: {nb.get('citation') or ''})\n{nb.get('text') or ''}"
        blocks.append(f"{head}\n{text}")
    return f"السؤال: {question}\n\nالنصوص:\n\n" + "\n\n---\n\n".join(blocks)


def stream(question: str, passages: list[dict[str, Any]], *, prompt_path: Path, base_url: str, api_key: str,
           model: str, max_tokens: int, reasoning_effort: str = "", timeout: int = 120) -> Iterator[dict[str, Any]]:
    """Yield ``{"delta": text}`` as the answer streams, then one ``{"done": {...}}``
    with the model, token counts, reported cost and the prompt version."""
    if not api_key:
        raise AnswerError("ANSWER_API_KEY is not set")
    prompt, version = system_prompt(prompt_path)
    started = time.time()
    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
        "usage": {"include": True},  # OpenRouter reports the call's cost in usage.cost
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": render_user_message(question, passages)},
        ],
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=body, stream=True, timeout=timeout,
                             headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    except requests.RequestException as exc:
        raise AnswerError(f"answer model unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise AnswerError(f"answer model HTTP {resp.status_code}: {resp.text[:500]}")

    resp.encoding = "utf-8"
    wrote, finish, usage, served_by = False, None, {}, model
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        served_by = chunk.get("model") or served_by
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            text = (choice.get("delta") or {}).get("content")
            if text:
                wrote = True
                yield {"delta": text}
    if not wrote:
        raise AnswerError(f"answer model returned no text (finish_reason={finish})")
    if finish == "length":
        raise AnswerError("the answer was cut off at the token limit")
    yield {"done": {
        "model": served_by,
        "prompt_version": version,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "latency_ms": int((time.time() - started) * 1000),
    }}
