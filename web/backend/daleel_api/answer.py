"""The answer step: one OpenRouter chat call over numbered passages."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

PROMPT_VERSION = "answer_v1"
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class AnswerError(RuntimeError):
    pass


def number_passages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The passages shown to the model, numbered from 1 in rank order."""
    return [{"n": i, **r} for i, r in enumerate(results, 1)]


def render_user_message(question: str, passages: list[dict[str, Any]]) -> str:
    blocks = []
    for p in passages:
        head = f"[{p['n']}] {p.get('citation') or p.get('parent_id')}"
        if p.get("grade"):
            head += f" (حكم الألباني: {p['grade']})"
        text = p.get("text") or ""
        for nb in p.get("neighbours") or []:
            where = "قبله" if (nb.get("offset") or 0) < 0 else "بعده"
            text += f"\n\n(نص متصل {where}: {nb.get('citation') or ''})\n{nb.get('text') or ''}"
        blocks.append(f"{head}\n{text}")
    return f"السؤال: {question}\n\nالنصوص:\n\n" + "\n\n---\n\n".join(blocks)


def generate(question: str, passages: list[dict[str, Any]], *, api_key: str, model: str,
             max_tokens: int, timeout: int = 120) -> dict[str, Any]:
    if not api_key:
        raise AnswerError("OPENROUTER_API_KEY is not set")
    started = time.time()
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_user_message(question, passages)},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "usage": {"include": True},  # OpenRouter reports the call's cost in usage.cost
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise AnswerError(f"answer model unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise AnswerError(f"answer model HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    if not text:
        raise AnswerError(f"answer model returned no text (finish_reason={choice.get('finish_reason')})")
    usage = data.get("usage") or {}
    return {
        "text": text,
        "model": data.get("model") or model,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "latency_ms": int((time.time() - started) * 1000),
    }
