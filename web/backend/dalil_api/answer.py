"""The answer step: the answer model searches with a ``search`` tool, then writes
the answer as OpenUI Lang, streamed.

The system prompt is the repository's answering prompt
(``prompts/answering_system_prompt.md``) followed by the OpenUI component
instructions in ``answer_components.md``, which ``web/frontend`` generates from
its component library (``npm run prompt``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

COMPONENTS_PATH = Path(__file__).parent / "answer_components.md"
log = logging.getLogger("dalil_api.answer")
MAX_SEARCHES = 3

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search dalil's library of classical Islamic books. Returns numbered passages; "
                       "numbers continue across searches.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query, in Arabic"}},
            "required": ["query"],
        },
    },
}
SEARCH_FIRST = "ابحث أولًا بأداة search، ثم أجب من النصوص التي تعيدها."
NO_MORE_SEARCHES = "لا مزيد من البحث لهذا السؤال. أجب الآن من النصوص التي وصلتك."


class AnswerError(RuntimeError):
    pass


class SearchLimit(RuntimeError):
    """The user has no searches left today; the message is shown to the model."""


@lru_cache(maxsize=4)
def system_prompt(prompt_path: Path) -> tuple[str, str]:
    """(prompt, version): the version is a hash of the prompt text, stored with each answer."""
    text = prompt_path.read_text(encoding="utf-8").rstrip() + "\n\n" + COMPONENTS_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def render_passages(passages: list[dict[str, Any]]) -> str:
    """Numbered passages as the model reads them in a search result."""
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
    return "\n\n---\n\n".join(blocks)


def run(question: str, search: Callable[[str], list[dict[str, Any]]], *, prompt_path: Path, base_url: str,
        api_key: str, model: str, max_tokens: int, reasoning_effort: str = "", max_searches: int = MAX_SEARCHES,
        timeout: int = 180) -> Iterator[dict[str, Any]]:
    """Answer ``question``, calling ``search(query)`` for each search the model asks for.

    Yields ``{"search": {"query", "passages"}}`` with the new numbered passages of each
    search, ``{"delta": text}`` as the answer streams, ``{"reset": True}`` when text
    already streamed turns out to precede a search (it is discarded), and finally
    ``{"done": {...}}`` with the model, token counts, reported cost, prompt version and
    every passage shown.
    """
    if not api_key:
        raise AnswerError("ANSWER_API_KEY is not set")
    prompt, version = system_prompt(prompt_path)
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    shown: list[dict[str, Any]] = []
    seen: set[str] = set()
    searches, nudged = 0, False
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None}
    started = time.time()
    served_by = model

    for _ in range(max_searches + 3):
        tools = searches < max_searches
        call = _Call()
        yield from call.stream(base_url, api_key, timeout, {
            "model": model, "stream": True, "temperature": 0.2, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "usage": {"include": True},  # OpenRouter reports the call's cost in usage.cost
            "messages": messages,
            **({"tools": [SEARCH_TOOL]} if tools else {}),
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        })
        served_by = call.model or served_by
        totals["prompt_tokens"] += call.usage.get("prompt_tokens") or 0
        totals["completion_tokens"] += call.usage.get("completion_tokens") or 0
        if call.usage.get("cost") is not None:
            totals["cost_usd"] = (totals["cost_usd"] or 0) + call.usage["cost"]

        if call.tool_calls:
            if call.content:
                yield {"reset": True}
            messages.append({"role": "assistant", "content": call.content, "tool_calls": call.tool_calls})
            for tc in call.tool_calls:
                result = NO_MORE_SEARCHES
                if searches < max_searches:
                    try:
                        query = str(json.loads(tc["function"]["arguments"] or "{}").get("query") or "").strip()
                    except json.JSONDecodeError:
                        query = ""
                    if query:
                        try:
                            found = search(query)
                        except SearchLimit as exc:
                            result, searches = str(exc), max_searches
                        except Exception as exc:  # the model reads the failure and may search again
                            log.warning("search failed: %s", exc)
                            result, searches = f"تعذّر البحث: {exc}", searches + 1
                        else:
                            searches += 1
                            new = []
                            for r in found:
                                if r.get("parent_id") not in seen:
                                    seen.add(r.get("parent_id"))
                                    new.append({"n": len(shown) + 1, **r})
                                    shown.append(new[-1])
                            yield {"search": {"query": query, "passages": new}}
                            result = render_passages(new) or "لا نصوص جديدة لهذا البحث."
                    else:
                        result = "البحث يحتاج إلى نص."
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            continue

        if searches == 0 and not nudged and tools:
            # The prompt requires a search before every answer.
            nudged = True
            if call.content:
                yield {"reset": True}
            messages += [{"role": "assistant", "content": call.content}, {"role": "user", "content": SEARCH_FIRST}]
            continue
        if not call.content:
            raise AnswerError(f"answer model returned no text (finish_reason={call.finish})")
        if call.finish == "length":
            raise AnswerError("the answer was cut off at the token limit")
        yield {"done": {"model": served_by, "prompt_version": version, **totals, "passages": shown,
                        "latency_ms": int((time.time() - started) * 1000)}}
        return
    raise AnswerError("the answer model kept searching without answering")


class _Call:
    """One streamed chat completion: forwards text as it arrives and collects tool calls."""

    def __init__(self) -> None:
        self.content = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.finish: str | None = None
        self.usage: dict[str, Any] = {}
        self.model: str | None = None

    def stream(self, base_url: str, api_key: str, timeout: int, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            resp = requests.post(f"{base_url}/chat/completions", json=body, stream=True, timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        except requests.RequestException as exc:
            raise AnswerError(f"answer model unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise AnswerError(f"answer model HTTP {resp.status_code}: {resp.text[:500]}")
        calls: dict[int, dict[str, Any]] = {}
        try:
            # Split as bytes, then decode: decoded text is split with str.splitlines, which
            # also breaks at U+2028 and other separators that can sit inside a JSON string.
            for raw in resp.iter_lines():
                line = raw.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                self.model = chunk.get("model") or self.model
                self.usage = chunk.get("usage") or self.usage
                for choice in chunk.get("choices") or []:
                    self.finish = choice.get("finish_reason") or self.finish
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        self.content += delta["content"]
                        yield {"delta": delta["content"]}
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", len(calls)), {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
        except requests.RequestException as exc:
            raise AnswerError(f"answer model stream broke: {exc}") from exc
        self.tool_calls = [calls[i] for i in sorted(calls)]
