"""The answer step: prompt, passages, the search loop with a scripted model, and the
stream parser on a real ``requests.Response``; no network needed."""
from __future__ import annotations

import io
import json
from unittest import mock

import requests

from dalil_api import answer as answering
from dalil_api.config import Settings


def test_passages_render_grade_and_neighbours():
    text = answering.render_passages([{"n": 1, "citation": "c", "grade": "صحيح", "text": "t",
                                       "neighbours": [{"offset": 1, "citation": "c2", "text": "t2"}]}])
    assert text.startswith("[1] c (حكم الألباني: صحيح)") and "t2" in text


def test_system_prompt_is_the_repository_prompt_then_the_components():
    settings = Settings()
    prompt, version = answering.system_prompt(settings.answer_prompt_path)
    repo_prompt = settings.answer_prompt_path.read_text(encoding="utf-8").rstrip()
    assert prompt.startswith(repo_prompt)
    assert "Answer(blocks:" in prompt[len(repo_prompt):]
    assert len(version) == 12


# ── the search loop, with a scripted model ─────────────────────────────────────
def scripted(*turns):
    """A stand-in for _Call.stream: each turn is (content, [queries], finish)."""
    script = iter(turns)

    def stream(self, base_url, api_key, timeout, body):
        content, queries, finish = next(script)
        stream.bodies.append(json.loads(json.dumps(body)))
        if content:
            self.content = content
            yield {"delta": content}
        self.tool_calls = [{"id": f"c{i}", "type": "function",
                            "function": {"name": "search", "arguments": json.dumps({"query": q})}}
                           for i, q in enumerate(queries)]
        self.finish, self.usage, self.model = finish, {"prompt_tokens": 1, "completion_tokens": 1}, "m"
    stream.bodies = []
    return stream


def run(stream, search, **kw):
    with mock.patch.object(answering._Call, "stream", stream):
        return list(answering.run("س", search, prompt_path=Settings().answer_prompt_path, base_url="u",
                                  api_key="k", model="m", max_tokens=100, **kw))


def test_run_searches_then_answers_and_numbers_across_searches():
    stream = scripted(("", ["أ", "ب"], "tool_calls"), ("root = Answer([])", [], "stop"))
    found = {"أ": [{"parent_id": "p1"}, {"parent_id": "p2"}], "ب": [{"parent_id": "p2"}, {"parent_id": "p3"}]}
    parts = run(stream, lambda q: found[q])
    searches = [p["search"] for p in parts if "search" in p]
    assert [[x["n"] for x in s["passages"]] for s in searches] == [[1, 2], [3]]
    assert parts[-1]["done"]["passages"][-1] == {"n": 3, "parent_id": "p3"}
    tool_results = [m for m in stream.bodies[1]["messages"] if m["role"] == "tool"]
    assert tool_results[0]["content"].startswith("[1] p1") and tool_results[1]["content"].startswith("[3] p3")


def test_run_asks_for_a_search_before_answering():
    stream = scripted(("from memory", [], "stop"), ("", ["أ"], "tool_calls"), ("root = Answer([])", [], "stop"))
    parts = run(stream, lambda q: [{"parent_id": "p1"}])
    assert {"reset": True} in parts and stream.bodies[1]["messages"][-1]["content"] == answering.SEARCH_FIRST


def test_run_stops_offering_the_tool_after_the_last_search():
    stream = scripted(("", ["أ"], "tool_calls"), ("", ["ب"], "tool_calls"), ("root = Answer([])", [], "stop"))
    run(stream, lambda q: [], max_searches=2)
    assert "tools" in stream.bodies[1] and "tools" not in stream.bodies[2]


def test_run_tells_the_model_when_searches_are_used_up():
    def no_more(q):
        raise answering.SearchLimit("no searches left")
    stream = scripted(("", ["أ"], "tool_calls"), ("root = Answer([])", [], "stop"))
    run(stream, no_more)
    assert stream.bodies[1]["messages"][-1]["content"] == "no searches left" and "tools" not in stream.bodies[1]


def test_run_tells_the_model_when_a_search_fails():
    def broken(q):
        raise RuntimeError("index unavailable")
    stream = scripted(("", ["أ"], "tool_calls"), ("root = Answer([])", [], "stop"))
    parts = run(stream, broken)
    assert stream.bodies[1]["messages"][-1]["content"] == "تعذّر البحث: index unavailable"
    assert "done" in parts[-1]


def sse(*chunks):
    """A real ``requests.Response`` over the raw bytes of a server-sent event stream."""
    r = requests.Response()
    r.status_code = 200
    r.encoding = "utf-8"
    r.raw = io.BytesIO(b"".join(b"data: " + json.dumps(c, ensure_ascii=False).encode("utf-8") + b"\n\n"
                                for c in chunks) + b"data: [DONE]\n\n")
    return r


def test_stream_keeps_line_separators_inside_the_text():
    text = 'lead = Lead("سطر\u2028سطر [1]")'
    args = json.dumps({"query": "المسح على الخف"}, ensure_ascii=False)
    resp = sse({"model": "m", "choices": [{"delta": {"content": text}}]},
               {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                       "function": {"name": "search", "arguments": args[:6]}}]}}]},
               {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[6:]}}]},
                             "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3}})
    call = answering._Call()
    with mock.patch.object(answering.requests, "post", return_value=resp):
        parts = list(call.stream("u", "k", 5, {}))
    assert parts == [{"delta": text}] and call.content == text
    assert call.tool_calls[0]["function"]["arguments"] == args and call.finish == "tool_calls"
