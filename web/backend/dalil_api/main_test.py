"""API tests with a fake search engine and SQLite; no stores or network needed."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dalil_api import answer as answering
from dalil_api.config import Settings
from dalil_api.main import create_app

ADMIN = {"Authorization": "Bearer admin-secret"}


class FakeSearch:
    state, detail = "ready", ""
    ready = True

    def __init__(self):
        self.calls = []

    def missing_files(self):
        return []

    def query(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {
            "status": {"retrieval": "hybrid", "planner": "off", "rerank": "off", "source_classifier": "off"},
            "results": [{
                "parent_id": "p1", "citation": "جامع الترمذي، حديث 1", "category": "hadith",
                "book_id": "hadith_api::jami_al_tirmidhi", "text_for_synthesis": "لا تقبل صلاة بغير طهور",
                "refs": {"hadith": {"albani_grade": "صحيح"}},
                "neighbours": [{"offset": 1, "parent_id": "p2", "citation": "حديث 2", "text": "نص"}],
            }],
        }


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url=f"sqlite:///{tmp_path}/t.sqlite",
                        admin_token="admin-secret", openrouter_api_key="k", answer_api_key="k", user_daily_asks=2,
                        daily_spend_cap_usd=1.0)
    search = FakeSearch()
    with TestClient(create_app(settings, search)) as c:
        c.fake_search = search
        yield c


def expert(client) -> dict:
    code = client.post("/admin/invites", json={"label": "test"}, headers=ADMIN).json()["code"]
    token = client.post("/auth/redeem", json={"code": code, "name": "Sheikh A"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_needs_no_auth(client):
    assert client.get("/health").json()["search"] == "ready"


def test_invites_are_single_use_and_admin_only(client):
    assert client.post("/admin/invites", json={}).status_code == 401
    assert client.post("/admin/invites", json={}, headers={"Authorization": "Bearer nope"}).status_code == 403
    code = client.post("/admin/invites", json={}, headers=ADMIN).json()["code"]
    assert client.post("/auth/redeem", json={"code": code, "name": "a"}).status_code == 200
    assert client.post("/auth/redeem", json={"code": code, "name": "b"}).status_code == 403


def test_search_logs_and_returns_grade(client):
    h = expert(client)
    assert client.post("/search", json={"question": "حكم الطهارة"}).status_code == 401
    r = client.post("/search", json={"question": "حكم الطهارة"}, headers=h).json()
    assert r["results"][0]["grade"] == "صحيح"
    assert r["status"]["retrieval"] == "hybrid"
    q = client.get(f"/queries/{r['query_id']}", headers=h).json()
    assert q["query"]["results"] == [{"parent_id": "p1", "citation": "جامع الترمذي، حديث 1"}]


def fake_run(question, search, **kw):
    found = search("حكم الطهارة")
    passages = [{"n": i, **r} for i, r in enumerate(found, 1)]
    yield {"search": {"query": "حكم الطهارة", "passages": passages}}
    yield {"delta": "I will search"}
    yield {"reset": True}
    yield {"delta": 'root = Answer([lead])\n'}
    yield {"delta": 'lead = Lead("لا تقبل [1]")'}
    yield {"done": {"model": "m", "prompt_version": "abc", "prompt_tokens": 10, "completion_tokens": 5,
                    "cost_usd": 0.6, "passages": passages, "latency_ms": 1}}


def ask(client, headers, question):
    r = client.post("/ask", json={"question": question}, headers=headers)
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()] if r.status_code == 200 else []
    return r, events


def test_ask_streams_searches_logs_answer_and_feedback(client):
    h = expert(client)
    with mock.patch.object(answering, "run", side_effect=fake_run):
        r, events = ask(client, h, "هل تقبل الصلاة بغير طهور؟")
    assert [e["type"] for e in events] == ["start", "search", "delta", "reset", "delta", "delta", "done"]
    assert events[1]["query"] == "حكم الطهارة" and events[1]["passages"][0]["grade"] == "صحيح"
    assert events[1]["status"]["retrieval"] == "hybrid"
    query_id = events[0]["query_id"]
    fb = client.post("/feedback", headers=h, json={
        "query_id": query_id, "rating": "correct",
        "passages": [{"parent_id": "p1", "relevance": "direct", "citation_ok": True}]})
    assert fb.status_code == 200
    q = client.get(f"/queries/{query_id}", headers=h).json()
    assert q["answer"]["text"] == 'root = Answer([lead])\nlead = Lead("لا تقبل [1]")'
    assert q["answer"]["prompt_version"] == "abc" and q["answer"]["cost_usd"] == 0.6
    assert q["query"]["results"] == [{"parent_id": "p1", "citation": "جامع الترمذي، حديث 1"}]
    assert q["query"]["status"]["searches"][0]["query"] == "حكم الطهارة"
    assert q["feedback"][0]["passages"][0]["relevance"] == "direct"
    export = client.get("/admin/export", headers=ADMIN).json()["items"]
    assert [i["kind"] for i in export] == ["ask", "search"]


def test_feedback_only_on_own_queries(client):
    a, b = expert(client), expert(client)
    with mock.patch.object(answering, "run", side_effect=fake_run):
        _, events = ask(client, a, "سؤال")
    fb = {"query_id": events[0]["query_id"], "rating": "correct"}
    assert client.post("/feedback", headers=b, json=fb).status_code == 404
    assert client.post("/feedback", headers=admin_user(client), json=fb).status_code == 200


def admin_user(client) -> dict:
    code = client.post("/admin/invites", json={"role": "admin"}, headers=ADMIN).json()["code"]
    token = client.post("/auth/redeem", json={"code": code, "name": "Admin"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_ask_limits(client):
    h = expert(client)
    with mock.patch.object(answering, "run", side_effect=fake_run):
        assert ask(client, h, "س1")[0].status_code == 200
        # 0.6 + 0.6 crosses the 1.0 spend cap
        assert ask(client, h, "س2")[0].status_code == 200
        r, _ = ask(client, h, "س3")
    assert r.status_code == 429


def test_searches_count_against_the_daily_limit(client):
    client.app.state.settings.user_daily_searches = 1
    h = expert(client)
    stream = scripted(("", ["أ", "ب"], "tool_calls"), ("root = Answer([])", [], "stop"))
    with mock.patch.object(answering._Call, "stream", stream):
        _, events = ask(client, h, "سؤال")
    assert [e["type"] for e in events] == ["start", "search", "delta", "done"]
    tool_results = [m["content"] for m in stream.bodies[1]["messages"] if m["role"] == "tool"]
    assert "انتهى عدد مرات البحث" in tool_results[1]


def test_answer_failure_is_logged_and_reported(client):
    h = expert(client)
    with mock.patch.object(answering, "run", side_effect=answering.AnswerError("model gone")):
        r, events = ask(client, h, "سؤال")
    assert events[-1] == {"type": "error", "stage": "answer", "message": "model gone"}
    q = client.get(f"/queries/{events[0]['query_id']}", headers=h).json()
    assert q["query"]["error"] == "model gone" and q["answer"] is None


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
