"""API tests with a fake search engine and SQLite; no stores or network needed."""
from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

import json

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
                        admin_token="admin-secret", openrouter_api_key="k", user_daily_asks=2,
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


def fake_stream(question, passages, **kw):
    assert passages[0]["n"] == 1
    yield {"delta": 'root = Answer([lead])\n'}
    yield {"delta": 'lead = Lead("لا تقبل [1]")'}
    yield {"done": {"model": "m", "prompt_version": "abc", "prompt_tokens": 10, "completion_tokens": 5,
                    "cost_usd": 0.6, "latency_ms": 1}}


def ask(client, headers, question):
    r = client.post("/ask", json={"question": question}, headers=headers)
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()] if r.status_code == 200 else []
    return r, events


def test_ask_streams_logs_answer_and_feedback(client):
    h = expert(client)
    with mock.patch.object(answering, "stream", side_effect=fake_stream):
        r, events = ask(client, h, "هل تقبل الصلاة بغير طهور؟")
    assert [e["type"] for e in events] == ["retrieval", "delta", "delta", "done"]
    retrieval = events[0]
    assert retrieval["passages"][0]["n"] == 1 and retrieval["passages"][0]["grade"] == "صحيح"
    query_id = retrieval["query_id"]
    fb = client.post("/feedback", headers=h, json={
        "query_id": query_id, "rating": "correct",
        "passages": [{"parent_id": "p1", "relevance": "direct", "citation_ok": True}]})
    assert fb.status_code == 200
    q = client.get(f"/queries/{query_id}", headers=h).json()
    assert q["answer"]["text"] == 'root = Answer([lead])\nlead = Lead("لا تقبل [1]")'
    assert q["answer"]["prompt_version"] == "abc" and q["answer"]["cost_usd"] == 0.6
    assert q["feedback"][0]["passages"][0]["relevance"] == "direct"
    export = client.get("/admin/export", headers=ADMIN).json()["items"]
    assert export[0]["feedback"][0]["rating"] == "correct"


def test_ask_limits(client):
    h = expert(client)
    with mock.patch.object(answering, "stream", side_effect=fake_stream):
        assert ask(client, h, "س1")[0].status_code == 200
        # 0.6 + 0.6 crosses the 1.0 spend cap
        assert ask(client, h, "س2")[0].status_code == 200
        r, _ = ask(client, h, "س3")
    assert r.status_code == 429


def test_answer_failure_is_logged_and_reported(client):
    h = expert(client)
    with mock.patch.object(answering, "stream", side_effect=answering.AnswerError("model gone")):
        r, events = ask(client, h, "سؤال")
    assert events[-1] == {"type": "error", "stage": "answer", "message": "model gone"}
    q = client.get(f"/queries/{events[0]['query_id']}", headers=h).json()
    assert q["query"]["error"] == "model gone" and q["answer"] is None


def test_prompt_renders_grade_and_neighbours():
    msg = answering.render_user_message("س", [{"n": 1, "citation": "c", "grade": "صحيح", "text": "t",
                                               "neighbours": [{"offset": 1, "citation": "c2", "text": "t2"}]}])
    assert "حكم الألباني: صحيح" in msg and "t2" in msg


def test_system_prompt_is_the_repository_prompt_then_the_components():
    settings = Settings()
    prompt, version = answering.system_prompt(settings.answer_prompt_path)
    repo_prompt = settings.answer_prompt_path.read_text(encoding="utf-8").rstrip()
    assert prompt.startswith(repo_prompt)
    assert "Answer(blocks:" in prompt[len(repo_prompt):]
    assert len(version) == 12
