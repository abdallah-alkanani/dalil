"""API tests with a fake search engine and SQLite; no stores or network needed."""
from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from daleel_api import answer as answering
from daleel_api.config import Settings
from daleel_api.main import create_app

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


def fake_generate(question, passages, **kw):
    assert passages[0]["n"] == 1
    return {"text": "لا تقبل [1]", "model": "m", "finish_reason": "stop", "prompt_tokens": 10,
            "completion_tokens": 5, "cost_usd": 0.6, "latency_ms": 1}


def test_ask_stores_answer_and_feedback(client):
    h = expert(client)
    with mock.patch.object(answering, "generate", side_effect=fake_generate):
        r = client.post("/ask", json={"question": "هل تقبل الصلاة بغير طهور؟"}, headers=h).json()
    assert r["answer"] == "لا تقبل [1]"
    fb = client.post("/feedback", headers=h, json={
        "query_id": r["query_id"], "rating": "correct",
        "passages": [{"parent_id": "p1", "relevance": "direct", "citation_ok": True}]})
    assert fb.status_code == 200
    q = client.get(f"/queries/{r['query_id']}", headers=h).json()
    assert q["answer"]["cost_usd"] == 0.6 and q["feedback"][0]["passages"][0]["relevance"] == "direct"
    export = client.get("/admin/export", headers=ADMIN).json()["items"]
    assert export[0]["feedback"][0]["rating"] == "correct"


def test_ask_limits(client):
    h = expert(client)
    with mock.patch.object(answering, "generate", side_effect=fake_generate):
        assert client.post("/ask", json={"question": "س1"}, headers=h).status_code == 200
        # 0.6 + 0.6 crosses the 1.0 spend cap
        assert client.post("/ask", json={"question": "س2"}, headers=h).status_code == 200
        r = client.post("/ask", json={"question": "س3"}, headers=h)
    assert r.status_code == 429


def test_answer_failure_is_logged_and_reported(client):
    h = expert(client)
    with mock.patch.object(answering, "generate", side_effect=answering.AnswerError("model gone")):
        r = client.post("/ask", json={"question": "سؤال"}, headers=h)
    assert r.status_code == 502 and r.json()["detail"]["error"] == "model gone"


def test_prompt_renders_grade_and_neighbours():
    msg = answering.render_user_message("س", [{"n": 1, "citation": "c", "grade": "صحيح", "text": "t",
                                               "neighbours": [{"offset": 1, "citation": "c2", "text": "t2"}]}])
    assert "حكم الألباني: صحيح" in msg and "t2" in msg
