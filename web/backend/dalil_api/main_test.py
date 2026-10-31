"""API tests with a fake search engine and SQLite; no stores or network needed."""
from __future__ import annotations

import json
import uuid
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dalil_api import answer as answering
from dalil_api import country
from dalil_api.answer_test import scripted
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
                        admin_token="admin-secret", site_token="site-secret", openrouter_api_key="k", answer_api_key="k", user_daily_asks=2,
                        daily_spend_cap_usd=1.0)
    search = FakeSearch()
    with TestClient(create_app(settings, search)) as c:
        c.fake_search = search
        yield c


def visitor(client, country_code: str = "EG") -> dict:
    """The headers the web app's server sends for one browser."""
    return {"Authorization": "Bearer site-secret", "X-Dalil-Visitor": uuid.uuid4().hex,
            "X-Dalil-Country": country_code}


def test_health_needs_no_auth(client):
    assert client.get("/health").json()["search"] == "ready"


def test_visitors_come_through_the_site_only(client):
    h = visitor(client)
    assert client.get("/visitor").status_code == 401
    assert client.get("/visitor", headers={**h, "Authorization": "Bearer nope"}).status_code == 403
    assert client.get("/visitor", headers={**h, "X-Dalil-Visitor": "short"}).status_code == 400
    # Opening the page creates no visitor; giving a name (or asking) does.
    assert client.get("/visitor", headers=h).json() == {"id": None, "name": ""}
    first = client.post("/visitor", headers=h, json={"name": ""}).json()
    assert client.get("/visitor", headers=h).json() == first and first["id"] is not None
    other = visitor(client)
    assert client.post("/visitor", headers=other, json={}).json()["id"] != first["id"]


def test_a_visitor_may_give_a_name_and_the_admin_sees_who_asks_from_where(client):
    friend, stranger = visitor(client, "EG"), visitor(client, "fr")
    assert client.post("/visitor", headers=friend, json={"name": " أحمد "}).json()["name"] == "أحمد"
    with mock.patch.object(answering, "run", side_effect=fake_run):
        ask(client, friend, "سؤال")
        ask(client, stranger, "question")
    assert client.get("/admin/visitors", headers=friend).status_code == 403
    seen = client.get("/admin/visitors", headers=ADMIN).json()
    people = {v["name"]: v for v in seen["visitors"]}
    assert people["أحمد"]["countries"] == ["EG"] and people["أحمد"]["asks"] == 1
    assert people[""]["countries"] == ["FR"]
    assert {c["country"]: c["asks"] for c in seen["countries"]} == {"EG": 1, "FR": 1}


def test_country_comes_from_the_host_or_the_offline_database():
    assert country.lookup("eg", "8.8.8.8", "") == "EG"
    assert country.lookup("XX", None, "") is None  # Cloudflare's unknown
    assert country.lookup(None, "8.8.8.8", "") is None  # no database configured
    with mock.patch.object(country, "_reader") as reader:
        reader.return_value.get.return_value = {"country": {"iso_code": "US"}}
        assert country.lookup(None, "8.8.8.8", "/geo.mmdb") == "US"


def test_search_logs_and_returns_grade(client):
    h = visitor(client)
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
    h = visitor(client)
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
    a, b = visitor(client), visitor(client)
    with mock.patch.object(answering, "run", side_effect=fake_run):
        _, events = ask(client, a, "سؤال")
    fb = {"query_id": events[0]["query_id"], "rating": "correct"}
    assert client.post("/feedback", headers=b, json=fb).status_code == 404
    assert client.post("/feedback", headers=a, json=fb).status_code == 200


def test_ask_limits(client):
    h = visitor(client)
    with mock.patch.object(answering, "run", side_effect=fake_run):
        assert ask(client, h, "س1")[0].status_code == 200
        # 0.6 + 0.6 crosses the 1.0 spend cap
        assert ask(client, h, "س2")[0].status_code == 200
        r, _ = ask(client, h, "س3")
    assert r.status_code == 429


def test_searches_count_against_the_daily_limit(client):
    client.app.state.settings.user_daily_searches = 1
    h = visitor(client)
    stream = scripted(("", ["أ", "ب"], "tool_calls"), ("root = Answer([])", [], "stop"))
    with mock.patch.object(answering._Call, "stream", stream):
        _, events = ask(client, h, "سؤال")
    assert [e["type"] for e in events] == ["start", "search", "delta", "done"]
    tool_results = [m["content"] for m in stream.bodies[1]["messages"] if m["role"] == "tool"]
    assert "انتهى عدد مرات البحث" in tool_results[1]


def test_answer_failure_is_logged_and_reported(client):
    h = visitor(client)
    with mock.patch.object(answering, "run", side_effect=answering.AnswerError("model gone")):
        r, events = ask(client, h, "سؤال")
    assert events[-1] == {"type": "error", "stage": "answer", "message": "model gone"}
    q = client.get(f"/queries/{events[0]['query_id']}", headers=h).json()
    assert q["query"]["error"] == "model gone" and q["answer"] is None
