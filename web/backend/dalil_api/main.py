"""dalil API: search, cited answers, and expert feedback.

    uvicorn dalil_api.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import answer as answering
from . import auth, store
from .auth import AdminOnly, CurrentUser
from .config import Settings
from .search import SearchService, compact_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dalil_api")

Category = Literal["quran", "hadith", "tafsir", "fiqh", "fiqh_usul", "aqeedah", "seerah"]


def create_app(settings: Settings | None = None, search: SearchService | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.db = store.connect(settings.database_url)
        app.state.search = search or SearchService(settings)
        if app.state.search.state == "starting":
            app.state.search.start_background()
        yield

    app = FastAPI(title="dalil API", version="0.1.0", lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["*"], allow_headers=["*"])
    _routes(app)
    return app


# ── request bodies ──────────────────────────────────────────────────────────────
class SearchBody(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    n_results: int = Field(10, ge=1, le=30)
    categories: list[Category] | None = None
    query_planning: Literal["off", "auto", "on"] = "off"
    rerank: bool = False
    expand_neighbours: int = Field(0, ge=0, le=2)
    passage_max_chars: int = Field(3000, ge=0, le=20000)


class AskBody(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    categories: list[Category] | None = None
    rerank: bool = False


class RedeemBody(BaseModel):
    code: str
    name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(None, max_length=320)


class InviteBody(BaseModel):
    label: str = Field("", max_length=200)
    role: Literal["expert", "admin"] = "expert"
    max_uses: int = Field(1, ge=1, le=1000)


class PassageJudgment(BaseModel):
    parent_id: str
    relevance: Literal["direct", "supporting", "topical", "irrelevant"] | None = None
    citation_ok: bool | None = None
    note: str | None = Field(None, max_length=4000)


class FeedbackBody(BaseModel):
    query_id: int
    rating: Literal["correct", "partly_correct", "incorrect", "unsafe"] | None = None
    comment: str | None = Field(None, max_length=20000)
    corrected_answer: str | None = Field(None, max_length=50000)
    passages: list[PassageJudgment] = []
    missing_sources: str | None = Field(None, max_length=20000)


# ── helpers ─────────────────────────────────────────────────────────────────────
def _require_ready(request: Request) -> SearchService:
    search: SearchService = request.app.state.search
    if not search.ready:
        raise HTTPException(503, f"Search is not ready yet ({search.state}{': ' + search.detail if search.detail else ''})")
    return search


def _check_quota(request: Request, user: dict[str, Any], kind: str) -> None:
    s: Settings = request.app.state.settings
    db = request.app.state.db
    if user["role"] == "admin":
        return
    per_user = s.user_daily_asks if kind == "ask" else s.user_daily_searches
    if store.count_today(db, kind, user["id"]) >= per_user:
        raise HTTPException(429, f"Daily limit of {per_user} {kind}s reached; it resets at 00:00 UTC")
    if kind == "ask":
        if store.count_today(db, "ask") >= s.global_daily_asks:
            raise HTTPException(429, "Today's question limit has been reached; it resets at 00:00 UTC")
        if store.spend_today(db) >= s.daily_spend_cap_usd:
            raise HTTPException(429, "Today's spending limit has been reached; it resets at 00:00 UTC")


def _run_search(search: SearchService, question: str, *, n_results: int, categories, query_planning: str,
                rerank: bool, expand_neighbours: int, passage_max_chars: int) -> tuple[dict[str, Any], list[dict]]:
    payload = search.query(
        question, n_results=n_results, categories=categories, query_planning=query_planning,
        use_listwise_rerank=rerank, expand_neighbours=expand_neighbours,
        parent_max_chars=passage_max_chars,
    )
    results = [compact_result(r, passage_max_chars) for r in payload.get("results", [])[:n_results]]
    return payload, results


def _log_query(db, *, user_id: int, kind: str, question: str, params: dict, status: dict,
               results: list[dict], latency_ms: int, error: str | None = None) -> int:
    with db.begin() as conn:
        return conn.execute(store.queries.insert().values(
            user_id=user_id, kind=kind, question=question, params=params, status=status,
            results=[{"parent_id": r["parent_id"], "citation": r["citation"]} for r in results],
            latency_ms=latency_ms, error=error,
        )).inserted_primary_key[0]


def _event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _finish_query(db, query_id: int, started: float, error: str | None, **values: Any) -> None:
    with db.begin() as conn:
        conn.execute(store.queries.update().where(store.queries.c.id == query_id).values(
            latency_ms=int((time.time() - started) * 1000), error=error, **values))


def _searches_left(request: Request, user: dict[str, Any]) -> bool:
    s: Settings = request.app.state.settings
    return user["role"] == "admin" or store.count_today(request.app.state.db, "search", user["id"]) < s.user_daily_searches


# ── routes ──────────────────────────────────────────────────────────────────────
def _routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Liveness for Railway: 200 while the stores download, so the deploy isn't killed."""
        search: SearchService = request.app.state.search
        return {"ok": True, "search": search.state, "detail": search.detail}

    @app.get("/status", dependencies=[AdminOnly])
    def status(request: Request) -> dict[str, Any]:
        s: Settings = request.app.state.settings
        search: SearchService = request.app.state.search
        db = request.app.state.db
        return {
            "search": search.state, "detail": search.detail, "missing_files": search.missing_files(),
            "compact_dims": s.compact_dims, "answer_model": s.answer_model,
            "openrouter_key_set": bool(s.openrouter_api_key), "answer_key_set": bool(s.answer_api_key),
            "today": {"asks": store.count_today(db, "ask"), "searches": store.count_today(db, "search"),
                      "spend_usd": round(store.spend_today(db), 4), "spend_cap_usd": s.daily_spend_cap_usd},
        }

    @app.post("/admin/invites", dependencies=[AdminOnly])
    def create_invite(body: InviteBody, request: Request) -> dict[str, Any]:
        code = auth.create_invite(request.app.state.db, body.label, body.role, body.max_uses)
        return {"code": code, "label": body.label, "role": body.role, "max_uses": body.max_uses}

    @app.post("/auth/redeem")
    def redeem(body: RedeemBody, request: Request) -> dict[str, Any]:
        user, token = auth.redeem_invite(request.app.state.db, body.code, body.name, body.email)
        return {"token": token, "user": {"id": user["id"], "name": user["name"], "role": user["role"]}}

    @app.get("/me")
    def me(user: dict = CurrentUser) -> dict[str, Any]:
        return {"id": user["id"], "name": user["name"], "role": user["role"]}

    @app.post("/search")
    def search_route(body: SearchBody, request: Request, user: dict = CurrentUser) -> dict[str, Any]:
        search = _require_ready(request)
        _check_quota(request, user, "search")
        started = time.time()
        payload, results = _run_search(
            search, body.question, n_results=body.n_results, categories=body.categories,
            query_planning=body.query_planning, rerank=body.rerank,
            expand_neighbours=body.expand_neighbours, passage_max_chars=body.passage_max_chars)
        latency = int((time.time() - started) * 1000)
        status = payload.get("status", {})
        query_id = _log_query(request.app.state.db, user_id=user["id"], kind="search", question=body.question,
                              params=body.model_dump(exclude={"question"}), status=status,
                              results=results, latency_ms=latency, error=payload.get("error"))
        return {"query_id": query_id, "status": status, "results": results, "latency_ms": latency,
                "error": payload.get("error")}

    @app.post("/ask")
    def ask(body: AskBody, request: Request, user: dict = CurrentUser) -> StreamingResponse:
        """The answer as NDJSON events: ``start`` (the query id), one ``search`` per
        search the answer model runs (its query and the new numbered passages),
        ``delta`` (OpenUI Lang text; ``reset`` discards text streamed so far), then
        ``done`` or ``error``. Each search counts against the user's daily searches;
        the question, the passages and the answer are logged."""
        s: Settings = request.app.state.settings
        db = request.app.state.db
        search = _require_ready(request)
        _check_quota(request, user, "ask")
        params = body.model_dump(exclude={"question"})

        def run_search(query: str) -> list[dict[str, Any]]:
            if not _searches_left(request, user):
                raise answering.SearchLimit(f"انتهى عدد مرات البحث اليوم ({s.user_daily_searches}). أجب من النصوص التي وصلتك.")
            started = time.time()
            payload, results = _run_search(
                search, query, n_results=s.answer_passages, categories=body.categories, query_planning="off",
                rerank=body.rerank, expand_neighbours=1, passage_max_chars=s.passage_max_chars)
            status = payload.get("status", {})
            _log_query(db, user_id=user["id"], kind="search", question=query, params={**params, "ask_id": query_id},
                       status=status, results=results, latency_ms=int((time.time() - started) * 1000),
                       error=payload.get("error"))
            statuses.append({"query": query, **status})
            return results

        statuses: list[dict[str, Any]] = []
        query_id = _log_query(db, user_id=user["id"], kind="ask", question=body.question, params=params,
                              status={}, results=[], latency_ms=0)

        def events():
            started = time.time()
            yield _event({"type": "start", "query_id": query_id})
            text, done = "", None
            try:
                for part in answering.run(
                        body.question, run_search, prompt_path=s.answer_prompt_path, base_url=s.answer_base_url,
                        api_key=s.answer_api_key, model=s.answer_model, max_tokens=s.answer_max_tokens,
                        reasoning_effort=s.answer_reasoning_effort):
                    if "search" in part:
                        yield _event({"type": "search", **part["search"], "status": statuses[-1]})
                    elif "reset" in part:
                        text = ""
                        yield _event({"type": "reset"})
                    elif "delta" in part:
                        text += part["delta"]
                        yield _event({"type": "delta", "text": part["delta"]})
                    else:
                        done = part["done"]
            except answering.AnswerError as exc:
                log.warning("answer failed: %s", exc)
                _finish_query(db, query_id, started, str(exc), status={"searches": statuses})
                yield _event({"type": "error", "stage": "answer", "message": str(exc)})
                return
            shown = [{"n": p["n"], "parent_id": p["parent_id"], "citation": p["citation"]} for p in done["passages"]]
            with db.begin() as conn:
                conn.execute(store.answers.insert().values(
                    query_id=query_id, model=done["model"], prompt_version=done["prompt_version"], text=text,
                    passages=shown, prompt_tokens=done["prompt_tokens"],
                    completion_tokens=done["completion_tokens"], cost_usd=done["cost_usd"],
                    latency_ms=done["latency_ms"]))
            _finish_query(db, query_id, started, None, status={"searches": statuses},
                          results=[{"parent_id": p["parent_id"], "citation": p["citation"]} for p in shown])
            yield _event({"type": "done", "model": done["model"], "cost_usd": done["cost_usd"],
                          "latency_ms": int((time.time() - started) * 1000)})

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/queries/{query_id}")
    def get_query(query_id: int, request: Request, user: dict = CurrentUser) -> dict[str, Any]:
        with request.app.state.db.connect() as conn:
            q = conn.execute(select(store.queries).where(store.queries.c.id == query_id)).mappings().first()
            if q is None or (q["user_id"] != user["id"] and user["role"] != "admin"):
                raise HTTPException(404, "Not found")
            a = conn.execute(select(store.answers).where(store.answers.c.query_id == query_id)).mappings().first()
            fb = conn.execute(select(store.feedback).where(store.feedback.c.query_id == query_id)
                              .order_by(store.feedback.c.id)).mappings().all()
        return {"query": dict(q), "answer": dict(a) if a else None, "feedback": [dict(f) for f in fb]}

    @app.post("/feedback")
    def give_feedback(body: FeedbackBody, request: Request, user: dict = CurrentUser) -> dict[str, Any]:
        db = request.app.state.db
        with db.begin() as conn:
            q = conn.execute(select(store.queries.c.user_id).where(store.queries.c.id == body.query_id)).first()
            if q is None or (q.user_id != user["id"] and user["role"] != "admin"):
                raise HTTPException(404, "Unknown query_id")
            feedback_id = conn.execute(store.feedback.insert().values(
                query_id=body.query_id, user_id=user["id"], rating=body.rating, comment=body.comment,
                corrected_answer=body.corrected_answer,
                passages=[p.model_dump() for p in body.passages], missing_sources=body.missing_sources,
            )).inserted_primary_key[0]
        return {"feedback_id": feedback_id}

    @app.get("/admin/export", dependencies=[AdminOnly])
    def export(request: Request, since_id: int = 0, limit: int = 500) -> dict[str, Any]:
        """Queries with their answers and feedback, for building eval sets."""
        with request.app.state.db.connect() as conn:
            qs = conn.execute(select(store.queries).where(store.queries.c.id > since_id)
                              .order_by(store.queries.c.id).limit(min(limit, 5000))).mappings().all()
            ids = [q["id"] for q in qs]
            ans = {a["query_id"]: dict(a) for a in conn.execute(
                select(store.answers).where(store.answers.c.query_id.in_(ids))).mappings()} if ids else {}
            fbs: dict[int, list] = {}
            for f in (conn.execute(select(store.feedback).where(store.feedback.c.query_id.in_(ids)))
                      .mappings() if ids else []):
                fbs.setdefault(f["query_id"], []).append(dict(f))
        return {"items": [{**dict(q), "answer": ans.get(q["id"]), "feedback": fbs.get(q["id"], [])} for q in qs]}


app = create_app()
