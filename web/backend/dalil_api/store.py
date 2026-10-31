"""The app's own database: visitors (``users``), the question log, answers and feedback.

Postgres on Railway, SQLite locally. Tables are created on startup.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    case,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine

metadata = MetaData()


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(200), nullable=False, default=""),  # what the visitor chose to give, or ""
    Column("role", String(20), nullable=False, default="visitor"),
    Column("token_hash", String(64), nullable=False, unique=True),  # SHA-256 of the visitor id
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
)

# One row per /search or /ask call: the question log.
queries = Table(
    "queries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False, index=True),
    Column("kind", String(10), nullable=False),  # search | ask
    Column("question", Text, nullable=False),
    Column("params", JSON, nullable=False, default=dict),
    Column("status", JSON, nullable=False, default=dict),  # what actually ran: retrieval, planner, rerank
    Column("results", JSON, nullable=False, default=list),  # parent ids and citations, in rank order
    Column("latency_ms", Integer, nullable=True),
    Column("error", Text, nullable=True),
    Column("country", String(2), nullable=True),  # two-letter code of where it was asked from
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow, index=True),
)

answers = Table(
    "answers", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_id", Integer, ForeignKey("queries.id"), nullable=False, unique=True),
    Column("model", String(200), nullable=False),
    Column("prompt_version", String(40), nullable=False),
    Column("text", Text, nullable=False),
    Column("passages", JSON, nullable=False, default=list),  # what the model was shown, numbered
    Column("prompt_tokens", Integer, nullable=True),
    Column("completion_tokens", Integer, nullable=True),
    Column("cost_usd", Float, nullable=True),
    Column("latency_ms", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow, index=True),
)

feedback = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_id", Integer, ForeignKey("queries.id"), nullable=False, index=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    # Overall verdict on the answer (or on the result list, for a search).
    Column("rating", String(20), nullable=True),  # correct | partly_correct | incorrect | unsafe
    Column("comment", Text, nullable=True),
    Column("corrected_answer", Text, nullable=True),
    # Per-passage judgments: [{"parent_id": ..., "relevance": "direct|supporting|topical|irrelevant",
    # "citation_ok": bool, "note": ...}]. These become eval qrels.
    Column("passages", JSON, nullable=False, default=list),
    # Sources the reader says were missing: free text or references.
    Column("missing_sources", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
)


def connect(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    _add_new_columns(engine)
    return engine


def _add_new_columns(engine: Engine) -> None:
    """``create_all`` makes missing tables, not missing columns: add the nullable
    columns a table gained since it was created."""
    existing = inspect(engine)
    for table in metadata.sorted_tables:
        have = {c["name"] for c in existing.get_columns(table.name)}
        for column in table.columns:
            if column.name not in have and column.nullable:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} "
                                      f"{column.type.compile(engine.dialect)}"))


def start_of_utc_day() -> dt.datetime:
    now = utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def count_today(db: Engine, kind: str, user_id: int | None = None) -> int:
    stmt = select(func.count()).select_from(queries).where(
        queries.c.kind == kind, queries.c.created_at >= start_of_utc_day())
    if user_id is not None:
        stmt = stmt.where(queries.c.user_id == user_id)
    with db.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def spend_today(db: Engine) -> float:
    stmt = select(func.coalesce(func.sum(answers.c.cost_usd), 0.0)).where(
        answers.c.created_at >= start_of_utc_day())
    with db.connect() as conn:
        return float(conn.execute(stmt).scalar_one())


def visitors(db: Engine) -> dict[str, list[dict[str, Any]]]:
    """Every visitor with the name they gave, their questions and searches, the
    countries they asked from and when; and every country with its visitors and questions."""
    asks = func.sum(case((queries.c.kind == "ask", 1), else_=0))
    per_visitor = (
        select(users.c.id, users.c.name, users.c.created_at, asks.label("asks"),
               func.count(queries.c.id).label("calls"), func.max(queries.c.created_at).label("last_seen"))
        .select_from(users.outerjoin(queries, queries.c.user_id == users.c.id))
        .group_by(users.c.id, users.c.name, users.c.created_at)
        .order_by(func.max(queries.c.created_at).desc()))
    per_country = (
        select(queries.c.country, queries.c.user_id, asks.label("asks"))
        .where(queries.c.country.is_not(None))
        .group_by(queries.c.country, queries.c.user_id))
    with db.connect() as conn:
        people = conn.execute(per_visitor).mappings().all()
        places = conn.execute(per_country).mappings().all()
    countries_of: dict[int, list[str]] = {}
    by_country: dict[str, dict[str, Any]] = {}
    for r in places:
        countries_of.setdefault(r["user_id"], []).append(r["country"])
        c = by_country.setdefault(r["country"], {"country": r["country"], "visitors": 0, "asks": 0})
        c["visitors"] += 1
        c["asks"] += int(r["asks"] or 0)
    return {
        "visitors": [{"id": r["id"], "name": r["name"], "countries": sorted(countries_of.get(r["id"], [])),
                      "asks": int(r["asks"] or 0), "searches": int(r["calls"]) - int(r["asks"] or 0),
                      "first_seen": r["created_at"], "last_seen": r["last_seen"]} for r in people],
        "countries": sorted(by_country.values(), key=lambda c: -c["asks"]),
    }
