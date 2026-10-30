"""The app's own database: users and invites, the question log, answers and feedback.

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
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


invites = Table(
    "invites", metadata,
    Column("code_hash", String(64), primary_key=True),
    Column("label", String(200), nullable=False, default=""),
    Column("role", String(20), nullable=False, default="expert"),
    Column("max_uses", Integer, nullable=False, default=1),
    Column("uses", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
)

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(200), nullable=False),
    Column("email", String(320), nullable=True),
    Column("role", String(20), nullable=False, default="expert"),
    Column("token_hash", String(64), nullable=False, unique=True),
    Column("invite_code_hash", String(64), ForeignKey("invites.code_hash"), nullable=True),
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
    # Sources the expert says were missing: free text or references.
    Column("missing_sources", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
)


def connect(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine


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
