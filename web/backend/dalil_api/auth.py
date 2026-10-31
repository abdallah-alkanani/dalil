"""Who is asking. The site is open to everyone, with no sign-in.

The web app's server calls the API with ``SITE_TOKEN`` and the browser's visitor
id, a random value it keeps in a cookie. Each visitor id is one row of ``users``,
stored only as its SHA-256 hash, with the name the visitor chose to give, if any.
No IP address or other trace of the visitor is stored. The admin routes take
``ADMIN_TOKEN``.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from . import store


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def _require_token(authorization: str | None, expected: str, name: str) -> None:
    if not expected:
        raise HTTPException(503, f"{name} is not configured")
    if not hmac.compare_digest(_bearer(authorization), expected):
        raise HTTPException(403, f"Not the {name}")


def require_admin(request: Request, authorization: str | None = Header(default=None)) -> None:
    _require_token(authorization, request.app.state.settings.admin_token, "ADMIN_TOKEN")


def _visitor(request: Request, authorization: str | None, visitor_id: str | None, create: bool) -> dict[str, Any] | None:
    _require_token(authorization, request.app.state.settings.site_token, "SITE_TOKEN")
    visitor_id = (visitor_id or "").strip()
    if not 16 <= len(visitor_id) <= 100:
        raise HTTPException(400, "Missing visitor id")
    token_hash = digest(visitor_id)
    find = select(store.users).where(store.users.c.token_hash == token_hash)
    db = request.app.state.db
    with db.connect() as conn:
        row = conn.execute(find).mappings().first()
    if row is None and create:
        try:
            with db.begin() as conn:
                conn.execute(store.users.insert().values(name="", role="visitor", token_hash=token_hash,
                                                         disabled=False))
        except IntegrityError:  # the same visitor's other first request won the insert
            pass
        with db.connect() as conn:
            row = conn.execute(find).mappings().one()
    if row is not None and row["disabled"]:
        raise HTTPException(403, "This visitor is blocked")
    return dict(row) if row is not None else None


def current_user(request: Request, authorization: str | None = Header(default=None),
                 x_dalil_visitor: str | None = Header(default=None)) -> dict[str, Any]:
    """The visitor the web app speaks for, created when they first ask, search or give a name."""
    return _visitor(request, authorization, x_dalil_visitor, create=True)


def known_user(request: Request, authorization: str | None = Header(default=None),
               x_dalil_visitor: str | None = Header(default=None)) -> dict[str, Any] | None:
    """The visitor if they have asked, searched or given a name before; opening the page creates nothing."""
    return _visitor(request, authorization, x_dalil_visitor, create=False)


def set_name(db, user_id: int, name: str) -> None:
    with db.begin() as conn:
        conn.execute(store.users.update().where(store.users.c.id == user_id).values(name=name))


CurrentUser = Depends(current_user)
KnownUser = Depends(known_user)
AdminOnly = Depends(require_admin)
