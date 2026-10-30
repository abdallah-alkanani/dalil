"""Invite-only access. An admin mints invite codes; an expert redeems one for a
bearer token. Only SHA-256 hashes of codes and tokens are stored."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select, update

from . import store


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def new_secret(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(24)}"


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def require_admin(request: Request, authorization: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.admin_token
    if not expected:
        raise HTTPException(503, "ADMIN_TOKEN is not configured")
    if not hmac.compare_digest(_bearer(authorization), expected):
        raise HTTPException(403, "Not an admin token")


def current_user(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token_hash = digest(_bearer(authorization))
    with request.app.state.db.connect() as conn:
        row = conn.execute(select(store.users).where(store.users.c.token_hash == token_hash)).mappings().first()
    if row is None or row["disabled"]:
        raise HTTPException(401, "Unknown or disabled token")
    return dict(row)


def create_invite(db, label: str, role: str, max_uses: int) -> str:
    code = new_secret("inv")
    with db.begin() as conn:
        conn.execute(store.invites.insert().values(
            code_hash=digest(code), label=label, role=role, max_uses=max_uses, uses=0))
    return code


def redeem_invite(db, code: str, name: str, email: str | None) -> tuple[dict[str, Any], str]:
    code_hash = digest(code)
    token = new_secret("dlt")
    with db.begin() as conn:
        # Conditional increment, so two redemptions racing for the last use can't both win.
        claimed = conn.execute(
            update(store.invites)
            .where(store.invites.c.code_hash == code_hash, store.invites.c.uses < store.invites.c.max_uses)
            .values(uses=store.invites.c.uses + 1)
        ).rowcount
        if not claimed:
            raise HTTPException(403, "Invite code is invalid or used up")
        invite = conn.execute(select(store.invites).where(store.invites.c.code_hash == code_hash)).mappings().one()
        user_id = conn.execute(store.users.insert().values(
            name=name, email=email, role=invite["role"], token_hash=digest(token),
            invite_code_hash=code_hash, disabled=False,
        )).inserted_primary_key[0]
        user = conn.execute(select(store.users).where(store.users.c.id == user_id)).mappings().one()
    return dict(user), token


CurrentUser = Depends(current_user)
AdminOnly = Depends(require_admin)
