"""The country a request comes from, as a two-letter code.

The web app passes the country its host reports (Vercel's ``x-vercel-ip-country``,
Cloudflare's ``cf-ipcountry``) when there is one, and otherwise the visitor's IP,
which is looked up in an offline DB-IP country database (``GEOIP_DB``). The IP
is used for that lookup only: it is never stored or logged.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, Request


@lru_cache(maxsize=2)
def _reader(path: str):
    import maxminddb

    return maxminddb.open_database(path)


def lookup(host_country: str | None, ip: str | None, db_path: str) -> str | None:
    code = (host_country or "").strip().upper()
    if len(code) == 2 and code.isalnum() and code != "XX":  # Cloudflare sends XX for unknown
        return code
    if not ip or not db_path:
        return None
    try:
        record = _reader(db_path).get(ip.strip())
    except (ValueError, OSError):
        return None
    return ((record or {}).get("country") or {}).get("iso_code")


def request_country(request: Request, x_dalil_country: str | None = Header(default=None),
                    x_dalil_ip: str | None = Header(default=None)) -> str | None:
    return lookup(x_dalil_country, x_dalil_ip, request.app.state.settings.geoip_db)


Country = Depends(request_country)
