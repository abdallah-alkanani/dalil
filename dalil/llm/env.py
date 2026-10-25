"""API keys: environment variables, the repo's optional ``.env`` file, and
``~/.config/openrouter.key`` for the OpenRouter key. Keys are never committed."""
from __future__ import annotations

import os
from pathlib import Path

from dalil.paths import REPO_ROOT

ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Copy KEY=VALUE lines into os.environ without overriding values already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key(name: str) -> str:
    """The key in environment variable ``name``, or "" when unset.

    ``OPENROUTER_API_KEY`` alone may also live in ``~/.config/openrouter.key``;
    a custom variable name never falls back to that file.
    """
    key = os.environ.get(name, "").strip()
    if key or name != "OPENROUTER_API_KEY":
        return key
    key_path = Path.home() / ".config" / "openrouter.key"
    return key_path.read_text(encoding="utf-8").strip() if key_path.exists() else ""
