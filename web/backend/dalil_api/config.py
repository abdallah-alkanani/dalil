"""Settings, all from the environment (Railway variables in production)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, "").strip() or default)


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, "").strip() or default)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return "sqlite:///" + str(Path(os.environ.get("DALIL_DATA_DIR", "data")) / "dalil_api.sqlite")
    # Railway hands out postgres:// or postgresql://; SQLAlchemy needs the driver named.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _default_prompt_path() -> Path:
    """prompts/answering_system_prompt.md at the repository root, or the copy the
    Docker image puts in /opt/dalil."""
    name = Path("prompts") / "answering_system_prompt.md"
    parents = Path(__file__).resolve().parents
    roots = ([parents[3]] if len(parents) > 3 else []) + [Path("/opt/dalil")]
    for root in roots:
        if (root / name).exists():
            return root / name
    return roots[0] / name


# The stores search needs, relative to the db directory. The compact index is only
# required when DALIL_DENSE_COMPACT_DIMS is set, and is built on the host.
STORE_FILES = ["faiss_metadata.sqlite", "parents.sqlite", "bm25.sqlite", "dense_faiss_ar.index"]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DALIL_DATA_DIR", "data")))
    database_url: str = field(default_factory=_database_url)
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip())
    admin_token: str = field(default_factory=lambda: os.environ.get("ADMIN_TOKEN", "").strip())
    # Shared with the web app's server, which calls the API for every visitor.
    site_token: str = field(default_factory=lambda: os.environ.get("SITE_TOKEN", "").strip())
    # Offline DB-IP country database (.mmdb), for requests whose host sends no country.
    geoip_db: str = field(default_factory=lambda: os.environ.get("GEOIP_DB", "").strip())

    # The answer model: any OpenAI-compatible chat endpoint that streams.
    answer_base_url: str = field(default_factory=lambda: os.environ.get(
        "ANSWER_BASE_URL", "https://ollama.com/v1").rstrip("/"))
    answer_api_key: str = field(default_factory=lambda: (
        os.environ.get("ANSWER_API_KEY", "") or os.environ.get("OLLAMA_API_KEY", "")).strip())
    answer_model: str = field(default_factory=lambda: os.environ.get("ANSWER_MODEL", "deepseek-v4.1-flash:1017"))
    # Sent as reasoning_effort; empty sends nothing. "none" keeps the whole token
    # budget for the answer.
    answer_reasoning_effort: str = field(default_factory=lambda: os.environ.get("ANSWER_REASONING_EFFORT", "none"))
    answer_max_tokens: int = field(default_factory=lambda: _int("ANSWER_MAX_TOKENS", 6000))
    answer_passages: int = field(default_factory=lambda: _int("ANSWER_PASSAGES", 8))
    passage_max_chars: int = field(default_factory=lambda: _int("PASSAGE_MAX_CHARS", 3000))
    # The repository's answering prompt; the OpenUI component instructions
    # (dalil_api/answer_components.md) are appended to it.
    answer_prompt_path: Path = field(default_factory=lambda: Path(
        os.environ.get("ANSWER_PROMPT_PATH", "") or _default_prompt_path()))

    # Caps. A user's asks per UTC day, everyone's asks per UTC day, and the
    # reported model cost per UTC day after which /ask refuses until midnight UTC.
    user_daily_asks: int = field(default_factory=lambda: _int("USER_DAILY_ASKS", 30))
    user_daily_searches: int = field(default_factory=lambda: _int("USER_DAILY_SEARCHES", 300))
    global_daily_asks: int = field(default_factory=lambda: _int("GLOBAL_DAILY_ASKS", 500))
    daily_spend_cap_usd: float = field(default_factory=lambda: _float("DAILY_SPEND_CAP_USD", 10.0))

    # Where the stores come from on first boot, when the volume is empty.
    hf_data_repo: str = field(default_factory=lambda: os.environ.get("HF_DATA_REPO", "").strip())
    hf_data_revision: str = field(default_factory=lambda: os.environ.get("HF_DATA_REVISION", "").strip() or "main")
    hf_data_prefix: str = field(default_factory=lambda: os.environ.get("HF_DATA_PREFIX", "").strip())
    hf_repo_type: str = field(default_factory=lambda: os.environ.get("HF_REPO_TYPE", "dataset"))

    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()])

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def compact_dims(self) -> int | None:
        dims = os.environ.get("DALIL_DENSE_COMPACT_DIMS", "").strip()
        return int(dims) if dims else None

    def compact_file(self) -> str | None:
        return f"dense_compact_{self.compact_dims}_f16.index" if self.compact_dims else None

    def store_files(self) -> list[str]:
        return STORE_FILES + ([self.compact_file()] if self.compact_dims else [])
