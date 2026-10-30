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
        return "sqlite:///" + str(Path(os.environ.get("DALIL_DATA_DIR", "data")) / "daleel_api.sqlite")
    # Railway hands out postgres:// or postgresql://; SQLAlchemy needs the driver named.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


# The stores search needs, relative to the db directory. The compact index is only
# required when DALIL_DENSE_COMPACT_DIMS is set.
STORE_FILES = ["faiss_metadata.sqlite", "parents.sqlite", "bm25.sqlite", "dense_faiss_ar.index"]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DALIL_DATA_DIR", "data")))
    database_url: str = field(default_factory=_database_url)
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip())
    admin_token: str = field(default_factory=lambda: os.environ.get("ADMIN_TOKEN", "").strip())

    answer_model: str = field(default_factory=lambda: os.environ.get("ANSWER_MODEL", "anthropic/claude-sonnet-5.5"))
    answer_max_tokens: int = field(default_factory=lambda: _int("ANSWER_MAX_TOKENS", 2500))
    answer_passages: int = field(default_factory=lambda: _int("ANSWER_PASSAGES", 10))
    passage_max_chars: int = field(default_factory=lambda: _int("PASSAGE_MAX_CHARS", 3000))

    # Spend caps. A user's asks per UTC day, everyone's asks per UTC day, and the
    # OpenRouter cost per UTC day after which /ask refuses until midnight UTC.
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

    def store_files(self) -> list[str]:
        files = list(STORE_FILES)
        if self.compact_dims:
            files.append(f"dense_compact_{self.compact_dims}_f16.index")
        return files
