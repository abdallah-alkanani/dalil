"""The search engine behind the API: getting the stores onto the volume, opening
them once, and running one query at a time.

``QueryEngine`` keeps per-query state on the instance, so queries are serialised
with a lock; the answer step runs outside it.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Any

from .config import Settings

log = logging.getLogger("dalil_api.search")


class SearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = "starting"  # starting | downloading | building | loading | ready | failed
        self.detail = ""
        self._engine = None
        self._lock = threading.Lock()

    # ── startup ───────────────────────────────────────────────────────────────
    def start_background(self) -> None:
        threading.Thread(target=self._prepare, name="stores-prepare", daemon=True).start()

    def _prepare(self) -> None:
        try:
            missing = self.missing_files()
            compact = self.settings.compact_file()
            to_download = [name for name in missing if name != compact]
            if to_download:
                self.state = "downloading"
                self.download(to_download)
            if compact in missing:
                # The compact index is not on the dataset; it is derived from the
                # full index on the host, in one pass over it.
                from dalil.s7_index.compact import build

                self.state, self.detail = "building", f"building {compact}"
                build(self.settings.db_dir / "dense_faiss_ar.index", [self.settings.compact_dims],
                      self.settings.db_dir)
            self.state = "loading"
            self.open_engine()
            self.detail = "warming caches"
            # The first query after boot reads cold SQLite pages; pay for that here
            # rather than in a user's request.
            self._engine.query("حكم الصلاة", n_results=3, retrieval_mode="bm25")
            self.state, self.detail = "ready", ""
        except Exception as exc:  # reported by /health; the process stays up so it can be inspected
            log.exception("search stores failed to load")
            self.state, self.detail = "failed", str(exc)

    def missing_files(self) -> list[str]:
        db_dir = self.settings.db_dir
        return [name for name in self.settings.store_files() if not (db_dir / name).exists()]

    def download(self, names: list[str]) -> None:
        """Fetch store files from the Hugging Face repo into the db directory.

        Files are downloaded under ``.download`` on the same volume and renamed into
        place when complete, so a half-downloaded file never looks present.
        """
        from huggingface_hub import hf_hub_download

        s = self.settings
        if not s.hf_data_repo:
            raise RuntimeError(f"missing store files {names} and HF_DATA_REPO is not set")
        s.db_dir.mkdir(parents=True, exist_ok=True)
        staging = s.data_dir / ".download"
        for name in names:
            self.detail = f"downloading {name}"
            log.info("downloading %s from %s", name, s.hf_data_repo)
            started = time.time()
            path = hf_hub_download(
                repo_id=s.hf_data_repo, repo_type=s.hf_repo_type, revision=s.hf_data_revision,
                filename=f"{s.hf_data_prefix}{name}", local_dir=staging,
                token=os.environ.get("HF_TOKEN") or None,
            )
            os.replace(path, s.db_dir / name)
            log.info("downloaded %s in %.0fs", name, time.time() - started)
        shutil.rmtree(staging, ignore_errors=True)

    def open_engine(self) -> None:
        os.environ.setdefault("DALIL_DATA_DIR", str(self.settings.data_dir))
        from dalil.s8_search.engine import QueryEngine

        engine = QueryEngine(db_dir=self.settings.db_dir, api_key=self.settings.openrouter_api_key)
        self.detail = "validating stores"
        engine.validate_startup(require_dense=True)
        self._engine = engine

    # ── queries ───────────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def query(self, question: str, **kwargs: Any) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"search is not ready ({self.state})")
        with self._lock:
            return self._engine.query(question, **kwargs)

    def list_sources(self) -> str:
        with self._lock:
            return self._engine.list_sources()


def compact_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """The fields a client needs from one engine result."""
    text = result.get("text_for_synthesis") or result.get("text_original") or ""
    return {
        "parent_id": result.get("parent_id"),
        "citation": result.get("citation"),
        "category": result.get("category"),
        "book_id": result.get("book_id"),
        "section_path": result.get("section_path") or [],
        "page_number": result.get("page_number"),
        "refs": result.get("refs") or {},
        # al-Albani's grade, stored for the four Sunan only.
        "grade": ((result.get("refs") or {}).get("hadith") or {}).get("albani_grade"),
        "text": text[:max_chars] if max_chars else text,
        "truncated": bool(max_chars) and len(text) > max_chars,
        "retrieval_sources": result.get("retrieval_sources") or [],
        "neighbours": [
            {"parent_id": n.get("parent_id"), "offset": n.get("offset"), "citation": n.get("citation"),
             "text": (n.get("text") or "")[:max_chars // 2] if max_chars else n.get("text") or ""}
            for n in result.get("neighbours") or []
        ],
    }
