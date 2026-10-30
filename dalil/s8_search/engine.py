"""The query engine: hybrid retrieval over the parent/child corpus.

Stores, in one directory (default ``data/db``):
  dense_faiss_ar.index    exact dense index (IndexFlatIP, 4096-d)
  faiss_metadata.sqlite   one row per indexed retrieval view (child), keyed by faiss_row
  parents.sqlite          parent passages: the evidence handed to the answering model
  bm25.sqlite             FTS5 lexical index over the children
  dense_compact_<dims>_f16.index  optional fp16 copy of the first <dims> dims, for
                          two-stage search (``DALIL_DENSE_COMPACT_DIMS``)

Pipeline per question:
  1. Optionally plan intent-checked query variants (``planning.py``).
  2. For each variant, retrieve children with BM25 and/or dense search and group
     them by parent; fuse the two rankings with weighted RRF.
  3. Fuse variant rankings with RRF, apply the source-category prior, MMR, and an
     optional listwise LLM rerank.
  4. Window long parents around the match and attach reading-order neighbours.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from dalil.s7_index.bm25 import staleness as bm25_staleness

from . import planning
from .api import (
    EMBED_DIM,
    EMBED_INSTRUCTION_AR,
    EMBED_MODEL,
    LISTWISE_MODEL,
    OPENROUTER_URL,
    PLANNER_MODEL,
    PLANNER_PROVIDER,
    PLANNER_TIMEOUT_SECONDS,
    request_with_retries,
)
from .dense import CompactDenseIndex, DenseIndex
from .filters import (
    build_bm25_where,
    build_child_filter_where,
    build_parent_filter_where,
    format_filter_summary,
    normalize_filters,
)
from .ranking import (
    fuse_parent_hits_rrf,
    mmr_select,
    normalize_arabic_for_search,
    parse_listwise_ranking,
    sanitize_fts_query,
    window_parent_text,
)
from .sources import (
    CATEGORY_ORDER,
    CATEGORY_TIERS,
    DEFAULT_TIER,
    apply_source_prior,
    book_label,
    detect_source_boost_keywords,
    render_citation,
)

DEFAULT_DENSE_TOP_K = 200
DEFAULT_BM25_TOP_K = 200
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_MMR_DEPTH = 80
LISTWISE_DEPTH = 20
MIN_PARENT_POOL_FACTOR = 5
# Dense retrieval widens k (doubling) until enough distinct parents survive the
# filters, for at most this many rounds.
MAX_SEARCH_ROUNDS = 5


def log(message: str) -> None:
    print(message, file=sys.stderr)


def dense_search_depth(n_results: int, dense_top_k: int, filtered: bool) -> int:
    """First-round k of the widening dense search. Filters discard most hits, so
    filtered searches start deeper."""
    return max(dense_top_k, n_results * (50 if filtered else 10))


class QueryEngine:
    def __init__(
        self,
        db_dir: Path,
        api_key: str,
        openrouter_url: str = OPENROUTER_URL,
        embed_model: str = EMBED_MODEL,
        listwise_model: str = LISTWISE_MODEL,
        planner_model: str = PLANNER_MODEL,
        planner_provider: str = PLANNER_PROVIDER,
    ) -> None:
        self.db_dir = db_dir
        self.faiss_path = db_dir / "dense_faiss_ar.index"
        self.child_db_path = db_dir / "faiss_metadata.sqlite"
        self.parent_db_path = db_dir / "parents.sqlite"
        self.bm25_db_path = db_dir / "bm25.sqlite"
        self.api_key = api_key
        self.openrouter_url = openrouter_url.rstrip("/")
        self.embed_model = embed_model
        self.listwise_model = listwise_model
        self.planner_model = planner_model
        self.planner_provider = planner_provider
        self._dense_index: DenseIndex | CompactDenseIndex | None = None
        self._child_conn: sqlite3.Connection | None = None
        self._parent_conn: sqlite3.Connection | None = None
        self._bm25_conn: sqlite3.Connection | None = None
        self._child_cache: dict[int, dict[str, Any] | None] = {}
        self._parent_cache: dict[str, dict[str, Any] | None] = {}
        self._prefetch: _QueryPrefetch | None = None
        # What actually ran in the current query(): returned as payload["status"].
        self._status: dict[str, str] = {}

    def close(self) -> None:
        for conn in (self._child_conn, self._parent_conn, self._bm25_conn):
            if conn is not None:
                conn.close()

    # ── stores ────────────────────────────────────────────────────────────────
    def validate_startup(self, *, require_dense: bool = True) -> None:
        missing = [str(p) for p in (self.child_db_path, self.parent_db_path) if not p.exists()]
        if require_dense and not self.faiss_path.exists():
            missing.insert(0, str(self.faiss_path))
        if missing:
            raise RuntimeError("Missing required DB files:\n- " + "\n- ".join(missing))

        child_count = self.child_count()
        parent_count = self.parent_db().execute("SELECT COUNT(*) FROM parents").fetchone()[0]
        if child_count <= 0:
            raise RuntimeError(f"child_metadata is empty: {self.child_db_path}")
        if parent_count <= 0:
            raise RuntimeError(f"parents is empty: {self.parent_db_path}")

        if require_dense:
            index = self.load_faiss_index()
            if index.d != EMBED_DIM:
                raise RuntimeError(f"FAISS dimension {index.d} != expected {EMBED_DIM}")
            if index.ntotal != child_count:
                raise RuntimeError(
                    f"FAISS rows {index.ntotal:,} != child_metadata rows {child_count:,}. "
                    "The index and metadata DB are not aligned."
                )

    def child_count(self) -> int:
        return self.child_db().execute("SELECT COUNT(*) FROM child_metadata").fetchone()[0]

    def child_db(self) -> sqlite3.Connection:
        if self._child_conn is None:
            self._child_conn = open_sqlite_readonly(self.child_db_path)
            self._child_conn.row_factory = sqlite3.Row
            require_table(self._child_conn, "child_metadata", self.child_db_path)
        return self._child_conn

    def parent_db(self) -> sqlite3.Connection:
        if self._parent_conn is None:
            self._parent_conn = open_sqlite_readonly(self.parent_db_path)
            self._parent_conn.row_factory = sqlite3.Row
            require_table(self._parent_conn, "parents", self.parent_db_path)
        return self._parent_conn

    def bm25_db(self) -> sqlite3.Connection:
        """Read-only lexical index. A missing or stale index is an error, not a
        silent multi-minute rebuild from inside a query."""
        if self._bm25_conn is None:
            if not self.bm25_db_path.exists():
                raise RuntimeError(
                    f"Missing {self.bm25_db_path}; build it with: python -m dalil.s7_index.bm25 --db {self.db_dir}")
            conn = open_sqlite_readonly(self.bm25_db_path)
            conn.row_factory = sqlite3.Row
            problem = bm25_staleness(conn, self.child_count())
            if problem:
                conn.close()
                raise RuntimeError(
                    f"BM25 index does not match the metadata store: {problem}. "
                    f"Rebuild it with: python -m dalil.s7_index.bm25 --db {self.db_dir}")
            self._bm25_conn = conn
        return self._bm25_conn

    def load_faiss_index(self) -> DenseIndex | CompactDenseIndex:
        """The exact streamed index, or with ``DALIL_DENSE_COMPACT_DIMS`` set (for
        example 1024) the two-stage compact index, for hosts with slow disks."""
        if self._dense_index is None:
            try:
                dims = os.environ.get("DALIL_DENSE_COMPACT_DIMS", "").strip()
                if dims:
                    from dalil.s7_index.compact import compact_path

                    rescore = int(os.environ.get("DALIL_DENSE_RESCORE", "1000"))
                    self._dense_index = CompactDenseIndex(
                        self.faiss_path, compact_path(self.db_dir, int(dims)), rescore=rescore)
                    log(f"  [Dense] Compact index: {dims} dims in RAM, exact rescore of top {rescore}")
                else:
                    self._dense_index = DenseIndex(self.faiss_path)
            except ImportError as exc:
                raise RuntimeError("Missing dependency: install faiss-cpu in this Python environment") from exc
        return self._dense_index

    def get_child_meta(self, faiss_row: int) -> dict[str, Any] | None:
        if faiss_row not in self._child_cache:
            self.prefetch_child_meta([faiss_row])
        return self._child_cache[faiss_row]

    def prefetch_child_meta(self, rows: Iterable[int]) -> None:
        """Load child rows into the cache in batched queries instead of one each."""
        missing = [r for r in dict.fromkeys(int(x) for x in rows) if r >= 0 and r not in self._child_cache]
        for chunk in chunks(missing, 500):
            placeholders = ",".join("?" * len(chunk))
            for row in self.child_db().execute(
                f"""SELECT faiss_row, view_id, parent_id, source, book_id, category,
                           subcategory, view_type, text_role, qwen_tokens, source_line,
                           section_path_json, section_path_ids_json, refs_json,
                           display_text, embedding_text, metadata_json
                    FROM child_metadata
                    WHERE faiss_row IN ({placeholders})""",
                chunk,
            ):
                rec = dict(row)
                rec["section_path"] = json_loads_or(rec.get("section_path_json"), [])
                rec["section_path_ids"] = json_loads_or(rec.get("section_path_ids_json"), [])
                rec["refs"] = json_loads_or(rec.get("refs_json"), {})
                rec["metadata"] = json_loads_or(rec.get("metadata_json"), {})
                self._child_cache[rec["faiss_row"]] = rec
            for r in chunk:
                self._child_cache.setdefault(r, None)

    def get_parents_batch(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        uncached = [pid for pid in parent_ids if pid not in self._parent_cache]
        for pid in parent_ids:
            cached = self._parent_cache.get(pid)
            if cached is not None:
                result[pid] = cached

        conn = self.parent_db()
        for chunk in chunks(uncached, 500):
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""SELECT parent_id, book_id, category, subcategory, parent_type,
                           citation, section_path_json, refs_json,
                           text_for_synthesis, text_original, metadata_json,
                           order_in_book, page_number
                    FROM parents
                    WHERE parent_id IN ({placeholders})""",
                chunk,
            ).fetchall()
            for row in rows:
                rec = dict(row)
                rec["section_path"] = json_loads_or(rec.get("section_path_json"), [])
                rec["refs"] = json_loads_or(rec.get("refs_json"), {})
                rec["metadata"] = json_loads_or(rec.get("metadata_json"), {})
                self._parent_cache[rec["parent_id"]] = rec
                result[rec["parent_id"]] = rec

            found = {row["parent_id"] for row in rows}
            for pid in chunk:
                if pid not in found:
                    self._parent_cache[pid] = None

        return result

    # ── remote models ─────────────────────────────────────────────────────────
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def embed_query(self, question: str) -> np.ndarray:
        try:
            return self._embed_query(question)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"query embedding failed: {exc}") from exc

    def _embed_query(self, question: str) -> np.ndarray:
        if not self.api_key:
            raise EmbeddingError("Missing OPENROUTER_API_KEY")

        text = f"{EMBED_INSTRUCTION_AR} {question}"
        resp = request_with_retries(
            "post",
            f"{self.openrouter_url}/embeddings",
            headers=self._auth_headers(),
            json_body={"model": self.embed_model, "input": [text]},
            timeout=90,
        )
        data = resp.json()
        emb = data["data"][0]["embedding"]
        qvec = np.asarray(emb, dtype=np.float32)
        if qvec.ndim != 1:
            raise RuntimeError(f"Bad query embedding shape: {qvec.shape}")
        if len(qvec) != EMBED_DIM:
            raise RuntimeError(f"Query embedding dimension {len(qvec)} != expected {EMBED_DIM}")
        norm = np.linalg.norm(qvec)
        if not np.isfinite(qvec).all() or norm <= 1e-8:
            raise RuntimeError("Query embedding is invalid")
        return qvec / norm

    def _planning_completion(self, prompt: str, max_tokens: int = 800) -> str:
        """Run a deterministic planner/critic completion through the configured LLM."""
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        try:
            resp = request_with_retries(
                "post",
                f"{self.openrouter_url}/chat/completions",
                headers=self._auth_headers(),
                json_body={
                    "model": self.planner_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    **({"provider": {"order": [self.planner_provider], "allow_fallbacks": True}}
                       if self.planner_provider else {}),
                },
                timeout=PLANNER_TIMEOUT_SECONDS,
                retries=0,
            )
        except requests.HTTPError as exc:
            detail = exc.response.text[:1000] if exc.response is not None else ""
            raise RuntimeError(f"planner HTTP error: {detail}") from exc
        message = resp.json()["choices"][0]["message"]
        return (message.get("content") or "").strip()

    def plan_query_variants(
        self,
        question: str,
        max_queries: int = 6,
        use_critic: bool = True,
        force_variants: bool = False,
    ) -> dict[str, Any]:
        """Create intent-preserving Arabic retrieval variants and verify drift.

        Rewrites are retrieval aids only; the exact user question remains
        immutable and is always variant ``Q0``.
        """
        try:
            response = self._planning_completion(
                planning.planner_prompt(question, max_queries, force_variants), max_tokens=900)
        except Exception as exc:
            response = ""
            planner_error = str(exc)
        else:
            planner_error = None
        plan = planning.parse_query_plan_response(response, question, max_queries=max_queries)
        if planner_error:
            plan["error"] = planner_error

        variants = []
        for idx, item in enumerate(plan["queries"]):
            variant = item.copy()
            variant["variant_id"] = f"Q{idx}"
            variants.append(variant)
        plan["queries"] = variants
        plan["critic_decisions"] = {}

        if not use_critic or len(variants) <= 1:
            return plan

        try:
            critic_response = self._planning_completion(
                planning.critic_prompt(question, plan, variants), max_tokens=500)
            accepted, decisions = planning.filter_variants_with_critic_response(variants, critic_response)
            if not decisions:
                plan["critic_error"] = "critic response was invalid or incomplete"
        except Exception as exc:
            accepted, decisions = variants[:1], {}
            plan["critic_error"] = str(exc)
        plan["queries"] = accepted
        plan["critic_decisions"] = decisions
        plan["use_variants"] = len(accepted) > 1
        return plan

    # ── retrieval ─────────────────────────────────────────────────────────────
    def _dense_search(self, question: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(query vector, scores, faiss rows): the exact top-k for ``question``.

        Inside ``query()`` every variant still waiting for dense results is
        embedded concurrently and searched in the same single pass over the index.
        """
        prefetch = self._prefetch
        if prefetch is None:
            log("  [Embed] Embedding query...")
            qvec = self.embed_query(question)
            scores, ids = self.load_faiss_index().search(qvec.reshape(1, -1), k)
            return qvec, scores[0], ids[0]
        if (question, k) not in prefetch.dense:
            batch = [question] + [q for q in prefetch.pending if q != question and (q, k) not in prefetch.dense]
            prefetch.pending = []
            log(f"  [Embed] Embedding {len(batch)} quer{'y' if len(batch) == 1 else 'ies'}...")
            vectors: dict[str, np.ndarray] = {}
            for q in batch:
                try:
                    vectors[q] = prefetch.embedding(self, q).result()
                except Exception as exc:
                    prefetch.dense[(q, k)] = exc
            if vectors:
                scores, ids = self.load_faiss_index().search(np.vstack(list(vectors.values())), k)
                for i, q in enumerate(vectors):
                    prefetch.dense[(q, k)] = (vectors[q], scores[i], ids[i])
        found = prefetch.dense[(question, k)]
        if isinstance(found, Exception):
            raise found
        return found

    def _retrieve_candidates_for_question(
        self,
        question: str,
        *,
        n_results: int,
        dense_top_k: int,
        bm25_top_k: int,
        retrieval_mode: str,
        rrf_k: int,
        dense_weight: float,
        bm25_weight: float,
        filters: dict[str, Any],
        allowed_rows: set[int] | None,
    ) -> dict[str, Any]:
        """Run hybrid retrieval for one query, without final reranking."""
        qvec: np.ndarray | None = None
        index = None
        dense_hits: dict[str, dict[str, Any]] = {}
        bm25_hits: dict[str, dict[str, Any]] = {}
        searched_k = 0
        if retrieval_mode in {"dense", "hybrid"}:
            index = self.load_faiss_index()
            categories = filters.get("category")
            first_k = dense_search_depth(n_results, dense_top_k, bool(categories) or allowed_rows is not None)
            # One search at the deepest round's k; each round then reads a prefix.
            # Exact top-k lists are prefix-stable, so this equals searching per round.
            qvec, scores, ids = self._dense_search(
                question, min(index.ntotal, first_k * 2 ** (MAX_SEARCH_ROUNDS - 1)))
            if index.d != len(qvec):
                raise RuntimeError(f"FAISS dimension {index.d} != query dimension {len(qvec)}")
            dense_hits, searched_k = self.retrieve_dense_parent_hits(
                scores, ids, ntotal=index.ntotal, n_results=n_results, first_k=first_k,
                allowed_rows=allowed_rows, categories=categories,
            )
            log(f"  [Dense] {len(dense_hits)} unique parents from top-{searched_k} child candidates")
        if retrieval_mode in {"bm25", "hybrid"}:
            bm25_hits = self.retrieve_bm25_parent_hits(
                question=question,
                n_results=n_results,
                bm25_top_k=bm25_top_k,
                filters=filters,
                allowed_rows=allowed_rows,
            )
            log(f"  [BM25] {len(bm25_hits)} unique parents from top lexical candidates")

        if retrieval_mode == "dense":
            parent_hits = dense_hits
        else:
            parent_hits = fuse_parent_hits_rrf(
                dense_hits=dense_hits, bm25_hits=bm25_hits, rrf_k=rrf_k,
                dense_weight=dense_weight, bm25_weight=bm25_weight,
            )
            if retrieval_mode == "hybrid":
                log(f"  [Hybrid] {len(parent_hits)} fused parents "
                    f"(RRF k={rrf_k}, dense={dense_weight}, bm25={bm25_weight})")

        if not parent_hits:
            return {
                "candidates": [], "index": index, "qvec": qvec,
                "searched_k": searched_k, "bm25_parent_count": len(bm25_hits),
            }
        parents = self.get_parents_batch(list(parent_hits))
        log(f"  [Parents] Fetched {len(parents)}/{len(parent_hits)} parent records")
        candidates = self.build_candidates(parent_hits, parents)
        # Dense-only candidates carry hybrid_score=None (nothing was fused); rank
        # them by similarity. `.get(key, default)` would return the None itself.
        candidates.sort(key=candidate_score, reverse=True)
        return {
            "candidates": candidates,
            "index": index,
            "qvec": qvec,
            "searched_k": searched_k,
            "bm25_parent_count": len(bm25_hits),
        }

    def query(
        self,
        question: str,
        n_results: int = 10,
        use_listwise_rerank: bool = False,
        mmr_lambda: float = 0.7,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        retrieval_mode: str = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        mmr_depth: int = DEFAULT_MMR_DEPTH,
        categories: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        boost_override: str | None = None,
        parent_max_chars: int = 0,
        expand_neighbours: int = 0,
        query_planning: str = "off",
        max_query_variants: int = 6,
        use_intent_critic: bool = True,
        variant_depth: int = 0,
    ) -> dict[str, Any]:
        """Retrieve evidence, optionally using intent-checked query variants.

        Variants are retrieved in input order. Network waits overlap: the original
        question's embedding and the source-category classification start before
        planning, and all variants share one pass over the dense index.
        """
        filters = normalize_filters(filters or {})
        if categories:
            filters.setdefault("category", categories)
        retrieval_mode = retrieval_mode.lower()
        if retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be dense, bm25, or hybrid")
        query_planning = query_planning.lower()
        if query_planning not in {"off", "auto", "on"}:
            raise ValueError("query_planning must be off, auto, or on")
        require_dense = retrieval_mode in {"dense", "hybrid"}
        self.validate_startup(require_dense=require_dense)

        log(f"  [Pipeline] {retrieval_mode} retrieval -> parent grouping -> "
            f"{'query-variant RRF -> ' if query_planning != 'off' else ''}source prior -> "
            f"{'MMR -> ' if require_dense else ''}"
            f"{'listwise rerank' if use_listwise_rerank else 'no listwise'}")
        if filters:
            log(f"  [Filter] Rich filters: {format_filter_summary(filters)}")
        allowed_rows = self.compute_allowed_faiss_rows(filters)
        if allowed_rows is not None:
            log(f"  [Filter] {len(allowed_rows):,} child rows match metadata filters")
            if not allowed_rows:
                return {"results": [], "question": question, "error": "No rows match filters"}

        self._status = {"retrieval": retrieval_mode, "planner": "off", "rerank": "off", "source_classifier": "off"}
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="dalil-query") as pool:
            self._prefetch = _QueryPrefetch(pool)
            try:
                if require_dense:
                    self._prefetch.embedding(self, question)
                boost_future = pool.submit(self.choose_boost_categories, question, boost_override)
                payload = self._query_variants(
                    question, boost_future, n_results=n_results, use_listwise_rerank=use_listwise_rerank,
                    mmr_lambda=mmr_lambda, dense_top_k=dense_top_k, bm25_top_k=bm25_top_k,
                    retrieval_mode=retrieval_mode, rrf_k=rrf_k, dense_weight=dense_weight,
                    bm25_weight=bm25_weight, mmr_depth=mmr_depth, filters=filters,
                    allowed_rows=allowed_rows, parent_max_chars=parent_max_chars,
                    expand_neighbours=expand_neighbours, query_planning=query_planning,
                    max_query_variants=max_query_variants, use_intent_critic=use_intent_critic,
                    variant_depth=variant_depth,
                )
            except EmbeddingError as exc:
                if retrieval_mode != "hybrid":
                    raise
                log(f"  [Retrieval] {exc}; falling back to BM25")
                fallback_reason = str(exc)
            else:
                payload["status"] = self._status
                return payload
            finally:
                self._prefetch = None
        # Re-run lexically; the payload records why under status.retrieval.
        payload = self.query(
            question, n_results=n_results, use_listwise_rerank=use_listwise_rerank, mmr_lambda=mmr_lambda,
            dense_top_k=dense_top_k, bm25_top_k=bm25_top_k, retrieval_mode="bm25", rrf_k=rrf_k,
            dense_weight=dense_weight, bm25_weight=bm25_weight, mmr_depth=mmr_depth, categories=categories,
            filters=filters, boost_override=boost_override, parent_max_chars=parent_max_chars,
            expand_neighbours=expand_neighbours, query_planning=query_planning,
            max_query_variants=max_query_variants, use_intent_critic=use_intent_critic,
            variant_depth=variant_depth,
        )
        payload["status"]["retrieval"] = f"bm25_fallback({fallback_reason[:200]})"
        return payload

    def _query_variants(
        self,
        question: str,
        boost_future: Future,
        *,
        n_results: int,
        use_listwise_rerank: bool,
        mmr_lambda: float,
        dense_top_k: int,
        bm25_top_k: int,
        retrieval_mode: str,
        rrf_k: int,
        dense_weight: float,
        bm25_weight: float,
        mmr_depth: int,
        filters: dict[str, Any],
        allowed_rows: set[int] | None,
        parent_max_chars: int,
        expand_neighbours: int,
        query_planning: str,
        max_query_variants: int,
        use_intent_critic: bool,
        variant_depth: int,
    ) -> dict[str, Any]:
        require_dense = retrieval_mode in {"dense", "hybrid"}
        if query_planning == "off":
            plan = {
                "use_variants": False,
                "intent": question,
                "forbidden_adjacent": [],
                "critic_decisions": {},
                "queries": [{"variant_id": "Q0", "query": question, "purpose": "original"}],
            }
        else:
            log("  [Planner] Generating intent-preserving retrieval queries...")
            plan = self.plan_query_variants(
                question,
                max_queries=max_query_variants,
                use_critic=use_intent_critic,
                force_variants=query_planning == "on",
            )
            log(f"  [Planner] Accepted {len(plan['queries'])} sequential query variant(s)")
            self._status["planner"] = f"failed({plan['error'][:200]})" if plan.get("error") else "ok"
        if require_dense and self._prefetch is not None:
            self._prefetch.pending = [v["query"] for v in plan["queries"]]

        lossless_variant_pool = not use_listwise_rerank and len(plan["queries"]) > 1
        if lossless_variant_pool:
            # Exhaustive/deep-research mode: preserve each accepted variant's top-N,
            # or an explicitly requested deeper per-variant pool.
            depth = variant_depth if variant_depth > 0 else n_results
        else:
            depth = variant_depth if variant_depth > 0 else max(30, n_results * 5)
        retrieval_target = variant_depth if variant_depth > 0 else n_results
        variant_metadata: dict[str, dict[str, Any]] = {}
        variant_errors: dict[str, str] = {}

        def retrieve_variant(variant: dict[str, Any]) -> list[dict[str, Any]]:
            log(f"  [Variant {variant['variant_id']}] {variant.get('purpose', '')}: {variant['query']}")
            try:
                payload = self._retrieve_candidates_for_question(
                    variant["query"], n_results=retrieval_target, dense_top_k=dense_top_k,
                    bm25_top_k=bm25_top_k, retrieval_mode=retrieval_mode, rrf_k=rrf_k,
                    dense_weight=dense_weight, bm25_weight=bm25_weight, filters=filters,
                    allowed_rows=allowed_rows,
                )
            except Exception as exc:
                if variant["variant_id"] == "Q0":
                    raise
                variant_errors[variant["variant_id"]] = str(exc)
                log(f"  [Variant {variant['variant_id']}] Failed; preserving Q0 results: {exc}")
                return []
            variant_metadata[variant["variant_id"]] = payload
            return payload["candidates"][:depth]

        variant_results = planning.run_variants_sequentially(plan["queries"], retrieve_variant)
        nonempty = {key: value for key, value in variant_results.items() if value}
        if not nonempty:
            return {
                "results": [], "question": question, "error": "No matching results",
                "query_plan": plan, "query_variants_executed": len(plan["queries"]),
                "variant_errors": variant_errors,
            }
        if len(nonempty) == 1:
            candidates = next(iter(nonempty.values()))
            for candidate in candidates:
                candidate.setdefault("query_variant_ids", list(nonempty))
                candidate.setdefault("rank_by_query", {})
        else:
            candidates = planning.fuse_query_variant_rankings(nonempty, rrf_k=rrf_k)

        boost_cats = boost_future.result()
        apply_source_prior(candidates, boost_cats)
        candidates.sort(key=lambda c: c["weighted_score"], reverse=True)

        original_meta = variant_metadata.get("Q0", {})
        index = original_meta.get("index")
        qvec = original_meta.get("qvec")
        if len(candidates) > 1 and index is not None and qvec is not None:
            before = [c["parent_id"] for c in candidates[: min(5, len(candidates))]]
            if mmr_depth > 0 and len(candidates) > mmr_depth:
                log(f"  [MMR] Limiting MMR to top {mmr_depth}/{len(candidates)} candidates")
                candidates = self.apply_mmr(index, qvec, candidates[:mmr_depth], lam=mmr_lambda) + candidates[mmr_depth:]
            elif mmr_depth > 0:
                candidates = self.apply_mmr(index, qvec, candidates, lam=mmr_lambda)
            after = [c["parent_id"] for c in candidates[: min(5, len(candidates))]]
            log(f"  [MMR] {'Applied rerank' if before != after else 'Applied, order unchanged'} "
                f"(lambda={mmr_lambda})")
        elif len(candidates) > 1 and retrieval_mode == "bm25" and mmr_depth > 0:
            log("  [MMR] Skipped for lexical-only retrieval")

        if use_listwise_rerank and candidates:
            log("  [Rerank] LLM listwise reranking...")
            results = self.llm_listwise_rerank(question, candidates, top_k=n_results)
            self._status["rerank"] = "ok" if results and "listwise_rank" in results[0] else self._status.get("rerank", "failed")
        elif lossless_variant_pool:
            results = candidates
        else:
            results = candidates[:n_results]

        if parent_max_chars > 0:
            for result in results:
                window_parent_text(result, planning.candidate_anchor_texts(result), parent_max_chars)

        self.attach_neighbours(results, expand_neighbours)
        searched_values = [int(meta.get("searched_k", 0)) for meta in variant_metadata.values()]
        return {
            "results": results,
            "question": question,
            "query_plan": plan,
            "query_variants_executed": len(plan["queries"]),
            "lossless_variant_pool": lossless_variant_pool,
            "variant_result_counts": {key: len(value) for key, value in variant_results.items()},
            "variant_errors": variant_errors,
            "neighbour_window": expand_neighbours,
            "n_parents_matched": len(candidates),
            "dense_top_k_searched": max(searched_values, default=0),
            "bm25_top_k": bm25_top_k if retrieval_mode in {"bm25", "hybrid"} else 0,
            "retrieval_mode": retrieval_mode,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "mmr_depth": mmr_depth,
            "filters": filters,
            "mmr_lambda": mmr_lambda,
            "listwise_reranked": self._status["rerank"] == "ok",
            "boost_categories": boost_cats,
        }

    def retrieve_dense_parent_hits(
        self,
        scores: np.ndarray,
        ids: np.ndarray,
        *,
        ntotal: int,
        n_results: int,
        first_k: int,
        allowed_rows: set[int] | None,
        categories: list[str] | None,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """Group the dense top-k into parents, widening k until the pool is big enough.

        ``scores``/``ids`` hold the deepest round's exact top-k; round r reads its
        first ``first_k * 2**r`` entries.
        """
        cat_set = set(categories) if categories else None
        min_parent_pool = max(n_results * MIN_PARENT_POOL_FACTOR, n_results + LISTWISE_DEPTH)

        parent_hits: dict[str, dict[str, Any]] = {}
        searched_k = 0
        for round_idx in range(MAX_SEARCH_ROUNDS):
            k_search = min(ntotal, first_k * (2**round_idx))
            if k_search <= searched_k:
                break
            log(f"  [Dense] FAISS retrieval top-{k_search}...")
            searched_k = k_search
            parent_hits = self.group_dense_child_hits(scores[:k_search], ids[:k_search], cat_set, allowed_rows)
            if len(parent_hits) >= min_parent_pool or k_search >= ntotal:
                break
            if not cat_set and allowed_rows is None and len(parent_hits) >= n_results:
                break

        return parent_hits, searched_k

    def group_dense_child_hits(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        cat_set: set[str] | None,
        allowed_rows: set[int] | None,
    ) -> dict[str, dict[str, Any]]:
        self.prefetch_child_meta(
            int(i) for i in indices if i >= 0 and (allowed_rows is None or int(i) in allowed_rows))
        parent_hits: dict[str, dict[str, Any]] = {}
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if idx < 0:
                continue
            row = int(idx)
            if allowed_rows is not None and row not in allowed_rows:
                continue
            child = self.get_child_meta(row)
            if child is None:
                continue
            child_cat = child.get("category") or ""
            if cat_set is not None and child_cat not in cat_set:
                continue

            pid = child["parent_id"]
            score_f = float(score)
            if pid not in parent_hits:
                parent_hits[pid] = {
                    "best_score": score_f,
                    "dense_score": score_f,
                    "dense_rank": rank,
                    "best_faiss_row": row,
                    "child": child,
                    "child_scores": [score_f],
                    "child_rows": [row],
                }
                continue

            hit = parent_hits[pid]
            hit["child_scores"].append(score_f)
            hit["child_rows"].append(row)
            if score_f > hit["best_score"]:
                hit["best_score"] = score_f
                hit["dense_score"] = score_f
                hit["dense_rank"] = rank
                hit["best_faiss_row"] = row
                hit["child"] = child

        return parent_hits

    def retrieve_bm25_parent_hits(
        self,
        question: str,
        n_results: int,
        bm25_top_k: int,
        filters: dict[str, Any],
        allowed_rows: set[int] | None,
    ) -> dict[str, dict[str, Any]]:
        match_query = sanitize_fts_query(normalize_arabic_for_search(question))
        if not match_query:
            return {}

        min_parent_pool = max(n_results * MIN_PARENT_POOL_FACTOR, n_results + LISTWISE_DEPTH)
        limit = max(bm25_top_k, n_results * 50)
        if allowed_rows is not None:
            limit = min(max(limit * 5, n_results * 200), int(self.child_count()))

        where_sql, params = build_bm25_where(filters)
        sql = f"""SELECT rowid AS faiss_row,
                         bm25(child_fts, 2.0, 0.35) AS bm25_score
                  FROM child_fts
                  WHERE child_fts MATCH ?{where_sql}
                  ORDER BY bm25_score
                  LIMIT ?"""
        rows = self.bm25_db().execute(sql, [match_query, *params, limit]).fetchall()

        parent_hits: dict[str, dict[str, Any]] = {}
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            # FTS5 docids are 1-based; FAISS/metadata rows are zero-based.
            self.prefetch_child_meta(
                int(r["faiss_row"]) - 1 for r in batch
                if allowed_rows is None or int(r["faiss_row"]) - 1 in allowed_rows)
            for rank, row_rec in enumerate(batch, start + 1):
                row = int(row_rec["faiss_row"]) - 1
                if allowed_rows is not None and row not in allowed_rows:
                    continue
                child = self.get_child_meta(row)
                if child is None:
                    continue
                score = float(row_rec["bm25_score"])
                lexical_score = 1.0 / (1.0 + max(score, 0.0))
                pid = child["parent_id"]
                if pid not in parent_hits:
                    parent_hits[pid] = {
                        "best_score": lexical_score,
                        "bm25_score": score,
                        "bm25_rank": rank,
                        "best_faiss_row": row,
                        "child": child,
                        "child_scores": [lexical_score],
                        "child_rows": [row],
                        "retrieval_sources": ["bm25"],
                    }
                else:
                    hit = parent_hits[pid]
                    hit["child_scores"].append(lexical_score)
                    hit["child_rows"].append(row)
                    if score < float(hit.get("bm25_score", float("inf"))):
                        hit["best_score"] = lexical_score
                        hit["bm25_score"] = score
                        hit["bm25_rank"] = rank
                        hit["best_faiss_row"] = row
                        hit["child"] = child
                if len(parent_hits) >= min_parent_pool and allowed_rows is None:
                    return parent_hits
        return parent_hits

    def compute_allowed_faiss_rows(self, filters: dict[str, Any]) -> set[int] | None:
        if not filters:
            return None

        parent_ids = self.compute_allowed_parent_ids(filters)
        if parent_ids is not None and not parent_ids:
            return set()

        where_sql, params = build_child_filter_where(filters)
        sql = f"SELECT faiss_row, parent_id FROM child_metadata{where_sql}"
        rows = self.child_db().execute(sql, params).fetchall()
        allowed: set[int] = set()
        for row in rows:
            if parent_ids is not None and row["parent_id"] not in parent_ids:
                continue
            allowed.add(int(row["faiss_row"]))
        return allowed

    def compute_allowed_parent_ids(self, filters: dict[str, Any]) -> set[str] | None:
        where_sql, params = build_parent_filter_where(filters)
        if not where_sql:
            return None
        rows = self.parent_db().execute(f"SELECT parent_id FROM parents{where_sql}", params).fetchall()
        return {row["parent_id"] for row in rows}

    def build_candidates(
        self,
        parent_hits: dict[str, dict[str, Any]],
        parents: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        missing = 0
        for pid, hit in parent_hits.items():
            parent = parents.get(pid)
            if parent is None:
                missing += 1
                parent = {
                    "parent_id": pid,
                    "book_id": hit["child"].get("book_id"),
                    "category": hit["child"].get("category"),
                    "citation": "",
                    "section_path": hit["child"].get("section_path", []),
                    "refs": hit["child"].get("refs", {}),
                    "text_for_synthesis": hit["child"].get("display_text") or "",
                    "text_original": "",
                    "metadata": {},
                }
            candidates.append(build_result_from_parent(parent, hit))

        if missing:
            log(f"  [Parents] {missing} parent_ids missing; using child display_text fallback")
        return candidates

    # ── reranking ─────────────────────────────────────────────────────────────
    def choose_boost_categories(self, question: str, override: str | None) -> list[str] | None:
        if override == "all":
            log("  [Boost] No boost: override all")
            return None
        if override:
            log(f"  [Boost] Manual category prior: {override}")
            return [override]

        keyword_hits = detect_source_boost_keywords(question)
        if keyword_hits:
            log(f"  [Boost] Keyword category prior: {', '.join(keyword_hits)}")
            return keyword_hits

        llm_hits = self.detect_source_boost_llm(question)
        if llm_hits:
            log(f"  [Boost] LLM category prior: {', '.join(llm_hits)}")
            return llm_hits
        log("  [Boost] No category prior")
        return None

    def detect_source_boost_llm(self, question: str) -> list[str] | None:
        found, status = self.classify_source_type(question)
        if status != "off":
            self._status["source_classifier"] = status
        return found

    def classify_source_type(self, question: str) -> tuple[list[str] | None, str]:
        """The source types the LLM says to prioritise for ``question`` (None for
        all), and ``ok``, ``off`` (no API key) or ``failed(<reason>)``."""
        if not self.api_key:
            return None, "off"
        prompt = f"""Classify which Islamic source type should be prioritized for this search query.

Allowed source types:
- hadith
- quran
- tafsir
- fiqh
- fiqh_usul
- aqeedah
- seerah

Return only comma-separated source type names, or all.

Query: {question}

Sources:"""
        try:
            resp = request_with_retries(
                "post",
                f"{self.openrouter_url}/chat/completions",
                headers=self._auth_headers(),
                json_body={
                    "model": self.listwise_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0,
                },
                timeout=20,
                retries=1,
            )
            message = resp.json()["choices"][0]["message"]
            answer = (message.get("content") or "").strip().lower()
            if not answer:
                return None, "failed(empty answer)"
            if "all" in answer:
                return None, "ok"
            return [cat for cat in CATEGORY_ORDER if cat in answer] or None, "ok"
        except Exception as exc:
            log(f"  [Boost] LLM classification skipped: {exc}")
            return None, f"failed({str(exc)[:200]})"

    def apply_mmr(
        self,
        index,
        query_vec: np.ndarray,
        candidates: list[dict[str, Any]],
        lam: float,
    ) -> list[dict[str, Any]]:
        if not hasattr(index, "reconstruct"):
            log("  [MMR] FAISS index cannot reconstruct vectors; skipped")
            return candidates

        try:
            doc_vecs: dict[int, np.ndarray] = {}
            rel_scores: dict[int, float] = {}
            by_row: dict[int, dict[str, Any]] = {}
            for c in candidates:
                row = c.get("faiss_row")
                if row is None:
                    continue
                row_i = int(row)
                vec = np.asarray(index.reconstruct(row_i), dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm <= 1e-8:
                    continue
                doc_vecs[row_i] = vec / norm
                rel_scores[row_i] = float(c.get("weighted_score", c.get("semantic_sim", 0.0)))
                by_row[row_i] = c

            if len(rel_scores) > 1:
                min_rel = min(rel_scores.values())
                max_rel = max(rel_scores.values())
                if max_rel > min_rel:
                    rel_scores = {
                        row: (score - min_rel) / (max_rel - min_rel)
                        for row, score in rel_scores.items()
                    }

            selected = mmr_select(
                doc_vecs=doc_vecs,
                rel_scores=rel_scores,
                candidate_rows=list(by_row),
                lam=lam,
                k=len(by_row),
            )
            if not selected:
                return candidates

            reranked: list[dict[str, Any]] = []
            for rank, (row, score) in enumerate(selected):
                rec = by_row[row].copy()
                rec["mmr_rank"] = rank
                rec["mmr_relevance_score"] = round(float(score), 6)
                reranked.append(rec)

            selected_rows = {row for row, _ in selected}
            reranked.extend(c for c in candidates if int(c.get("faiss_row", -1)) not in selected_rows)
            return reranked
        except Exception as exc:
            log(f"  [MMR] Skipped: {exc}")
            return candidates

    def llm_listwise_rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        depth: int = LISTWISE_DEPTH,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            self._status["rerank"] = "failed(no OPENROUTER_API_KEY)"
            return candidates[:top_k]

        to_rerank = candidates[:depth]
        lines = []
        for i, c in enumerate(to_rerank, 1):
            anchors = planning.candidate_anchor_texts(c)
            text = " | ".join(anchors) or (
                c.get("text_for_synthesis")
                or c.get("text_original")
                or ""
            )
            text = text[:1200]
            text = re.sub(r"\s+", " ", text).strip()
            citation = c.get("citation") or c.get("book_label") or c.get("parent_id") or ""
            lines.append(f"{i}. ({citation}) {text}")

        prompt = f"""أنت خبير في العلوم الإسلامية. رتب النصوص التالية حسب صلتها بالسؤال.

السؤال: {question}

النصوص:
{chr(10).join(lines)}

اكتب أرقام النصوص فقط من الأكثر صلة إلى الأقل، مفصولة بفواصل.
الترتيب:"""
        try:
            resp = request_with_retries(
                "post",
                f"{self.openrouter_url}/chat/completions",
                headers=self._auth_headers(),
                json_body={
                    "model": self.listwise_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0,
                },
                timeout=45,
                retries=1,
            )
            text = resp.json()["choices"][0]["message"].get("content", "")
            ranking = parse_listwise_ranking(text, len(to_rerank))
            reranked = [to_rerank[idx].copy() for idx in ranking]
            for i, rec in enumerate(reranked):
                rec["listwise_rank"] = i
            log(f"  [Listwise] Reranked {len(reranked)} candidates")
            # The reranked head first, then the rest in their previous order.
            return (reranked + candidates[depth:])[:top_k]
        except Exception as exc:
            log(f"  [Listwise] Failed: {exc}")
            self._status["rerank"] = f"failed({str(exc)[:200]})"
            return candidates[:top_k]

    # ── browsing ──────────────────────────────────────────────────────────────
    def list_sources(self) -> str:
        rows = self.child_db().execute(
            """SELECT category, book_id, COUNT(*) as cnt
               FROM child_metadata
               GROUP BY category, book_id
               ORDER BY category, cnt DESC"""
        ).fetchall()
        parent_counts = {
            (row["category"], row["book_id"]): row["cnt"]
            for row in self.parent_db().execute(
                "SELECT category, book_id, COUNT(*) as cnt FROM parents GROUP BY category, book_id"
            ).fetchall()
        }

        lines = [f"{'Category':<14} {'Book ID':<60} {'Children':>8} {'Parents':>8} Display"]
        lines.append("-" * 120)
        cur_cat = None
        cat_child_total = 0
        cat_parent_total = 0
        for row in rows:
            cat = row["category"] or ""
            book_id = row["book_id"] or ""
            cnt = int(row["cnt"])
            if cur_cat is not None and cat != cur_cat:
                lines.append(f"{'':<75} {cat_child_total:>8} {cat_parent_total:>8} [TOTAL {cur_cat}]")
                cat_child_total = 0
                cat_parent_total = 0
            cur_cat = cat
            pcount = int(parent_counts.get((cat, book_id), 0))
            cat_child_total += cnt
            cat_parent_total += pcount
            lines.append(f"{cat:<14} {book_id:<60} {cnt:>8} {pcount:>8} {book_label(book_id)}")
        if cur_cat is not None:
            lines.append(f"{'':<75} {cat_child_total:>8} {cat_parent_total:>8} [TOTAL {cur_cat}]")
        return "\n".join(lines)

    def attach_neighbours(self, results: list[dict[str, Any]], window: int) -> None:
        """Attach the parents immediately before/after each hit, in reading order.

        An author's argument often runs across several parents: the topic is named
        in the first, the ruling lands in the third. Retrieval scores the third
        poorly because the words a user searches for are not in it. Handing the
        answering model the neighbours lets it walk the argument instead of seeing
        a fragment. Ordering comes from ``parents.order_in_book``, populated by
        ``dalil.s7_index.navigation``.
        """
        if window <= 0 or not results:
            return
        conn = self.parent_db()
        try:
            conn.execute("SELECT order_in_book FROM parents LIMIT 1")
        except sqlite3.OperationalError:
            log("  [neighbours] parents.order_in_book missing; run python -m dalil.s7_index.navigation")
            return
        for res in results:
            book, order = res.get("book_id"), res.get("order_in_book")
            if not book or order is None:
                continue
            rows = conn.execute(
                """SELECT order_in_book, parent_id, citation, text_for_synthesis
                   FROM parents
                   WHERE book_id = ? AND order_in_book BETWEEN ? AND ?
                         AND order_in_book != ?
                   ORDER BY order_in_book""",
                (book, order - window, order + window, order),
            ).fetchall()
            res["neighbours"] = [
                {"offset": r["order_in_book"] - order,
                 "parent_id": r["parent_id"],
                 "citation": r["citation"],
                 "text": r["text_for_synthesis"] or ""}
                for r in rows
            ]


class EmbeddingError(RuntimeError):
    """The query could not be embedded (no key, or the embedding API failed)."""


class _QueryPrefetch:
    """Per-``query()`` network and dense-search state shared across variants."""

    def __init__(self, pool: ThreadPoolExecutor):
        self.pool = pool
        self.embeddings: dict[str, Future] = {}
        self.dense: dict[tuple[str, int], Any] = {}
        self.pending: list[str] = []

    def embedding(self, engine: QueryEngine, text: str) -> Future:
        if text not in self.embeddings:
            self.embeddings[text] = self.pool.submit(engine.embed_query, text)
        return self.embeddings[text]


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    # These stores use WAL mode and are extended by dalil.s7_index.append. An
    # immutable connection would ignore committed rows that have not yet been
    # checkpointed from the WAL into the main database file.
    uri = f"file:{path.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=60, check_same_thread=False)


def require_table(conn: sqlite3.Connection, table: str, path: Path) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing table {table!r} in {path}")


def json_loads_or(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def chunks(values: list, size: int) -> list[list]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def candidate_score(candidate: dict[str, Any]) -> float:
    """First-stage score: the fused score when there is one, else dense similarity."""
    for key in ("hybrid_score", "semantic_sim"):
        if candidate.get(key) is not None:
            return float(candidate[key])
    return 0.0


def build_result_from_parent(parent: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    child = hit["child"]
    category = parent.get("category") or child.get("category") or ""
    tier, tier_label, tier_tag = CATEGORY_TIERS.get(category, DEFAULT_TIER)
    score = float(hit["dense_score"]) if hit.get("dense_score") is not None else None
    hybrid_score = hit.get("hybrid_score")
    return {
        # Heading-only parents in the merged corpus have empty parent text.
        # Fall back to the matched child so retrieval never emits blank evidence.
        "text_for_synthesis": (
            parent.get("text_for_synthesis")
            or parent.get("text_original")
            or child.get("display_text")
            or ""
        ),
        "text_original": parent.get("text_original") or child.get("display_text") or "",
        "matched_child_text": child.get("display_text") or "",
        "matched_embedding_text": child.get("embedding_text") or "",
        "citation": render_citation(parent),
        "category": category,
        "book_id": parent.get("book_id") or child.get("book_id") or "",
        "book_label": book_label(parent.get("book_id") or child.get("book_id")),
        "parent_id": parent.get("parent_id") or child.get("parent_id") or "",
        "order_in_book": parent.get("order_in_book"),
        "page_number": parent.get("page_number"),
        "view_id": child.get("view_id"),
        "faiss_row": int(hit["best_faiss_row"]),
        "child_rows": [int(x) for x in hit.get("child_rows", [])],
        "view_type": child.get("view_type"),
        "text_role": child.get("text_role"),
        "section_path": parent.get("section_path") or child.get("section_path") or [],
        "section_path_ids": child.get("section_path_ids") or [],
        "refs": parent.get("refs") or child.get("refs") or {},
        "metadata": parent.get("metadata") or child.get("metadata") or {},
        "semantic_sim": round(score, 6) if score is not None else None,
        "hybrid_score": round(float(hybrid_score), 6) if hybrid_score is not None else None,
        "bm25_score": round(float(hit["bm25_score"]), 6) if hit.get("bm25_score") is not None else None,
        "dense_rank": hit.get("dense_rank"),
        "bm25_rank": hit.get("bm25_rank"),
        "retrieval_sources": hit.get("retrieval_sources", ["dense"]),
        "weighted_score": round(score if score is not None else float(hybrid_score or 0.0), 6),
        "child_scores": [round(float(s), 6) for s in hit.get("child_scores", [])],
        "n_child_hits": len(hit.get("child_scores", []) or [score or 0.0]),
        "credibility_tier": tier,
        "tier_label": tier_label,
        "tier_tag": tier_tag,
        "mmr_rank": None,
        "listwise_rank": None,
    }
