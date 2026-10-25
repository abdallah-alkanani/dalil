"""The FTS5 lexical index (``bm25.sqlite``): its schema, its rows, and a rebuild.

One module owns the index so the three code paths that touch it (this rebuild,
``dalil.s7_index.append`` and the search engine's freshness check) cannot drift
apart. They did once: the query engine carried its own builder that skipped
source-marker cleaning, so a silent rebuild from the query path would have
produced a different index from this one.

Rows are keyed ``rowid = faiss_row + 1`` (FTS5 rowids are 1-based, FAISS rows
0-based).

Rebuild from stored evidence (``display_text`` / ``embedding_text`` in
``faiss_metadata.sqlite``) rather than patching: normalisation is lossy (once
صلى became صلي you cannot tell which ي was ى), so regenerating from the original
spelling is the only correct route when the rules change. ``display_text`` itself
is never modified; cleaning applies only to the derived search fields.

    python -m dalil.s7_index.bm25 --db data/db
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from dalil.arabic.normalize import normalize_for_search
from dalil.paths import DB_DIR

SCHEMA_VERSION = "4"
NORMALIZATION = (
    "arabic_norm.v1: strip tashkeel; fold alif seats + persian codepoints + digits; "
    "KEEP ya-maqsura, hamza-on-waw, hamza-on-ya, ta-marbuta"
)
COLUMNS = (
    "search_text", "context_text", "parent_id", "category", "subcategory",
    "book_id", "source", "view_type", "text_role", "source_corpus", "work_id",
    "author_id", "file_format", "parser_profile", "quran_surah",
    "quran_ayah_start", "quran_ayah_end", "hadith_collection", "hadith_number",
    "hadith_book_number", "hadith_chapter_id", "source_record_id",
)
FTS_DDL = (
    "CREATE VIRTUAL TABLE child_fts USING fts5("
    + ", ".join(c if c in ("search_text", "context_text") else f"{c} UNINDEXED" for c in COLUMNS)
    + ", tokenize = \"unicode61 categories 'L* N* Co'\")"
)
INSERT_SQL = (
    f"INSERT INTO child_fts (rowid, {', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('?' * (len(COLUMNS) + 1))})"
)


def _get(tree: dict | None, *path: str) -> Any:
    cur: Any = tree or {}
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
        if cur is None:
            return ""
    return cur


def fts_row(faiss_row: int, *, display_text: str | None, embedding_text: str | None,
            parent_id: str, category: str | None, subcategory: str | None, book_id: str | None,
            source: str | None, view_type: str | None, text_role: str | None,
            refs: dict | None, meta: dict | None) -> tuple:
    """One child_fts row, in INSERT_SQL order."""
    meta = meta or {}
    return (
        faiss_row + 1,
        normalize_for_search(display_text or ""),
        normalize_for_search(embedding_text or ""),
        parent_id, category or "", subcategory or "", book_id or "", source or "",
        view_type or "", text_role or "",
        meta.get("source_corpus") or "", meta.get("work_id") or "",
        meta.get("author_id") or "", meta.get("file_format") or "",
        meta.get("parser_profile") or "",
        _get(refs, "quran", "surah"), _get(refs, "quran", "ayah_start"), _get(refs, "quran", "ayah_end"),
        _get(refs, "hadith", "collection"), _get(refs, "hadith", "hadith_number"),
        _get(refs, "hadith", "book_number"), _get(refs, "hadith", "chapter_id"),
        _get(refs, "source_record", "record_id"),
    )


def view_row(item: tuple[int, dict]) -> tuple:
    """The row of one ``(faiss_row, retrieval view)``, as ``dalil.s7_index.append`` writes it."""
    faiss_row, r = item
    return fts_row(faiss_row, display_text=r.get("display_text"),
                   embedding_text=r.get("embedding_text"), parent_id=r.get("parent_id"),
                   category=r.get("category"), subcategory=r.get("subcategory"),
                   book_id=r.get("book_id"), source="local", view_type=r.get("view_type"),
                   text_role=r.get("text_role"), refs=r.get("refs"), meta=r)


def metadata_row(row: tuple) -> tuple:
    """The row of one ``child_metadata`` record, as ``rebuild`` selects it."""
    (faiss_row, pid, bid, cat, sub, vt, role, source, dt, et, rj, mj) = row
    return fts_row(faiss_row, display_text=dt, embedding_text=et, parent_id=pid, category=cat,
                   subcategory=sub, book_id=bid, source=source, view_type=vt, text_role=role,
                   refs=json.loads(rj) if rj else {}, meta=json.loads(mj) if mj else {})


def insert_rows(dst: sqlite3.Connection, rows, total: int, batch: int = 20000) -> int:
    """Insert an iterable of rows in batches, printing progress; returns the empty-text count.

    Normalising the text is the slow part, so callers produce ``rows`` with a process
    pool (``Pool.imap``, which keeps order); SQLite takes one writer, this one.
    """
    done, empty, out, t0 = 0, 0, [], time.time()
    for r in rows:
        out.append(r)
        empty += not r[1]
        if len(out) >= batch:
            dst.executemany(INSERT_SQL, out)
            done += len(out)
            out = []
            print(f"  {done:,}/{total:,} ({100 * done / total:.1f}%) "
                  f"{done / max(1e-9, time.time() - t0):,.0f} rows/s", flush=True)
    dst.executemany(INSERT_SQL, out)
    return empty


def write_meta(conn: sqlite3.Connection, child_count: int) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS bm25_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.executemany(
        "INSERT OR REPLACE INTO bm25_meta VALUES (?, ?)",
        [("child_count", str(child_count)), ("schema_version", SCHEMA_VERSION),
         ("normalization", NORMALIZATION), ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))],
    )


def staleness(bm25: sqlite3.Connection, child_count: int) -> str | None:
    """Why the index does not match ``child_count`` metadata rows, or None if it does.

    Uses FTS5's own docsize table, so the check costs milliseconds rather than a
    full-text COUNT(*). A rebuild drops ``bm25_meta`` first and writes it last, so
    an interrupted build is reported as missing metadata.
    """
    try:
        meta = dict(bm25.execute("SELECT key, value FROM bm25_meta").fetchall())
        count, lo, hi = bm25.execute("SELECT count(*), min(id), max(id) FROM child_fts_docsize").fetchone()
    except sqlite3.Error as exc:
        return f"index unreadable or incomplete ({exc})"
    if meta.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}"
    if not int(meta.get("child_count", -1)) == child_count == count:
        return f"{count:,} indexed rows (meta says {meta.get('child_count')}) for {child_count:,} metadata rows"
    if child_count and (lo, hi) != (1, child_count):
        return f"rowids span {lo}..{hi}, expected 1..{child_count}"
    return None


def rebuild(db_dir: Path, *, batch: int = 20000, backup: bool = True,
            workers: int | None = None) -> int:
    meta_db, bm25_db = db_dir / "faiss_metadata.sqlite", db_dir / "bm25.sqlite"
    if not meta_db.exists():
        raise SystemExit(f"missing: {meta_db}")
    # The pool's feeder thread reads the cursor, hence check_same_thread=False.
    src = sqlite3.connect(f"file:{meta_db}?mode=ro", uri=True, check_same_thread=False)
    total = src.execute("SELECT COUNT(*) FROM child_metadata").fetchone()[0]
    print(f"source rows in child_metadata: {total:,}", flush=True)

    if backup and bm25_db.exists():
        bak = bm25_db.with_name(f"bm25.sqlite.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        print(f"backup -> {bak.name}", flush=True)
        shutil.copy2(bm25_db, bak)

    dst = sqlite3.connect(bm25_db)
    dst.execute("PRAGMA journal_mode=OFF")
    dst.execute("PRAGMA synchronous=OFF")
    dst.execute("DROP TABLE IF EXISTS bm25_meta")
    dst.execute("DROP TABLE IF EXISTS child_fts")
    dst.execute(FTS_DDL)

    cur = src.execute(
        "SELECT faiss_row, parent_id, book_id, category, subcategory, view_type, text_role, "
        "source, display_text, embedding_text, refs_json, metadata_json "
        "FROM child_metadata ORDER BY faiss_row")
    t0 = time.time()
    with Pool(workers or os.cpu_count()) as pool:
        empty = insert_rows(dst, pool.imap(metadata_row, cur, chunksize=500), total, batch)
    dst.commit()

    written = dst.execute("SELECT COUNT(*) FROM child_fts").fetchone()[0]
    dst.execute("INSERT INTO child_fts(child_fts) VALUES('optimize')")
    write_meta(dst, written)
    dst.commit()
    print(f"\nrebuilt child_fts: {written:,} rows ({empty:,} had empty search_text) "
          f"in {(time.time() - t0) / 60:.1f} min")
    if written != total:
        print(f"WARNING: wrote {written:,} but source had {total:,}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_DIR, help="directory holding the search stores")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show sample normalised rows; write nothing")
    args = ap.parse_args(argv)

    if args.dry_run:
        src = sqlite3.connect(f"file:{args.db / 'faiss_metadata.sqlite'}?mode=ro", uri=True)
        for (dt,) in src.execute("SELECT display_text FROM child_metadata WHERE length(display_text) > 120 LIMIT 3"):
            print(f"  IN : {dt[:88]}\n  OUT: {normalize_for_search(dt)[:88]}\n")
        print("dry-run: nothing written")
        return 0
    return rebuild(args.db, batch=args.batch, backup=not args.no_backup)


if __name__ == "__main__":
    sys.exit(main())
