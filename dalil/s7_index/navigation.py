"""Add prev/next navigation to the parents store.

Retrieval finds a passage; reading it in context means pulling what sits either
side of it. That needs a per-book sequence number, which the store did not
originally carry. This adds one and backfills it.

Reading order is recovered differently per collection, but both are exact:

  existing 78 books  ordered by each parent's smallest `source_line` in
                     child_metadata, which is the line number the view had in
                     the corpus file it was built from. Verified against
                     Sahih al-Bukhari, where it reproduces hadith 1,2,3,… in
                     order.

  new 545 books      ordered by position in parents.jsonl, which the parser
                     emits in reading order per book. Ketabonline additionally
                     carries real page numbers, joined in from elements.jsonl.

Columns added (all nullable, so existing rows are untouched where unknown):
  order_in_book  INTEGER   1-based sequence within the book  (every book)
  page_number    INTEGER   printed page                      (ketabonline)
  page_end       INTEGER   last page when a parent spans some (ketabonline)
  part_name      TEXT      volume / juz                      (ketabonline)
  page_index     INTEGER   global monotonic page sequence     (ketabonline)

A fresh store from ``dalil.s7_index.schema`` already has these columns.
``--dry-run`` opens both stores read-only and writes nothing.

    python -m dalil.s7_index.navigation --db data/db \\
        --corpus data/build/corpora/collection --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

NAV_COLUMNS = [
    ("order_in_book", "INTEGER"),
    ("page_number", "INTEGER"),
    ("page_end", "INTEGER"),
    ("part_name", "TEXT"),
    ("page_index", "INTEGER"),
]


def connect(path: Path, dry: bool) -> sqlite3.Connection:
    """Read-only in a dry run, so a dry run cannot change the store."""
    if dry:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return sqlite3.connect(path)


def ensure_columns(con: sqlite3.Connection) -> list[str]:
    have = {r[1] for r in con.execute("PRAGMA table_info(parents)")}
    added = []
    for name, typ in NAV_COLUMNS:
        if name not in have:
            con.execute(f"ALTER TABLE parents ADD COLUMN {name} {typ}")
            added.append(name)
    con.execute("CREATE INDEX IF NOT EXISTS idx_parents_nav "
                "ON parents(book_id, order_in_book)")
    con.commit()
    return added


def backfill_existing(parents_db: Path, meta_db: Path, dry: bool) -> int:
    """Number the pre-existing parents using child_metadata.source_line."""
    con = connect(parents_db, dry)
    if not dry:
        ensure_columns(con)
    con.execute("ATTACH DATABASE ? AS meta",
                (f"file:{meta_db}?mode=ro" if dry else str(meta_db),))
    # smallest source_line per parent = where the parent starts in its corpus file
    rows = con.execute("""
        SELECT book_id, parent_id, MIN(source_line) AS sl, MIN(faiss_row) AS fr
        FROM meta.child_metadata
        GROUP BY book_id, parent_id
        ORDER BY book_id, sl, fr
    """).fetchall()
    updates, current, seq = [], None, 0
    for book_id, parent_id, _sl, _fr in rows:
        if book_id != current:
            current, seq = book_id, 0
        seq += 1
        updates.append((seq, parent_id))
    print(f"  existing: {len(updates):,} parents across "
          f"{len({r[0] for r in rows})} books", flush=True)
    if not dry:
        con.executemany("UPDATE parents SET order_in_book=? WHERE parent_id=?", updates)
        con.commit()
    con.execute("DETACH DATABASE meta")
    con.close()
    return len(updates)


def element_pages(elements_path: Path, wanted: set[str]) -> dict[str, tuple]:
    """element_id -> (page_number, part_name, page_index) for the ids we need."""
    out: dict[str, tuple] = {}
    with elements_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            eid = r.get("element_id")
            if eid not in wanted:
                continue
            sr = (r.get("refs") or {}).get("source_record") or {}
            pn = sr.get("page_number")
            if pn is None:
                continue
            out[eid] = (pn, sr.get("part_name"), sr.get("page_index"))
    return out


def backfill_new(parents_db: Path, corpus: Path, dry: bool) -> int:
    """Number the new parents by their order in parents.jsonl, plus page info."""
    con = connect(parents_db, dry)
    if not dry:
        ensure_columns(con)
    present = {r[0] for r in con.execute("SELECT parent_id FROM parents")}

    order_rows, need = [], set()
    seq_by_book: dict[str, int] = {}
    parent_elems: dict[str, list[str]] = {}
    with (corpus / "parents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pid = p["parent_id"]
            if pid not in present:
                continue
            book = p.get("book_id")
            seq_by_book[book] = seq_by_book.get(book, 0) + 1
            order_rows.append((seq_by_book[book], pid))
            els = p.get("source_element_ids") or []
            if els:
                parent_elems[pid] = els
                need.update(els)
    print(f"  new: {len(order_rows):,} parents across {len(seq_by_book)} books", flush=True)
    if not order_rows:
        con.close()
        return 0

    print(f"  resolving page info for {len(need):,} source elements "
          f"(streaming elements.jsonl)…", flush=True)
    pages = element_pages(corpus / "elements.jsonl", need)
    print(f"  page info found for {len(pages):,} elements", flush=True)

    page_rows = []
    for pid, els in parent_elems.items():
        hits = [pages[e] for e in els if e in pages]
        if not hits:
            continue
        nums = [h[0] for h in hits if h[0] is not None]
        idxs = [h[2] for h in hits if h[2] is not None]
        page_rows.append((
            min(nums) if nums else None,
            max(nums) if nums else None,
            hits[0][1],
            min(idxs) if idxs else None,
            pid,
        ))
    print(f"  parents with page numbers: {len(page_rows):,}", flush=True)

    if not dry:
        con.executemany("UPDATE parents SET order_in_book=? WHERE parent_id=?", order_rows)
        con.executemany(
            "UPDATE parents SET page_number=?, page_end=?, part_name=?, page_index=? "
            "WHERE parent_id=?", page_rows)
        con.commit()
    con.close()
    return len(order_rows)


def verify(parents_db: Path) -> None:
    con = sqlite3.connect(parents_db)
    total, numbered = con.execute(
        "SELECT COUNT(*), COUNT(order_in_book) FROM parents").fetchone()
    pages = con.execute("SELECT COUNT(page_number) FROM parents").fetchone()[0]
    print(f"\nparents: {total:,} | with order_in_book: {numbered:,} "
          f"| with page_number: {pages:,}")
    bad = con.execute("""
        SELECT book_id, COUNT(*) c, COUNT(DISTINCT order_in_book) d
        FROM parents WHERE order_in_book IS NOT NULL
        GROUP BY book_id HAVING c <> d LIMIT 5
    """).fetchall()
    print(f"books with duplicate order_in_book: {len(bad)}"
          + (f"  {bad}" if bad else "  (none)"))
    gaps = con.execute("""
        SELECT book_id, MIN(order_in_book), MAX(order_in_book), COUNT(*)
        FROM parents WHERE order_in_book IS NOT NULL GROUP BY book_id
        HAVING MAX(order_in_book) <> COUNT(*) LIMIT 5
    """).fetchall()
    print(f"books with gaps in sequence: {len(gaps)}" + (f"  {gaps}" if gaps else "  (none)"))
    con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True,
                    help="directory holding parents.sqlite and faiss_metadata.sqlite")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="structured_corpus dir; omit to only backfill existing books")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    parents_db = args.db / "parents.sqlite"
    meta_db = args.db / "faiss_metadata.sqlite"
    for p in (parents_db, meta_db):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    if not args.dry_run and not args.no_backup:
        bak = parents_db.with_name(f"parents.sqlite.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        print(f"backup -> {bak.name}", flush=True)
        shutil.copy2(parents_db, bak)

    print("backfilling existing books from child_metadata.source_line…", flush=True)
    backfill_existing(parents_db, meta_db, args.dry_run)
    if args.corpus:
        print("numbering new books from parents.jsonl order…", flush=True)
        backfill_new(parents_db, args.corpus, args.dry_run)
    if not args.dry_run:
        verify(parents_db)
    else:
        print("\ndry-run: no writes performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
