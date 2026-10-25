"""Remove books or individual views from the search stores, writing a new set.

Rows are keyed by ``faiss_row``, which must stay contiguous, so removal means
renumbering: this writes a complete new set of stores to ``--out`` and never
modifies ``--db``. Swap the directories yourself once the new set is checked.

    python -m dalil.s7_index.prune --db data/db --out data/db_new \\
        --drop-books books.txt --drop-views faiss_rows.txt

- child_metadata keeps the surviving rows in their original order, renumbered.
- parents: a parent goes when its book is dropped, or when every one of its views
  was dropped; otherwise it stays, with dropped view ids removed from its list.
- The dense index is streamed row by row from the old file (never loaded whole);
  the result is checked against the old vectors.
- bm25.sqlite is rebuilt from the new metadata with ``dalil.s7_index.bm25``.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

from dalil.s7_index import bm25
from dalil.s7_index.schema import CHILD_METADATA_DDL, PARENTS_DDL
from dalil.paths import DB_DIR

BLOCK_ROWS = 32768


def read_ids(path: Path | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def flat_index_layout(path: Path) -> tuple[bytes, int, int, int]:
    """(header bytes up to ntotal, ntotal, d, offset of the vectors) of an IndexFlat file."""
    import faiss

    index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    if type(index) not in (faiss.IndexFlatIP, faiss.IndexFlat):
        raise SystemExit(f"{path} is {type(index).__name__}; only flat indexes can be pruned")
    d, ntotal = index.d, index.ntotal
    offset = path.stat().st_size - ntotal * d * 4
    with path.open("rb") as f:
        head = f.read(offset)
    # Layout written by faiss: fourcc, d (int32), ntotal (int64), 2 x int64,
    # is_trained (uint8), metric (int32), then the codes: their length in floats
    # (uint64; faiss's WRITEXBVECTOR), then the raw float32 bytes.
    if struct.unpack_from("<i", head, 4)[0] != d or struct.unpack_from("<q", head, 8)[0] != ntotal:
        raise SystemExit(f"unexpected header layout in {path}")
    if struct.unpack_from("<Q", head, offset - 8)[0] != ntotal * d:
        raise SystemExit(f"unexpected codes size in {path}")
    return head, ntotal, d, offset


def write_pruned_index(src: Path, dst: Path, keep_rows: np.ndarray) -> None:
    head, ntotal, d, offset = flat_index_layout(src)
    new_total = len(keep_rows)
    new_head = bytearray(head)
    struct.pack_into("<q", new_head, 8, new_total)
    struct.pack_into("<Q", new_head, offset - 8, new_total * d)
    keep = np.zeros(ntotal, dtype=bool)
    keep[keep_rows] = True
    row_bytes = d * 4
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(new_head)
        for start in range(0, ntotal, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, ntotal - start)
            fin.seek(offset + start * row_bytes)
            block = np.frombuffer(fin.read(rows * row_bytes), dtype=np.float32).reshape(rows, d)
            fout.write(block[keep[start:start + rows]].tobytes())


def check_index(src: Path, dst: Path, keep_rows: np.ndarray, samples: int = 2000) -> None:
    import faiss

    old = faiss.read_index(str(src), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    new = faiss.read_index(str(dst), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    if new.ntotal != len(keep_rows) or new.d != old.d:
        raise SystemExit(f"new index has {new.ntotal} x {new.d}, expected {len(keep_rows)} x {old.d}")
    picks = random.Random(0).sample(range(len(keep_rows)), min(samples, len(keep_rows)))
    for new_row in picks + [0, len(keep_rows) - 1] if len(keep_rows) else []:
        if not np.array_equal(new.reconstruct(new_row), old.reconstruct(int(keep_rows[new_row]))):
            raise SystemExit(f"vector mismatch at new row {new_row}")


def prune(db: Path, out: Path, drop_books: list[str], drop_rows: list[int]) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    meta_new, parents_new = out / "faiss_metadata.sqlite", out / "parents.sqlite"
    for path, ddl in ((meta_new, CHILD_METADATA_DDL), (parents_new, PARENTS_DDL)):
        con = sqlite3.connect(path)
        con.executescript(ddl)
        con.close()

    con = sqlite3.connect(f"file:{meta_new}", uri=True)  # URI mode, so ATTACH can open the old DB read-only
    con.execute("PRAGMA journal_mode=OFF")
    con.execute(f"ATTACH 'file:{db / 'faiss_metadata.sqlite'}?mode=ro' AS old")
    con.execute("CREATE TEMP TABLE drop_books(book_id TEXT PRIMARY KEY)")
    con.execute("CREATE TEMP TABLE drop_rows(faiss_row INTEGER PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO drop_books VALUES (?)", [(b,) for b in drop_books])
    con.executemany("INSERT OR IGNORE INTO drop_rows VALUES (?)", [(r,) for r in drop_rows])
    unknown = [b for (b,) in con.execute(
        "SELECT book_id FROM drop_books WHERE book_id NOT IN (SELECT DISTINCT book_id FROM old.child_metadata)")]
    if unknown:
        raise SystemExit(f"{len(unknown)} book ids to drop are not in the DB, e.g. {unknown[:3]}")
    con.execute("""INSERT INTO drop_rows SELECT faiss_row FROM old.child_metadata
                   WHERE book_id IN (SELECT book_id FROM drop_books)
                   AND faiss_row NOT IN (SELECT faiss_row FROM drop_rows)""")
    columns = [r[1] for r in con.execute("PRAGMA old.table_info(child_metadata)")]
    if columns != [r[1] for r in con.execute("PRAGMA main.table_info(child_metadata)")]:
        raise SystemExit("child_metadata columns differ from dalil.s7_index.schema; update the schema first")
    rest = ", ".join(c for c in columns if c != "faiss_row")
    t0 = time.time()
    con.execute(f"""INSERT INTO child_metadata (faiss_row, {rest})
                    SELECT ROW_NUMBER() OVER (ORDER BY faiss_row) - 1, {rest}
                    FROM old.child_metadata
                    WHERE faiss_row NOT IN (SELECT faiss_row FROM drop_rows)
                    ORDER BY faiss_row""")
    con.commit()
    keep_rows = np.fromiter(
        (r for (r,) in con.execute("""SELECT faiss_row FROM old.child_metadata
                                      WHERE faiss_row NOT IN (SELECT faiss_row FROM drop_rows)
                                      ORDER BY faiss_row""")), dtype=np.int64)
    dropped_views = {v for (v,) in con.execute(
        "SELECT view_id FROM old.child_metadata WHERE faiss_row IN (SELECT faiss_row FROM drop_rows)")}
    # Parents left with no view at all (their views were all dropped).
    emptied = {p for (p,) in con.execute(
        """SELECT parent_id FROM old.child_metadata GROUP BY parent_id
           HAVING SUM(faiss_row IN (SELECT faiss_row FROM drop_rows)) = COUNT(*)""")}
    counts = {"views_before": con.execute("SELECT COUNT(*) FROM old.child_metadata").fetchone()[0],
              "views_after": con.execute("SELECT COUNT(*) FROM child_metadata").fetchone()[0]}
    con.close()
    print(f"child_metadata: {counts['views_before']:,} -> {counts['views_after']:,} ({time.time() - t0:.0f}s)", flush=True)

    con = sqlite3.connect(f"file:{parents_new}", uri=True)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute(f"ATTACH 'file:{db / 'parents.sqlite'}?mode=ro' AS old")
    con.execute("CREATE TEMP TABLE drop_parents(parent_id TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO drop_parents VALUES (?)", [(p,) for p in emptied])
    con.executemany("INSERT OR IGNORE INTO drop_parents SELECT parent_id FROM old.parents WHERE book_id = ?",
                    [(b,) for b in drop_books])
    pcols = [r[1] for r in con.execute("PRAGMA old.table_info(parents)")]
    if pcols != [r[1] for r in con.execute("PRAGMA main.table_info(parents)")]:
        raise SystemExit("parents columns differ from dalil.s7_index.schema; update the schema first")
    col_list = ", ".join(pcols)
    con.execute(f"INSERT INTO parents ({col_list}) SELECT {col_list} FROM old.parents "
                "WHERE parent_id NOT IN (SELECT parent_id FROM drop_parents)")
    touched = 0
    for pid, ids_json in con.execute("SELECT parent_id, child_view_ids_json FROM parents").fetchall():
        ids = json.loads(ids_json) if ids_json else []
        kept = [v for v in ids if v not in dropped_views]
        if len(kept) != len(ids):
            con.execute("UPDATE parents SET child_view_ids_json = ? WHERE parent_id = ?",
                        (json.dumps(kept, ensure_ascii=False), pid))
            touched += 1
    counts["parents_before"] = con.execute("SELECT COUNT(*) FROM old.parents").fetchone()[0]
    counts["parents_after"] = con.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
    counts["parents_view_lists_updated"] = touched
    con.commit()
    con.close()
    print(f"parents: {counts['parents_before']:,} -> {counts['parents_after']:,} "
          f"({touched} view lists updated)", flush=True)

    t0 = time.time()
    write_pruned_index(db / "dense_faiss_ar.index", out / "dense_faiss_ar.index", keep_rows)
    check_index(db / "dense_faiss_ar.index", out / "dense_faiss_ar.index", keep_rows)
    print(f"dense index: {len(keep_rows):,} vectors copied and spot-checked ({time.time() - t0:.0f}s)", flush=True)

    if bm25.rebuild(out, backup=False) != 0:
        raise SystemExit("BM25 rebuild did not match the metadata row count")
    con = sqlite3.connect(f"file:{out / 'bm25.sqlite'}?mode=ro", uri=True)
    problem = bm25.staleness(con, counts["views_after"])
    con.close()
    if problem:
        raise SystemExit(f"new BM25 index is inconsistent: {problem}")
    counts["books_dropped"] = len(drop_books)
    counts["views_dropped"] = counts["views_before"] - counts["views_after"]
    (out / "PRUNE_REPORT.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_DIR, help="existing stores (read only)")
    ap.add_argument("--out", type=Path, required=True, help="new, empty directory for the pruned stores")
    ap.add_argument("--drop-books", type=Path, help="file with one book_id per line")
    ap.add_argument("--drop-views", type=Path, help="file with one faiss_row per line")
    args = ap.parse_args(argv)
    books = read_ids(args.drop_books)
    rows = [int(r) for r in read_ids(args.drop_views)]
    if not books and not rows:
        raise SystemExit("nothing to drop")
    counts = prune(args.db, args.out, books, rows)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
