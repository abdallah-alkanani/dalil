"""The last step of `run_build.sh index`, after the corpora are appended.

1. author ids: checks that every view and parent carries its book's registry id
   (author_ids.json; dalil.build.select wrote them into the corpora, so append
   already put them in the metadata and the BM25 rows);
2. reading order and pages: dalil.s7_index.navigation once, over all appended corpora;
3. BM25 optimised (its segments merged into one, for faster queries);
4. checks, written to <db>/BUILD_REPORT.json.

    python -m dalil.build.finalize --db STORES --corpora corpora/ --authors author_ids.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from dalil.s7_index import bm25, navigation


def check_authors(db: Path, authors: dict) -> dict:
    out = {}
    for store, table in (("faiss_metadata.sqlite", "child_metadata"), ("parents.sqlite", "parents")):
        con = sqlite3.connect(f"file:{db / store}?mode=ro", uri=True)
        pairs = set(con.execute(f"SELECT DISTINCT book_id, json_extract(metadata_json, '$.author_id') FROM {table}"))
        wrong = sorted(b for b, a in pairs if b not in authors or authors[b] != a)
        if wrong:
            raise SystemExit(f"{table}: books whose author id is not the one in the author map: {wrong[:5]}")
        out[table] = len({a for _, a in pairs if a is not None})
        con.close()
    return out


def combined(corpora: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("parents.jsonl", "elements.jsonl"):
        with open(out / name, "wb") as dst:
            for d in sorted(p for p in corpora.iterdir() if p.is_dir() and not p.name.startswith("_")):
                with open(d / name, "rb") as src:
                    shutil.copyfileobj(src, dst)
    return out


def checks(db: Path) -> dict:
    import faiss

    meta = sqlite3.connect(f"file:{db / 'faiss_metadata.sqlite'}?mode=ro", uri=True)
    par = sqlite3.connect(f"file:{db / 'parents.sqlite'}?mode=ro", uri=True)
    rows, lo, hi = meta.execute("SELECT COUNT(*), MIN(faiss_row), MAX(faiss_row) FROM child_metadata").fetchone()
    index = faiss.read_index(str(db / "dense_faiss_ar.index"), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    view_ids = {v for (v,) in meta.execute("SELECT view_id FROM child_metadata")}
    parent_ids = {p for (p,) in par.execute("SELECT parent_id FROM parents")}
    orphan_views = sum(1 for (p,) in meta.execute("SELECT parent_id FROM child_metadata") if p not in parent_ids)
    listed_not_indexed = 0
    for (ids,) in par.execute("SELECT child_view_ids_json FROM parents"):
        listed_not_indexed += sum(1 for v in json.loads(ids or "[]") if v not in view_ids)
    with sqlite3.connect(f"file:{db / 'bm25.sqlite'}?mode=ro", uri=True) as con:
        stale = bm25.staleness(con, rows)
    report = {
        "views": rows, "faiss_ntotal": index.ntotal, "faiss_rows_contiguous": (lo, hi) == (0, rows - 1),
        "parents": len(parent_ids),
        "books": meta.execute("SELECT COUNT(DISTINCT book_id) FROM child_metadata").fetchone()[0],
        "views_whose_parent_is_missing": orphan_views,
        "parent_view_ids_not_indexed": listed_not_indexed,
        "parents_without_order": par.execute("SELECT COUNT(*) FROM parents WHERE order_in_book IS NULL").fetchone()[0],
        "bm25_problem": stale,
    }
    report["ok"] = (report["views"] == report["faiss_ntotal"] and report["faiss_rows_contiguous"]
                    and orphan_views == 0 and not stale)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--corpora", type=Path, required=True)
    ap.add_argument("--authors", type=Path, required=True)
    args = ap.parse_args(argv)
    report = {"distinct_author_ids": check_authors(args.db, json.loads(args.authors.read_text(encoding="utf-8")))}
    print("authors:", report["distinct_author_ids"], flush=True)
    navigation.main(["--db", str(args.db), "--corpus", str(combined(args.corpora, args.corpora / "_all")), "--no-backup"])
    with sqlite3.connect(args.db / "bm25.sqlite") as con:
        con.execute("INSERT INTO child_fts(child_fts) VALUES('optimize')")
    report.update(checks(args.db))
    (args.db / "BUILD_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not report["ok"]:
        raise SystemExit("checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
