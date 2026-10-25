"""Append a new structured corpus into the existing search stores.

This is an APPEND, not a rebuild. The existing index keeps its vectors and
`faiss_row` numbering; new vectors are added after the current end, and new
rows are written into the same tables with the same schemas, so there is one
set of stores rather than an old set and a new set.

The two collections use disjoint id namespaces (`hadith_api::…`, `openiti:…`,
`saaid:…` vs `book:…`), which is verified before anything is written.

Stores touched, all in place after a backup:
  dense_faiss_ar.index    IndexFlatIP, 4096-d  <- new vectors appended
  faiss_metadata.sqlite   child_metadata       <- one row per new vector
  parents.sqlite          parents              <- new parent records
  bm25.sqlite             child_fts            <- lexical rows + child_count

Vector order is authoritative: the shard `.ids.json` files, concatenated in
shard order, are exactly the views that were embedded, in the order they were
read from retrieval_views.jsonl. The loader replays that same filtered stream
and asserts the ids line up, so a vector can never be paired with the wrong
metadata.

Needs ``faiss`` and, for the token counts, ``transformers``; both are imported
only when a run starts, so ``--help`` works without them.

    python -m dalil.s7_index.append --db data/db --dry-run \\
        --corpus data/build/corpora/collection \\
        --embeddings data/build/embeddings/collection
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dalil.s6_embed.embed_views import too_little_arabic
from dalil.s7_index.bm25 import insert_rows, view_row
from dalil.s7_index.bm25 import staleness as bm25_staleness

METADATA_KEYS = [
    "author_id", "book_id", "category", "display_text", "embedding_profile",
    "file_format", "parent_id", "parser_profile", "quality_flags", "refs",
    "section_path", "section_path_ids", "source_corpus", "subcategory",
    "text_role", "token_counts", "view_id", "view_type", "work_id",
]


def iter_embedded_views(views_path: Path, min_arabic: int, skip_types: set[str]):
    """Replay the exact filter dalil.s6_embed.embed_views used, yielding (line_no, record)."""
    with views_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("view_type") in skip_types:
                continue
            if not (r.get("embedding_text") or "").strip():
                continue
            if too_little_arabic(r, min_arabic):
                continue
            yield line_no, r


def shard_files(emb_dir: Path) -> list[tuple[Path, Path]]:
    npys = sorted(emb_dir.glob("shard_*.npy"))
    out = []
    for npy in npys:
        ids = npy.parent / (npy.stem + ".ids.json")
        if not ids.exists():
            raise SystemExit(f"missing ids file for {npy}")
        out.append((npy, ids))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="structured_corpus dir (retrieval_views.jsonl, parents.jsonl)")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True, help="dir holding the sqlite/faiss stores")
    ap.add_argument("--min-arabic", type=int, default=3)
    ap.add_argument("--skip-view-types", default="")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="processes normalising the BM25 text")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify alignment and report counts without writing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    import faiss  # imported late so --help works without it

    views_path = args.corpus / "retrieval_views.jsonl"
    parents_path = args.corpus / "parents.jsonl"
    index_path = args.db / "dense_faiss_ar.index"
    meta_path = args.db / "faiss_metadata.sqlite"
    parents_db = args.db / "parents.sqlite"
    bm25_db = args.db / "bm25.sqlite"
    for p in (views_path, parents_path, index_path, meta_path, parents_db, bm25_db):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    skip_types = {t for t in args.skip_view_types.split(",") if t}
    shards = shard_files(args.embeddings)
    if not shards:
        raise SystemExit(f"no shards in {args.embeddings}")

    # ---- ids in vector order -------------------------------------------------
    ordered_ids: list[str] = []
    for _, idsf in shards:
        ordered_ids.extend(json.loads(idsf.read_text()))
    print(f"shards: {len(shards)}  vectors: {len(ordered_ids):,}", flush=True)

    # ---- namespace collision check ------------------------------------------
    con_meta = sqlite3.connect(meta_path)
    existing_views = {r[0] for r in con_meta.execute("select view_id from child_metadata")}
    base_row = con_meta.execute("select coalesce(max(faiss_row), -1) from child_metadata").fetchone()[0] + 1
    clash = existing_views.intersection(ordered_ids)
    if clash:
        raise SystemExit(f"view_id collision with existing index ({len(clash)}), refusing: "
                         f"{sorted(clash)[:3]}")
    print(f"existing rows: {len(existing_views):,}  new rows start at faiss_row {base_row:,}"
          f"  collisions: 0", flush=True)

    index = faiss.read_index(str(index_path))
    if index.d != 4096:
        raise SystemExit(f"unexpected index dim {index.d}")
    if index.ntotal != base_row:
        raise SystemExit(f"index ntotal {index.ntotal} != metadata rows {base_row}; stores are "
                         "out of sync, resolve before appending")
    print(f"faiss: {type(index).__name__} d={index.d} ntotal={index.ntotal:,}", flush=True)
    # New FTS rows are keyed faiss_row + 1, so the lexical index must already match
    # the metadata store exactly; otherwise fail before anything is written.
    with sqlite3.connect(f"file:{bm25_db}?mode=ro", uri=True) as con_check:
        problem = bm25_staleness(con_check, base_row)
    if problem:
        raise SystemExit(f"bm25.sqlite is out of sync ({problem}); rebuild it with "
                         f"python -m dalil.s7_index.bm25 --db {args.db} before appending")

    # ---- align metadata stream with vector order -----------------------------
    stream = iter_embedded_views(views_path, args.min_arabic, skip_types)
    records: list[tuple[int, dict]] = []
    for expected in ordered_ids:
        try:
            line_no, rec = next(stream)
        except StopIteration:
            raise SystemExit("retrieval_views.jsonl exhausted before shard ids did; "
                             "corpus and embeddings are from different runs")
        if rec["view_id"] != expected:
            raise SystemExit(f"order mismatch at faiss_row {base_row+len(records)}: "
                             f"shard says {expected}, corpus says {rec['view_id']}")
        records.append((line_no, rec))
    print(f"aligned {len(records):,} views with vectors (ids verified in order)", flush=True)

    if args.dry_run:
        print("dry-run: no writes performed")
        return 0

    if not args.no_backup:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for p in (index_path, meta_path, parents_db, bm25_db):
            bak = p.with_name(f"{p.name}.bak_{stamp}")
            print(f"backup {p.name} -> {bak.name}", flush=True)
            shutil.copy2(p, bak)

    # ---- qwen token counts ---------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B",
                                        revision="4e423935c619ae4df87b646a3ce949610c66241c")
    qwen_tokens: list[int] = []
    for i in range(0, len(records), 2000):
        chunk = [r["embedding_text"] for _, r in records[i:i + 2000]]
        qwen_tokens.extend(len(x) for x in tok(chunk, add_special_tokens=False)["input_ids"])
    print(f"counted qwen tokens for {len(qwen_tokens):,} views", flush=True)

    # ---- vectors -------------------------------------------------------------
    added = 0
    for npy, _ in shards:
        vecs = np.load(npy).astype(np.float32)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        added += vecs.shape[0]
        print(f"  added {npy.name}: {vecs.shape[0]:,} (index ntotal {index.ntotal:,})", flush=True)
    if index.ntotal != base_row + len(ordered_ids):
        raise SystemExit(f"vector count mismatch after add: {index.ntotal}")
    faiss.write_index(index, str(index_path))
    print(f"wrote {index_path.name}: ntotal={index.ntotal:,}", flush=True)

    # ---- child_metadata ------------------------------------------------------
    rows = []
    for offset, (line_no, r) in enumerate(records):
        meta = {k: r.get(k) for k in METADATA_KEYS}
        rows.append((
            base_row + offset, r["view_id"], r["parent_id"], "local",
            r.get("book_id"), r.get("category"), r.get("subcategory"),
            r.get("view_type"), r.get("text_role"), qwen_tokens[offset], line_no,
            json.dumps(r.get("section_path"), ensure_ascii=False),
            json.dumps(r.get("section_path_ids"), ensure_ascii=False),
            json.dumps(r.get("refs"), ensure_ascii=False),
            json.dumps(r.get("token_counts"), ensure_ascii=False),
            r.get("display_text"), r.get("embedding_text"),
            json.dumps(meta, ensure_ascii=False, sort_keys=True),
        ))
    con_meta.executemany(
        "insert into child_metadata (faiss_row, view_id, parent_id, source, book_id, category,"
        " subcategory, view_type, text_role, qwen_tokens, source_line, section_path_json,"
        " section_path_ids_json, refs_json, token_counts_json, display_text, embedding_text,"
        " metadata_json) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con_meta.commit()
    print(f"child_metadata: +{len(rows):,} rows "
          f"(total {con_meta.execute('select count(*) from child_metadata').fetchone()[0]:,})",
          flush=True)

    # ---- parents -------------------------------------------------------------
    con_par = sqlite3.connect(parents_db)
    # Name the columns. A bare `insert into parents values (...)` binds by position,
    # so it breaks the moment the table grows a column — and it did: navigation
    # added order_in_book/page_number/page_end/part_name/page_index, after which a
    # 13-value positional insert fails against an 18-column table. Naming them lets
    # the nav columns stay NULL until dalil.s7_index.navigation backfills them.
    PARENT_INSERT = (
        "insert into parents (parent_id, book_id, category, subcategory, parent_type, "
        "citation, section_path_json, section_path_ids_json, refs_json, child_view_ids_json, "
        "text_for_synthesis, text_original, metadata_json) "
        "values (?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    have = {r[0] for r in con_par.execute("select parent_id from parents")}
    wanted = {r["parent_id"] for _, r in records}
    pbatch, inserted = [], 0
    with parents_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pid = p["parent_id"]
            if pid in have or pid not in wanted:
                continue
            have.add(pid)
            pbatch.append((
                pid, p.get("book_id"), p.get("category"), p.get("subcategory"),
                p.get("parent_type"), p.get("citation"),
                json.dumps(p.get("section_path"), ensure_ascii=False),
                json.dumps(p.get("section_path_ids"), ensure_ascii=False),
                json.dumps(p.get("refs"), ensure_ascii=False),
                json.dumps(p.get("child_view_ids"), ensure_ascii=False),
                p.get("text_for_synthesis") or "", p.get("text_original"),
                json.dumps(p, ensure_ascii=False, sort_keys=True),
            ))
            if len(pbatch) >= args.batch:
                con_par.executemany(PARENT_INSERT, pbatch)
                inserted += len(pbatch); pbatch = []
    if pbatch:
        con_par.executemany(PARENT_INSERT, pbatch)
        inserted += len(pbatch)
    con_par.commit()
    print(f"parents: +{inserted:,} rows "
          f"(total {con_par.execute('select count(*) from parents').fetchone()[0]:,})", flush=True)

    # ---- bm25 ----------------------------------------------------------------
    # Rows come from dalil.s7_index.bm25 so an appended row is identical to the one a
    # rebuild would write from child_metadata, rowid = faiss_row + 1 included.
    con_fts = sqlite3.connect(bm25_db)
    with Pool(args.workers) as pool:
        insert_rows(con_fts, pool.imap(view_row, ((base_row + o, r) for o, (_, r) in enumerate(records)),
                                       chunksize=500), len(records), args.batch)
    total_fts = con_fts.execute("select count(*) from child_fts").fetchone()[0]
    # Only child_count changes. bm25.write_meta would also restamp schema_version,
    # normalization and built_at, claiming a rebuild that did not happen.
    con_fts.execute("update bm25_meta set value=? where key='child_count'", (str(total_fts),))
    con_fts.commit()
    print(f"bm25 child_fts: +{len(records):,} rows (total {total_fts:,})", flush=True)

    print("\nAPPEND COMPLETE")
    print(f"  vectors      {index.ntotal:,}")
    print(f"  child_meta   {con_meta.execute('select count(*) from child_metadata').fetchone()[0]:,}")
    print(f"  parents      {con_par.execute('select count(*) from parents').fetchone()[0]:,}")
    print(f"  bm25         {total_fts:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
