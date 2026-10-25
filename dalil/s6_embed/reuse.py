"""Reuse existing vectors for views whose embedding text has not changed.

A re-extraction changes the text of only some views (a section renamed, text
recovered), so most of a rebuilt book's views already have a vector in the
search stores. Views are matched on their exact `embedding_text`; only the rest
are embedded.

    # 1. views with no vector yet, one per distinct text, for dalil.s6_embed.embed_views
    python -m dalil.s6_embed.reuse plan --corpus BUILD --db data/db --out BUILD/to_embed.jsonl
    # 2. on the GPU:  python -m dalil.s6_embed.embed_views --views BUILD/to_embed.jsonl --out NEW/
    # 3. shards covering every embeddable view of BUILD, in the order dalil.s7_index.append reads them
    python -m dalil.s6_embed.reuse assemble --corpus BUILD --db data/db --new NEW/ --out BUILD/embeddings

`assemble` takes each vector from the stores (by text) or from NEW (by text, so
one embedding serves every view with that text, in any corpus) and fails if a view
has neither. The views are read with the same filter as embed_views and append.
Reused vectors are copied as float32 from the index, which append renormalises.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from dalil.s6_embed.embed_views import iter_views

SHARD_SIZE = 50_000


def text_key(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def stored_rows(db: Path) -> dict[bytes, int]:
    """faiss_row of every stored view, keyed by its embedding text (first row wins)."""
    con = sqlite3.connect(f"file:{db / 'faiss_metadata.sqlite'}?mode=ro", uri=True)
    rows: dict[bytes, int] = {}
    for row, text in con.execute("SELECT faiss_row, embedding_text FROM child_metadata ORDER BY faiss_row"):
        rows.setdefault(text_key(text or ""), row)
    con.close()
    return rows


def embeddable(corpus: Path, skip_types: set[str], min_arabic: int):
    return iter_views(corpus / "retrieval_views.jsonl", "embedding_text", skip_types, min_arabic)


def plan(corpus: Path, db: Path, out: Path, skip_types: set[str], min_arabic: int) -> dict:
    rows = stored_rows(db)
    counts = {"views": 0, "reused": 0, "to_embed_views": 0, "to_embed_texts": 0}
    first: dict[bytes, str] = {}  # text -> the first embeddable view carrying it
    for vid, text in embeddable(corpus, skip_types, min_arabic):
        counts["views"] += 1
        k = text_key(text)
        if k in rows:
            counts["reused"] += 1
        else:
            counts["to_embed_views"] += 1
            first.setdefault(k, vid)
    pick = set(first.values())
    with (corpus / "retrieval_views.jsonl").open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.strip() and json.loads(line)["view_id"] in pick:
                dst.write(line if line.endswith("\n") else line + "\n")
    counts["to_embed_texts"] = len(pick)
    return counts


def new_vectors(new: list[Path], texts: dict[str, bytes]) -> dict[bytes, np.ndarray]:
    """Vectors from embed_views shards, keyed by the text of the view they embed."""
    out: dict[bytes, np.ndarray] = {}
    for d in new:
        for npy in sorted(d.glob("shard_*.npy")):
            ids = json.loads((npy.parent / (npy.stem + ".ids.json")).read_text())
            vecs = np.load(npy)
            if len(ids) != len(vecs):
                raise SystemExit(f"{npy}: {len(vecs)} vectors for {len(ids)} ids")
            for vid, vec in zip(ids, vecs):
                if vid not in texts:
                    raise SystemExit(f"{npy}: view {vid} is not in the to_embed files given with --texts")
                out[texts[vid]] = vec
    return out


def assemble(corpus: Path, db: Path, new: list[Path], texts_files: list[Path], out: Path,
             skip_types: set[str], min_arabic: int) -> dict:
    import faiss

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = stored_rows(db)
    texts: dict[str, bytes] = {}
    for f in texts_files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    texts[r["view_id"]] = text_key(r.get("embedding_text") or "")
    fresh = new_vectors(new, texts)
    index = faiss.read_index(str(db / "dense_faiss_ar.index"), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    counts = {"views": 0, "reused": 0, "new": 0}
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    shard = 0

    def flush() -> None:
        nonlocal shard, ids, vecs
        if not ids:
            return
        np.save(out / f"shard_{shard:05d}.npy", np.stack(vecs).astype(np.float32))
        (out / f"shard_{shard:05d}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
        shard, ids, vecs = shard + 1, [], []

    for vid, text in embeddable(corpus, skip_types, min_arabic):
        k = text_key(text)
        if k in rows:
            vecs.append(index.reconstruct(rows[k]))
            counts["reused"] += 1
        elif k in fresh:
            vecs.append(fresh[k].astype(np.float32))
            counts["new"] += 1
        else:
            raise SystemExit(f"no vector for view {vid}: not in the stores and not in --new")
        ids.append(vid)
        counts["views"] += 1
        if len(ids) >= SHARD_SIZE:
            flush()
    flush()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", type=Path, required=True, help="structured dir with retrieval_views.jsonl")
        p.add_argument("--db", type=Path, required=True, help="stores whose vectors are reused (read only)")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--skip-view-types", default="")
        p.add_argument("--min-arabic", type=int, default=3)
        if name == "assemble":
            p.add_argument("--new", type=Path, action="append", default=[], help="embed_views output dir (repeatable)")
            p.add_argument("--texts", type=Path, action="append", default=[],
                           help="the to_embed.jsonl files those dirs were embedded from (repeatable)")
    args = ap.parse_args(argv)
    skip = {t for t in args.skip_view_types.split(",") if t}
    if args.cmd == "plan":
        counts = plan(args.corpus, args.db, args.out, skip, args.min_arabic)
    else:
        counts = assemble(args.corpus, args.db, args.new, args.texts, args.out, skip, args.min_arabic)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
