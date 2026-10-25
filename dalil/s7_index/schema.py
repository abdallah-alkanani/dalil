"""Schemas of the two SQLite stores that sit beside the FAISS index.

  faiss_metadata.sqlite  child_metadata  one row per vector; faiss_row = FAISS row
  parents.sqlite         parents         one row per parent (the unit sent for synthesis)

``bm25.sqlite`` is owned by ``dalil.s7_index.bm25``. ``create_stores`` is only for
bootstrapping a new, empty set of stores; it refuses to touch existing files.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

CHILD_METADATA_DDL = """
CREATE TABLE child_metadata (
    faiss_row INTEGER PRIMARY KEY,
    view_id TEXT NOT NULL UNIQUE,
    parent_id TEXT NOT NULL,
    source TEXT NOT NULL,
    book_id TEXT,
    category TEXT,
    subcategory TEXT,
    view_type TEXT,
    text_role TEXT,
    qwen_tokens INTEGER NOT NULL,
    source_line INTEGER NOT NULL,
    section_path_json TEXT,
    section_path_ids_json TEXT,
    refs_json TEXT,
    token_counts_json TEXT,
    display_text TEXT,
    embedding_text TEXT,
    metadata_json TEXT NOT NULL
);
CREATE INDEX idx_child_parent_id ON child_metadata(parent_id);
CREATE INDEX idx_child_book_id ON child_metadata(book_id);
CREATE INDEX idx_child_category ON child_metadata(category);
CREATE INDEX idx_child_view_type ON child_metadata(view_type);
"""

# The last five columns and idx_parents_nav are the navigation fields that
# dalil.s7_index.navigation adds to stores built before they existed; all nullable.
PARENTS_DDL = """
CREATE TABLE parents (
    parent_id TEXT PRIMARY KEY,
    book_id TEXT,
    category TEXT,
    subcategory TEXT,
    parent_type TEXT,
    citation TEXT,
    section_path_json TEXT,
    section_path_ids_json TEXT,
    refs_json TEXT,
    child_view_ids_json TEXT,
    text_for_synthesis TEXT NOT NULL,
    text_original TEXT,
    metadata_json TEXT NOT NULL,
    order_in_book INTEGER,
    page_number INTEGER,
    page_end INTEGER,
    part_name TEXT,
    page_index INTEGER
);
CREATE INDEX idx_parents_book_id ON parents(book_id);
CREATE INDEX idx_parents_category ON parents(category);
CREATE INDEX idx_parents_nav ON parents(book_id, order_in_book);
"""


def create_stores(db_dir: Path, dim: int = 4096) -> None:
    """Create an empty, consistent set of search stores in ``db_dir``: metadata,
    parents, an empty BM25 index and an empty ``IndexFlatIP``. Ready for
    ``dalil.s7_index.append``."""
    import faiss

    from dalil.s7_index.bm25 import FTS_DDL, write_meta

    db_dir = Path(db_dir)
    paths = {name: db_dir / name for name in
             ("faiss_metadata.sqlite", "parents.sqlite", "bm25.sqlite", "dense_faiss_ar.index")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing store: {path}")
    db_dir.mkdir(parents=True, exist_ok=True)
    for name, ddl in (("faiss_metadata.sqlite", CHILD_METADATA_DDL), ("parents.sqlite", PARENTS_DDL)):
        con = sqlite3.connect(paths[name])
        try:
            con.executescript(ddl)
        finally:
            con.close()
    con = sqlite3.connect(paths["bm25.sqlite"])
    try:
        con.execute(FTS_DDL)
        write_meta(con, 0)
        con.commit()
    finally:
        con.close()
    faiss.write_index(faiss.IndexFlatIP(dim), str(paths["dense_faiss_ar.index"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create an empty set of search stores.")
    ap.add_argument("--db", type=Path, required=True, help="new directory for the stores")
    args = ap.parse_args(argv)
    create_stores(args.db)
    print(f"created empty stores in {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
