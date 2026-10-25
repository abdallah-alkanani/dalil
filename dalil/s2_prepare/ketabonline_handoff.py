"""Emit the Ketabonline handoff (elements.jsonl) that dalil.s4_extract.collection reads.

Each package is emitted from its own row of the handoff manifest
(recipe/books/ketabonline_handoff_manifest.jsonl: status, review flags and repaired
source_path as the 2026-08-16 dedup left them) with the current preparer.
Block-overlap hash sets are empty: in the August handoff they flagged 5 of 984,825
elements and nothing reads them. Packages are emitted in parallel (--workers) and
written in manifest order, so the output does not depend on the worker count.

    python -m dalil.s2_prepare.ketabonline_handoff OUT_DIR [--manifest FILE] [--workers N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from dalil.paths import DATA_DIR, RECIPE_BOOKS
from dalil.s2_prepare import ketabonline as K


def emit_package(row: dict) -> tuple[str, Counter]:
    """One package's elements.jsonl lines and its element counts."""
    out, stats = io.StringIO(), Counter()
    K.emit_elements(row, K.read_package(DATA_DIR / row["source_path"]), out, set(), set(), stats)
    return out.getvalue(), stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="new handoff directory; must not exist")
    ap.add_argument("--manifest", type=Path, default=RECIPE_BOOKS / "ketabonline_handoff_manifest.jsonl")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    stats: Counter = Counter()
    with open(args.out / "elements.jsonl", "w", encoding="utf-8") as out, Pool(args.workers) as pool:
        for text, counts in pool.imap(emit_package, rows):
            out.write(text)
            stats.update(counts)
    shutil.copy2(args.manifest, args.out / "books_manifest.jsonl")
    print(len(rows), "packages", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
