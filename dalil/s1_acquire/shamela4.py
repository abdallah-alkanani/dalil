"""Fetch single books from the AuthenticIlm/Shamela4_Full_DB Hugging Face dataset.

The collection's own Shamela books (raw/shamela4_modern_scholars/) came from the
same dataset, so the four per-book files are downloaded unchanged and the
existing Shamela parser reads them as is. Only the requested books are fetched,
pinned to one dataset revision, one request per second, and each file is
checked against the SHA-256 the dataset's own manifest.json records.

    python -m dalil.s1_acquire.shamela4 --out DIR --shamela-id 251 --shamela-id 390 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dalil.paths import RECIPE_DIR

DATASET = "AuthenticIlm/Shamela4_Full_DB"
# Dataset revision seen on 2026-10-16 (huggingface.co/api/datasets/AuthenticIlm/Shamela4_Full_DB).
REVISION = "07554bee488a12955dd5231d08487ae7ce767d1e"
USER_AGENT = "Mozilla/5.0 (dalil research corpus)"
CATALOG = RECIPE_DIR / "sources" / "shamela4_catalog.jsonl"  # Shamela id -> the dataset's folder for the book
FILES = ("manifest.json", "book_metadata.json", "toc.jsonl", "pages.jsonl")


def fetch(path: str, dest: Path) -> None:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)
    time.sleep(1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shamela-id", type=int, action="append", required=True)
    args = ap.parse_args(argv)

    catalog = {}
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                catalog[row["shamela_id"]] = row
    log = []
    for sid in args.shamela_id:
        row = catalog[sid]
        book_dir = args.out / row["path"]
        book_dir.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            dest = book_dir / name
            if not dest.exists():
                fetch(f"{row['path']}/{name}", dest)
        manifest = json.loads((book_dir / "manifest.json").read_text(encoding="utf-8"))
        checks = {}
        for entry in manifest["files"]:
            actual = sha256(book_dir / entry["path"])
            checks[entry["path"]] = actual == entry["sha256"]
        rows = sum(1 for line in (book_dir / "pages.jsonl").open(encoding="utf-8") if line.strip())
        ok = all(checks.values()) and rows == manifest["page_count"]
        log.append({"shamela_id": sid, "dataset_book_id": row["book_id"], "path": row["path"],
                    "sha256_ok": checks, "pages": rows, "page_count_manifest": manifest["page_count"], "ok": ok})
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    (args.out / "fetch_log.json").write_text(json.dumps(
        {"dataset": DATASET, "revision": REVISION, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "books": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(b["ok"] for b in log) else 1


if __name__ == "__main__":
    sys.exit(main())
