"""Download Ketabonline's structured text packages for the target scholars.

This downloader requests only the API's ``files.data`` endpoint.  It does not
request audio, PDF, image, or reader-media endpoints.  The downloaded data
packages are kept as source evidence and only JSON members are extracted for
later cleaning/deduplication.  Nothing here writes to FAISS, SQLite, or the
active corpus.

    python -m dalil.s1_acquire.ketabonline [--output DIR] [--delay SECONDS]
    python -m dalil.s1_acquire.ketabonline --output DIR --book 92317:walid_alsaeedan --book 6006:al_albani
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dalil.paths import DATA_DIR

API = "https://backend.ketabonline.com/api/v2/books"
USER_AGENT = "Mozilla/5.0 (dalil text-source download)"
AUTHORS = {
    67: ("ibn_baz", "Ibn Baz"),
    414: ("al_albani", "al-Albani"),
    14: ("ibn_al_qayyim", "Ibn al-Qayyim"),
}
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".mp4", ".webm",
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
}
# Downloads land here in Ketabonline's own layout (<author_slug>/raw/<id>.data.zip,
# catalog.json, download_manifest.json). A package the recipe uses then goes to its
# book folder, books/original/<author>/<book>/, and its records to recipe/sources/.
DEFAULT_OUTPUT = DATA_DIR / "downloads" / "ketabonline"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def fetch_catalog(author_id: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        query = (
            f"{API}?is_active=1&is_deleted=0&page={page}&limit=200&q="
            f"&scope=titles&sort_field=created_at&sort_direction=DESC"
            f"&authors[]={author_id}"
        )
        payload = request_json(query)
        batch = payload.get("data") or []
        records.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return records


def fetch_books(specs: list[str]) -> list[dict[str, Any]]:
    """Catalog rows for single books, each named ``<ketabonline id>:<author slug>``.

    The slug is the package's folder. A slug of one of AUTHORS gets that scholar's
    id and label, as a book of his catalog does (مشكاة المصابيح lists al-Tabrizi
    first); any other slug gets the book's first listed author.
    """
    by_slug = {slug: (author_id, label) for author_id, (slug, label) in AUTHORS.items()}
    records: list[dict[str, Any]] = []
    for spec in specs:
        book_id, slug = spec.split(":", 1)
        record = request_json(f"{API}/{book_id}")["data"]
        first = (record.get("authors") or [{}])[0]
        author_id, label = by_slug.get(slug, (first.get("id"), first.get("name")))
        item = {k: v for k, v in record.items() if k not in ("index", "pages")}
        item.update(target_author_id=author_id, target_author_slug=slug, target_author_label=label,
                    package_kind="text_transcript" if is_transcript(record) else "text_book")
        records.append(item)
    return records


def is_transcript(record: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(record.get(key) or "")
        for key in ("title", "description", "source")
    )
    return bool(re.search(r"تفريغ|سلسلة الهدى والنور|لقاءات الشيخ", haystack))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_size or destination.stat().st_size == expected_size):
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    if expected_size and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {destination.name}: "
            f"expected {expected_size}, got {partial.stat().st_size}"
        )
    partial.replace(destination)


def extract_json_members(archive_path: Path, extracted_dir: Path, book_id: int) -> list[str]:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"corrupt zip member: {bad_member}")
        for member in archive.infolist():
            member_name = Path(member.filename).name
            suffix = Path(member_name).suffix.lower()
            if not member_name or suffix in MEDIA_EXTENSIONS or suffix != ".json":
                continue
            destination = extracted_dir / f"{book_id}.data.json"
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            extracted.append(destination.name)
            break
    if not extracted:
        raise RuntimeError("no JSON text member found in data package")
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="source staging directory (default: %(default)s). Not raw/ketabonline "
             "itself: that is the deduplicated canonical copy, and a re-run there "
             "would rewrite its catalog and manifest",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="delay between downloads")
    parser.add_argument("--book", action="append", default=[], metavar="ID:AUTHOR_SLUG",
                        help="download only this book (repeatable) and add it to --output's catalog "
                             "and manifest, instead of every book of the target scholars")
    args = parser.parse_args(argv)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    manifest_path = root / "download_manifest.json"

    all_records: list[dict[str, Any]] = []
    if args.book:
        all_records = fetch_books(args.book)
    else:
        for author_id, (slug, label) in AUTHORS.items():
            print(f"Fetching {label} catalog ({author_id})...", flush=True)
            for record in fetch_catalog(author_id):
                item = dict(record)
                item["target_author_id"] = author_id
                item["target_author_slug"] = slug
                item["target_author_label"] = label
                item["package_kind"] = "text_transcript" if is_transcript(record) else "text_book"
                all_records.append(item)

    catalog_records = all_records
    if args.book and catalog_path.exists():  # single books join the books already listed there
        by_id = {str(r["id"]): r for r in json.loads(catalog_path.read_text(encoding="utf-8")).get("records", [])}
        by_id.update((str(r["id"]), r) for r in all_records)
        catalog_records = list(by_id.values())
    catalog = {
        "generated_at": now_iso(),
        "source": "https://ketabonline.com",
        "api": API,
        "download_scope": "files.data only; no audio/PDF/image endpoints",
        "records": catalog_records,
    }
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: ignoring invalid existing {manifest_path}", file=sys.stderr)

    total = len(all_records)
    for index, record in enumerate(all_records, start=1):
        author_slug = record["target_author_slug"]
        book_id = int(record["id"])
        files = record.get("files") or {}
        data_file = files.get("data") or {}
        url = data_file.get("url")
        expected_size = int(data_file.get("size") or 0)
        key = str(book_id)
        if not url:
            manifest[key] = {"status": "no_data_url", "book_id": book_id, "updated_at": now_iso()}
            continue

        raw_dir = root / author_slug / "raw"
        extracted_dir = root / author_slug / "json"
        archive_path = raw_dir / f"{book_id}.data.zip"
        try:
            print(f"[{index}/{total}] {author_slug} {book_id}: {record.get('title', '')}", flush=True)
            download_file(url, archive_path, expected_size)
            extracted = extract_json_members(archive_path, extracted_dir, book_id)
            actual_size = archive_path.stat().st_size
            manifest[key] = {
                "status": "downloaded",
                "book_id": book_id,
                "target_author_id": record["target_author_id"],
                "target_author_slug": author_slug,
                "title": record.get("title"),
                "source": record.get("source"),
                "package_kind": record["package_kind"],
                "pages_count": record.get("pages_count"),
                "expected_size": expected_size,
                "archive_size": actual_size,
                "archive_sha256": sha256_file(archive_path),
                "extracted_json": extracted,
                "url": url,
                "updated_at": now_iso(),
            }
        except Exception as exc:  # keep the batch resumable
            manifest[key] = {
                "status": "failed",
                "book_id": book_id,
                "target_author_id": record["target_author_id"],
                "target_author_slug": author_slug,
                "title": record.get("title"),
                "url": url,
                "error": repr(exc),
                "updated_at": now_iso(),
            }
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(args.delay, 0.0))

    counts: dict[str, int] = {}
    for item in manifest.values():
        status = item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"records": total, "statuses": counts, "output": str(root)}, ensure_ascii=False))
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
