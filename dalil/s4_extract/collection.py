"""Build the 545-book collection: apply each book's parser profile (chosen by the
structure audit, ``dalil.s3_structure.parent_policy``) and emit the
element / parent / retrieval-view corpus plus the Quran-reference and isnad side
indexes. It does not embed, index, or touch the search stores.

    python -m dalil.s4_extract.collection --output-root data/build/structured/collection \
        --boundaries recipe/decisions/semantic_boundaries.jsonl

Parsers by source format: OpenITI TXT, Shamela 4 JSON, Ketabonline JSON, Ibn Baz
fatwa JSONL and saaid.net TXT (``dalil.s4_extract.sources``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from dalil.paths import BUILD_DIR, DATA_DIR, RECIPE_BOOKS

from .boundaries import (
    BOUNDARY_FLAG_PENDING,
    PARENT_TARGET_WORDS,
    load_semantic_boundaries,
)
from .records import BuildState, write_json
from .sources.ibn_baz import parse_ibn_baz
from .sources.ketabonline import parse_ketabonline_book
from .sources.openiti import parse_openiti
from .sources.saaid import parse_saaid
from .sources.shamela import parse_shamela

MANIFEST_PATH = RECIPE_BOOKS / "collection_manifest.jsonl"
PROFILES_PATH = RECIPE_BOOKS / "collection_profiles.jsonl"
EXCLUSIONS_PATH = RECIPE_BOOKS / "collection_record_exclusions.jsonl"
REVIEW_PATH = RECIPE_BOOKS / "collection_record_review.jsonl"
KETAB_ELEMENTS_PATH = BUILD_DIR / "prepared" / "ketabonline_handoff" / "elements.jsonl"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["book_id"]] = r
    return rows


def load_profiles(path: Path = PROFILES_PATH) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["book_id"]] = r
    return rows


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> dict[str, set[str]]:
    """book_id -> set of excluded record ids.

    OpenITI record ids are stored fully qualified as '<source_unit_id>::PageV01P190'
    but the TXT itself only carries the bare 'PageV01P190' marker, so both forms are
    registered and the readers can match on whichever they have.
    """
    out: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rid = str(r.get("record_id"))
                out[r.get("book_id")].add(rid)
                if "::" in rid:
                    out[r.get("book_id")].add(rid.rsplit("::", 1)[1])
    return out


def load_review(path: Path = REVIEW_PATH) -> dict[str, set[str]]:
    """book_id -> set of review-flagged record_ids (shamela page ids)."""
    out: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r.get("book_id")].add(str(r.get("record_id")))
    return out


def build_profile(man_row: dict[str, Any], rec_row: dict[str, Any]) -> dict[str, Any]:
    p = dict(man_row)
    p.update(rec_row)
    p["source_paths"] = [man_row["source_path"]]
    if man_row.get("clean_source_path"):
        p["source_paths"].append(man_row["clean_source_path"])
    return p


def record_coverage(profile: dict[str, Any], state: BuildState, before: tuple[int, int, int]) -> None:
    after = (len(state.elements), len(state.parents), len(state.views))
    state.report_rows.append(
        {
            "book_id": profile["book_id"],
            "parser_profile": profile["parser_profile"],
            "parent_policy": profile["parent_policy"],
            "elements": after[0] - before[0],
            "parents": after[1] - before[1],
            "retrieval_views": after[2] - before[2],
        }
    )
    if state.footnote_counts:
        state.report_rows[-1]["footnotes"] = dict(state.footnote_counts)
        state.footnote_counts.clear()


TITLE_NOISE_RE = re.compile(r"[\d\u0660-\u0669\s/:.\-]+")


def drop_empty_duplicates(state: BuildState, start: int = 0) -> None:
    """Drop an empty parent that only repeats the title of the parent after it.

    Some sources list a heading twice, once from the book's index and once from the
    text, with different digits (س/200 and س/٢٠٠): the first becomes a parent with
    no text and the answer sits in the second. An empty parent before a *different*
    title is kept, because it is the chapter heading over the sections that follow.
    """
    def title(p: dict[str, Any]) -> str:
        return TITLE_NOISE_RE.sub("", (p.get("section_path") or [""])[-1])

    book = state.parents[start:]  # in reading order
    drop = {i for i, (p, q) in enumerate(pairwise(book))
            if not (p.get("text_original") or "").strip() and (q.get("text_original") or "").strip()
            and title(p) and title(p) == title(q)}
    if not drop:
        return
    views = {v for i in drop for v in book[i].get("child_view_ids", [])}
    state.parents[start:] = [p for i, p in enumerate(book) if i not in drop]
    state.views[:] = [v for v in state.views if v["view_id"] not in views]


def parse_one(profile: dict[str, Any], state: BuildState) -> None:
    before = (len(state.elements), len(state.parents), len(state.views))
    parser = profile["parser_profile"]
    if parser == "openiti_txt_parser":
        parse_openiti(profile, state)
    elif parser == "shamela4_json_parser":
        parse_shamela(profile, state)
    elif parser == "ibn_baz_fatwa_jsonl_parser":
        parse_ibn_baz(profile, state)
    elif parser == "saaid_txt_parser":
        parse_saaid(profile, state)
    else:
        raise RuntimeError(f"unhandled parser {parser} for {profile['book_id']}")
    drop_empty_duplicates(state, before[1])
    record_coverage(profile, state, before)


# Rows of the Ketabonline elements file are compact JSON with sorted keys, so the
# top-level book_id is the first one on the line. Workers re-check it after parsing.
KETAB_BOOK_ID_RE = re.compile(rb'"book_id"\s*:\s*"([^"\\]*)"')


def ketabonline_book_ranges(units: set[str], path: Path = KETAB_ELEMENTS_PATH) -> dict[str, tuple[int, int]]:
    """Byte range of each wanted book's rows in the 2.5 GB Ketabonline elements file.

    One cheap pass that reads book ids without parsing JSON, so every book can
    then be parsed in its own worker. Rows must be grouped by book.
    """
    ranges: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    current: str | None = None
    start = offset = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                m = KETAB_BOOK_ID_RE.search(line)
                unit = m.group(1).decode("utf-8") if m else None
                if unit != current:
                    if current in units:
                        ranges[current] = (start, offset)
                    if unit in seen and unit in units:
                        raise RuntimeError(
                            f"ketabonline elements are not grouped by book_id: {unit} reappears after being processed")
                    seen.add(unit)
                    current, start = unit, offset
            offset += len(line)
    if current in units:
        ranges[current] = (start, offset)
    missing = units - set(ranges)
    if missing:
        raise RuntimeError(f"ketabonline books missing from elements file: {sorted(missing)[:10]}")
    return ranges


def read_ketabonline_book(unit: str, byte_range: tuple[int, int], path: Path = KETAB_ELEMENTS_PATH) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        f.seek(byte_range[0])
        data = f.read(byte_range[1] - byte_range[0])
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(r.get("book_id") != unit for r in records):
        raise RuntimeError(f"ketabonline byte range for {unit} holds rows of another book")
    return records


KetabSlice = tuple[str, tuple[int, int]]  # (elements file, byte range of one book)


def parse_book(task: tuple[dict[str, Any], KetabSlice | None]) -> BuildState:
    """Parse one book into a fresh BuildState. ``task`` = (profile, Ketabonline slice or None).

    The elements path travels with the task: worker processes re-import this
    module, so a path changed in main() would not reach them.
    """
    profile, ketab = task
    state = BuildState([], [], [], [], [], [])
    if ketab is None:
        parse_one(profile, state)
    else:
        path, byte_range = ketab
        parse_ketabonline_book(profile, read_ketabonline_book(profile["source_unit_id"], byte_range, Path(path)), state)
        drop_empty_duplicates(state)
        record_coverage(profile, state, (0, 0, 0))
    return state


def _parse_book_worker(task: tuple[dict[str, Any], KetabSlice | None]) -> dict[str, list[str]]:
    """Parse one book in a worker process and return serialised JSONL rows.

    Rows are serialised here rather than in the parent so the pickling cost is
    paid on the worker side and the parent only concatenates strings.
    """
    state = parse_book(task)
    dump = lambda rows: [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    return {
        "elements": dump(state.elements),
        "parents": dump(state.parents),
        "retrieval_views": dump(state.views),
        "quran_ref_index": dump(state.quran_refs),
        "isnad_edges": dump(state.isnad_edges),
        "report_rows": dump(state.report_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--book-id", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new directory for the build (existing JSONL files there are overwritten)")
    parser.add_argument(
        "--boundaries", type=Path, default=None,
        help="semantic boundary decisions from dalil.s3_structure.semantic_boundaries; "
             "oversized sections with no recorded boundary are emitted whole and flagged "
             f"{BOUNDARY_FLAG_PENDING} rather than cut at a word offset",
    )
    parser.add_argument(
        "--parent-target-words", type=int, default=PARENT_TARGET_WORDS,
        help="wanted parent size when applying boundaries. Recorded boundaries are "
             "denser than this; the nearest one to each target is used, so "
             "pieces land on real boundaries near this size. Changing it re-cuts "
             "from the same boundary file without asking the model again",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="books are independent and parsed in parallel, largest first. With 1, books "
             "are written in a fixed order; with more, in completion order (ids do not "
             "depend on order)",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="books manifest (default: the collection's); use another file to build added books")
    parser.add_argument("--profiles", type=Path, default=PROFILES_PATH, help="parser profiles for those books")
    parser.add_argument("--exclusions", type=Path, default=EXCLUSIONS_PATH, help="record exclusions (optional file)")
    parser.add_argument("--review", type=Path, default=REVIEW_PATH, help="record review queue (optional file)")
    parser.add_argument("--ketab-elements", type=Path, default=KETAB_ELEMENTS_PATH,
                        help="Ketabonline elements.jsonl from dalil.s2_prepare.ketabonline")
    args = parser.parse_args(argv)

    out_root: Path = args.output_root
    manifest = load_manifest(args.manifest)
    profiles = load_profiles(args.profiles)
    exclusions = load_exclusions(args.exclusions)
    review = load_review(args.review)

    ordered = [build_profile(manifest[bid], profiles[bid]) for bid in manifest if bid in profiles]
    if args.book_id:
        wanted = set(args.book_id)
        ordered = [p for p in ordered if p["book_id"] in wanted]
    if args.limit:
        ordered = ordered[: args.limit]

    boundaries = load_semantic_boundaries(args.boundaries)
    if args.boundaries:
        print(f"semantic boundaries: {sum(len(v) for v in boundaries.values()):,} parents "
              f"across {len(boundaries):,} books", file=sys.stderr)
    for p in ordered:
        p["_excluded_record_ids"] = exclusions.get(p["book_id"], set())
        p["_review_record_ids"] = review.get(p["book_id"], set())
        p["_semantic_boundaries"] = boundaries.get(p["book_id"], {})
        p["_parent_target_words"] = args.parent_target_words

    plain = [p for p in ordered if p["source_corpus"] != "ketabonline"]
    ketab = {p["source_unit_id"]: p for p in ordered if p["source_corpus"] == "ketabonline"}
    ranges = ketabonline_book_ranges(set(ketab), args.ketab_elements) if ketab else {}
    # Plain books in manifest order, then Ketabonline books in file order.
    tasks: list[tuple[dict[str, Any], KetabSlice | None]] = [(p, None) for p in plain]
    tasks += [(ketab[unit], (str(args.ketab_elements), byte_range))
              for unit, byte_range in sorted(ranges.items(), key=lambda kv: kv[1][0])]

    out_root.mkdir(parents=True, exist_ok=True)
    names = ("elements", "parents", "retrieval_views", "quran_ref_index", "isnad_edges")
    out_files = {name: (out_root / f"{name}.jsonl").open("w", encoding="utf-8") for name in names}
    totals: Counter = Counter()
    report_rows: list[dict[str, Any]] = []

    def write(payload: dict[str, list[str]]) -> None:
        for name, handle in out_files.items():
            if payload[name]:
                handle.write("\n".join(payload[name]) + "\n")
            totals[name] += len(payload[name])
        report_rows.extend(json.loads(r) for r in payload["report_rows"])

    if args.workers > 1 and len(tasks) > 1:
        def size(task: tuple[dict[str, Any], KetabSlice | None]) -> int:
            profile, ketab = task
            if ketab is not None:
                return ketab[1][1] - ketab[1][0]
            path = DATA_DIR / profile["source_path"]
            return path.stat().st_size if path.exists() else 0
        # Largest first, so one big book does not start last and finish alone.
        tasks.sort(key=size, reverse=True)
        with Pool(args.workers) as pool:
            for payload in pool.imap_unordered(_parse_book_worker, tasks, chunksize=1):
                write(payload)
    else:
        for task in tasks:
            write(_parse_book_worker(task))

    for f in out_files.values():
        f.close()

    summary = {
        "schema": "dalil.parser_coverage_report.v1",
        "books_processed": len(ordered),
        "total_elements": totals["elements"],
        "total_parents": totals["parents"],
        "total_retrieval_views": totals["retrieval_views"],
        "total_quran_refs": totals["quran_ref_index"],
        "total_isnad_edges": totals["isnad_edges"],
        "excluded_records": {bid: len(ids) for bid, ids in exclusions.items()},
        "review_records": {bid: len(ids) for bid, ids in review.items()},
        "by_parser_profile": dict(Counter(row["parser_profile"] for row in report_rows)),
        "footnotes": dict(sum((Counter(row.get("footnotes", {})) for row in report_rows), Counter())),
        "rows": report_rows,
    }
    write_json(out_root / "parser_coverage_report.json", summary)
    print(
        json.dumps(
            {k: summary[k] for k in ("books_processed", "total_elements", "total_parents", "total_retrieval_views", "total_quran_refs", "total_isnad_edges", "footnotes")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
