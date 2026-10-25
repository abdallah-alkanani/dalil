"""Validate a structured corpus build (elements / parents / retrieval_views and
the Quran-reference and isnad side indexes).

- Every file is read in parallel (``--workers``), cut into byte ranges.
- JSON Schema validation: every row with ``--sample 0``, otherwise about ``--sample``
  rows per dataset (every n-th line).
- On every row: required keys, id prefixes, duplicate ids, and no leaked
  source-format delimiters (page markers, Shamela field names).
- Cross-references: element parents/ancestors, parent source elements,
  parent <-> view links, Quran-reference and isnad-edge sources.

    python -m dalil.s4_extract.validate --root data/build/structured/core --sample 0
    python -m dalil.s4_extract.validate --root data/build/structured/collection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

DATA = {
    "elements": ("elements.jsonl", "element_id"),
    "parents": ("parents.jsonl", "parent_id"),
    "retrieval_views": ("retrieval_views.jsonl", "view_id"),
    "quran_ref_index": ("quran_ref_index.jsonl", "quran_ref_id"),
    "isnad_edges": ("isnad_edges.jsonl", "edge_id"),
}
# The side indexes are optional; a build without them is still valid.
REQUIRED_DATASETS = ("elements", "parents", "retrieval_views")

REQUIRED = {
    "elements": ["schema_version", "element_id", "book_id", "work_id", "author_id",
                 "source_corpus", "file_format", "source_path", "source_record_id",
                 "element_type", "level", "parent_element_id", "ancestor_ids",
                 "order_in_parent", "section_path", "section_path_ids", "refs",
                 "text_original", "text_clean", "normalized_text", "char_span",
                 "token_counts", "parser_profile", "parent_policy", "view_policy",
                 "recommended_detectors", "parser", "quality_flags"],
    "parents": ["schema_version", "parent_id", "book_id", "work_id", "author_id",
                "category", "subcategory", "parent_type", "source_element_ids",
                "ancestor_ids", "section_path", "section_path_ids", "refs",
                "text_for_synthesis", "text_original", "normalized_text", "citation",
                "child_view_ids", "token_counts", "parser_profile", "parent_policy",
                "view_policy", "quality_flags"],
    "retrieval_views": ["schema_version", "view_id", "parent_id", "book_id", "work_id",
                        "author_id", "category", "subcategory", "source_corpus",
                        "file_format", "view_type", "text_role", "order_in_parent",
                        "section_path", "section_path_ids", "refs", "display_text",
                        "normalized_text", "embedding_text", "token_counts",
                        "embedding_profile", "parser_profile", "parent_policy",
                        "view_policy", "quality_flags"],
    "quran_ref_index": ["schema_version", "quran_ref_id", "surah", "ayah_start",
                        "ayah_end", "source_kind", "source_id", "book_id",
                        "match_method", "confidence", "matched_text", "quality_flags"],
    "isnad_edges": ["schema_version", "edge_id", "hadith_parent_id", "collection",
                    "book_id", "narrator_a", "narrator_b", "order",
                    "extraction_confidence", "quality_flags"],
}

ID_PATTERNS = {
    "elements": "elem:",
    "parents": "parent:",
    "retrieval_views": "view:",
    "quran_ref_index": "qref:",
    "isnad_edges": "edge:",
}

ID_PATTERNS = {
    "elements": "elem:",
    "parents": "parent:",
    "retrieval_views": "view:",
    "quran_ref_index": "qref:",
    "isnad_edges": "edge:",
}

# Source-format delimiters that must never reach the output. Assembled from parts
# so this file does not itself contain them.
FORBIDDEN_OUTPUT_STRINGS = (
    "pa" "ge_marker",
    "pa" "ge_marker_detector",
    "P" "ageV",
    "معرف" "_الشاملة",
    "رقم" "_الصفحة",
)


# jsonschema takes ~0.5 ms a row, 20x the JSON parse, and the JSON parse of a 5 GB
# build is minutes on one core. Each file is cut into ranges of this size and every
# range is read by a pool worker; this process only joins the results in file order.
RANGE_BYTES = 64 * 1024 * 1024


@cache
def _validator(schema_dir: Path, name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")))


def _ref_checks(name: str, r: dict) -> list[tuple[str, Any]]:
    """The row's references to other rows, as (kind, id), in the order they are reported."""
    if name == "elements":
        out = [("pe", r.get("parent_element_id"))] if r.get("parent_element_id") else []
        return out + [("anc", a) for a in r.get("ancestor_ids", [])]
    if name == "parents":
        out = [("src", e) for e in r.get("source_element_ids", [])]
        for a in r.get("ancestor_ids", []):
            if a.startswith("elem:"):
                out.append(("eanc", a))
            if a.startswith("parent:"):
                out.append(("panc", a))
        return out + [("cv", v) for v in r.get("child_view_ids", [])]
    if name == "retrieval_views":
        return [("vp", r.get("parent_id"))]
    if name == "quran_ref_index":
        kind = {"parent": "qp", "element": "qe", "retrieval_view": "qv"}.get(r.get("source_kind"))
        return [(kind, r.get("source_id"))] if kind else []
    return [("isn", r.get("hadith_parent_id"))]


# kind -> (ids it must be among, message after "<dataset>:<line> ")
REF_MESSAGES = {
    "pe": ("elements", "{rid} parent_element_id not found: {v}"),
    "anc": ("elements", "{rid} ancestor not found: {v}"),
    "src": ("elements", "{rid} source_element not found: {v}"),
    "eanc": ("elements", "{rid} elem ancestor not found: {v}"),
    "panc": ("parents", "{rid} parent ancestor not found: {v}"),
    "cv": ("retrieval_views", "{rid} child_view not found: {v}"),
    "vp": ("parents", "{rid} parent not found: {v}"),
    "qp": ("parents", "parent source not found: {v}"),
    "qe": ("elements", "element source not found: {v}"),
    "qv": ("retrieval_views", "view source not found: {v}"),
    "isn": ("parents", "hadith_parent not found: {v}"),
}


def scan_range(task: tuple[Path, Path, str, int, int, int]) -> dict[str, Any]:
    """Check the rows of one file that start in bytes [start, end).

    Line numbers are local to the range. Returns the row checks (`errors`), every
    id, the schema errors of every `schema_every`-th line (0: none), and the
    references this range cannot settle itself (a reference to a row of the same
    file and range is checked here and not returned).
    """
    path, schema_dir, name, start, end, schema_every = task
    idf = DATA[name][1]
    validator = _validator(schema_dir, name) if schema_every else None
    errors: list[tuple[int, str]] = []
    ids: list[tuple[int, Any]] = []
    refs: list[tuple[int, Any, str, Any]] = []
    schema: list[str] = []
    ln = 0
    with path.open("rb") as f:
        if start:
            f.seek(start - 1)
            f.readline()  # the row running across `start` belongs to the range before
        pos = f.tell()
        for raw in f:
            if pos >= end:
                break
            pos += len(raw)
            ln += 1
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                errors.append((ln, "JSON parse error"))
                continue
            rid = r.get(idf)
            leaked = next((s for s in FORBIDDEN_OUTPUT_STRINGS if s in line), None)
            if leaked:
                errors.append((ln, f"{rid} source delimiter leaked into output: {leaked}"))
            missing = [k for k in REQUIRED[name] if k not in r]
            if missing:
                errors.append((ln, f"{rid} missing keys: {missing[:5]}"))
            if not isinstance(rid, str) or not rid.startswith(ID_PATTERNS[name]):
                errors.append((ln, f"bad {idf}: {rid}"))
            ids.append((ln, rid))
            refs.extend((ln, rid, kind, v) for kind, v in _ref_checks(name, r))
            if validator and ln % schema_every == 0:
                schema.extend(f"{name} schema: {rid}: {'.'.join(str(p) for p in e.path)}: {e.message[:120]}"
                              for e in validator.iter_errors(r))
    own = {rid for _, rid in ids if isinstance(rid, str)}
    refs = [x for x in refs if not (REF_MESSAGES[x[2]][0] == name and x[3] in own)]
    return {"lines": ln, "errors": errors, "ids": ids, "refs": refs, "schema": schema}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory holding the *.jsonl corpus files")
    ap.add_argument("--schema-dir", type=Path, default=SCHEMA_DIR)
    ap.add_argument("--sample", type=int, default=2000,
                    help="about this many rows per dataset checked against the JSON Schema; 0 checks every row")
    ap.add_argument("--max-errors", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes reading the files")
    args = ap.parse_args(argv)
    root: Path = args.root
    schema_dir: Path = args.schema_dir
    for name in REQUIRED_DATASETS:
        if not (root / DATA[name][0]).exists():
            raise SystemExit(f"missing {root / DATA[name][0]}")
    present = {name: spec for name, spec in DATA.items() if (root / spec[0]).exists()}
    for name in present:
        Draft202012Validator.check_schema(json.loads((schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")))

    tasks = []
    for name, (fname, _) in present.items():
        path = root / fname
        size = path.stat().st_size
        every = 1
        if args.sample > 0:  # every n-th line, n from the file's size and its first lines' length
            with path.open("rb") as f:
                head = f.read(1 << 20)
            every = max(1, round(size / max(len(head) / max(head.count(b"\n"), 1), 1) / args.sample))
        tasks += [(path, schema_dir, name, start, start + RANGE_BYTES, every)
                  for start in range(0, max(size, 1), RANGE_BYTES)]
    with Pool(args.workers) as pool:
        results = pool.map(scan_range, tasks, chunksize=1)

    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in present}
    for task, res in zip(tasks, results):
        by_name[task[2]].append(res)

    errors: list[str] = []
    counts: dict[str, int] = {}
    ids: dict[str, set[Any]] = {}
    # rows, in file order: the row checks, then whether the id was seen before
    for name, parts in by_name.items():
        idf = DATA[name][1]
        seen: set[Any] = set()
        off = 0
        for res in parts:
            rows: dict[int, list[str]] = {}
            for ln, msg in res["errors"]:
                rows.setdefault(ln, []).append(msg)
            dup, rid_at = set(), dict(res["ids"])
            for ln, rid in res["ids"]:
                if rid in seen:
                    dup.add(ln)
                    rows.setdefault(ln, [])
                seen.add(rid)
            for ln in sorted(rows):
                errors.extend(f"{name}:{off + ln} {m}" for m in rows[ln])
                if ln in dup:
                    errors.append(f"{name}:{off + ln} duplicate {idf}: {rid_at[ln]}")
            off += res["lines"]
        counts[name] = sum(len(r["ids"]) for r in parts)
        ids[name] = seen
    errors.extend(e for parts in by_name.values() for res in parts for e in res["schema"])
    # references, in the order the rows and their fields come
    for name in ("elements", "parents", "retrieval_views", "quran_ref_index", "isnad_edges"):
        off = 0
        for res in by_name.get(name, []):
            for ln, rid, kind, v in res["refs"]:
                target, msg = REF_MESSAGES[kind]
                if v not in ids.get(target, ()):
                    errors.append(f"{name}:{off + ln} " + msg.format(rid=rid, v=v))
            off += res["lines"]

    print("counts:", counts)
    print("errors:", len(errors))
    for e in errors[: args.max_errors]:
        print("  -", e)
    if errors:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
