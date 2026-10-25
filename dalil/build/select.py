"""Choose what goes into the search stores, from the structured builds.

Every book of the builds goes in except those in recipe/decisions/drop_books.txt.
Garbage views are left out by the 2026-10-16 audit's rules (GARBAGE below), and
every view and parent gets its author id from recipe/books/authors.jsonl via the
catalog (recipe/books/catalog.json), or the collection manifest for books the
catalog lacks. Writes into --work:

  corpora/<name>/retrieval_views.jsonl, parents.jsonl   what dalil.s7_index.append reads
  corpora/<name>/elements.jsonl                          page fields only, for dalil.s7_index.navigation
  author_ids.json                                         book_id -> author id, for dalil.build.finalize
  SELECTION.json                                          counts and every decision

    python -m dalil.build.select --work DIR --corpus core=DIR --corpus collection=DIR
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dalil.paths import RECIPE_BOOKS, RECIPE_DECISIONS


def drop_list(path: Path) -> set[str]:
    """book_ids of a drop file: one per line, '#' starts a comment."""
    return {line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()} - {""}


# Garbage views, per book, by the audit's rules.
LETTER = r"\u0621-\u064A"
URLENC = re.compile(r"(?:[٠-٩A-F]{0,2}\s?%[0-9A-F٠-٩]{0,2}\s?)+")


def english_only(t):
    """Under 10% Arabic letters: an English translation. Mixed views keep their Arabic."""
    latin, arabic = len(re.findall(r"[A-Za-z]", t)), len(re.findall(f"[{LETTER}]", t))
    return latin > arabic and arabic / max(latin + arabic, 1) < 0.1


def url_heavy(t):
    return bool(re.search(r"%[0-9A-F٠-٩]{1,2}%", t)) and \
        sum(len(m.group(0)) for m in URLENC.finditer(t)) / max(len(t), 1) >= 0.5


GARBAGE = {
    "book:fiqh:ibn_baz_1330_1420:f74bc1a710c3": [
        ("web_css_debris", lambda t: re.search(r"tooltip|div#|a:\s?link|font-style|z-index|visibility:", t))],
    "book:hadith:al_albani_1332_1420:bbae966ebf1a": [
        ("pdf_copied_text", lambda t: re.search(rf"(?<![{LETTER}])[وبفل]?اهل[لاأ](?![{LETTER}])", t)),
        ("english_text", english_only)],
    "book:fiqh:ibn_al_qayyim_691_751:2170f3f28a88": [
        ("symbol_font_conversion", lambda t: re.search(r"\{[^{}]{0,40}- -", t) or re.search(r"(?:\(\s*-\s*){2,}", t) or re.search(
            r"(?:صلى الله عليه وسلم|رضي الله عنهم?|عليه السلام)\s*\(?\s*(?:صلى الله عليه وسلم|رضي الله عنهم?|عليه السلام)", t))],
    "book:hadith:al_albani_1332_1420:0b3a5d6b4f85": [("url_encoding_debris", url_heavy)],
    "book:aqeedah:al_albani_1332_1420:d2354a2c7007": [("url_encoding_debris", url_heavy)],
}


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="where corpora/, author_ids.json and SELECTION.json go")
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=DIR",
                    help="a structured build (elements, parents, retrieval_views); repeatable")
    ap.add_argument("--drop-books", type=Path, default=RECIPE_DECISIONS / "drop_books.txt")
    args = ap.parse_args(argv)
    work: Path = args.work
    corpora = dict(spec.split("=", 1) for spec in args.corpus)
    dropped = drop_list(args.drop_books)

    authors = {r["author_id"] for r in jsonl(RECIPE_BOOKS / "authors.jsonl")}
    catalog = json.loads((RECIPE_BOOKS / "catalog.json").read_text(encoding="utf-8"))["items"]
    author_of = {r["id"]: r["author_id"] for r in catalog}
    # The Quran and the Qudsi collection have no human author; the catalog says so.
    authorless = {r["id"] for r in catalog if r["author_id"] is None}
    for r in jsonl(RECIPE_BOOKS / "collection_manifest.jsonl"):
        author_of.setdefault(r["book_id"], r["author_id"])

    report = {"drop_books": str(args.drop_books), "dropped": [], "corpora": {}, "garbage_views": {}}
    taken: dict[str, str] = {}
    for name, build in corpora.items():
        build = Path(build)
        books_in_build = {v["book_id"] for v in jsonl(build / "parents.jsonl")}
        take = books_in_build - dropped
        report["dropped"] += sorted(books_in_build & dropped)
        unknown = sorted(b for b in take if b not in authorless and author_of.get(b) not in authors)
        if unknown:
            raise SystemExit(f"{name}: books without a registered author id: {unknown[:3]}")
        twice = sorted(b for b in take if b in taken)
        if twice:
            raise SystemExit(f"{name}: books already taken from {taken[twice[0]]}: {twice[:3]}")
        taken.update(dict.fromkeys(take, name))
        out = work / "corpora" / name
        out.mkdir(parents=True, exist_ok=True)
        gone: set[str] = set()
        reasons = Counter()
        n_views = 0
        with open(out / "retrieval_views.jsonl", "w", encoding="utf-8") as fv:
            for v in jsonl(build / "retrieval_views.jsonl"):
                if v["book_id"] not in take:
                    continue
                rule = next((r for r, test in GARBAGE.get(v["book_id"], []) if test(v.get("display_text") or "")), None)
                if rule:
                    gone.add(v["view_id"])
                    reasons[(v["book_id"], rule)] += 1
                    continue
                v["author_id"] = author_of[v["book_id"]]
                fv.write(json.dumps(v, ensure_ascii=False) + "\n")
                n_views += 1
        n_parents = emptied = 0
        need: set[str] = set()
        with open(out / "parents.jsonl", "w", encoding="utf-8") as fp:
            for p in jsonl(build / "parents.jsonl"):
                if p["book_id"] not in take:
                    continue
                kids = p.get("child_view_ids") or []
                kept = [k for k in kids if k not in gone]
                if kids and not kept:
                    emptied += 1
                    continue
                p["child_view_ids"] = kept
                p["author_id"] = author_of[p["book_id"]]
                need.update(p.get("source_element_ids") or [])
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_parents += 1
        with open(out / "elements.jsonl", "w", encoding="utf-8") as fe:
            for e in jsonl(build / "elements.jsonl"):
                if e["element_id"] in need:
                    sr = (e.get("refs") or {}).get("source_record") or {}
                    fe.write(json.dumps({"element_id": e["element_id"], "refs": {"source_record": {
                        k: sr.get(k) for k in ("page_number", "part_name", "page_index")}}}, ensure_ascii=False) + "\n")
        report["corpora"][name] = {"build": str(build), "books": len(take), "views": n_views,
                                   "parents": n_parents, "parents_emptied_by_garbage": emptied}
        for (b, r), n in reasons.items():
            report["garbage_views"][f"{b} {r}"] = n
        print(name, report["corpora"][name], flush=True)

    report["books"] = len(taken)
    report["drop_ids_not_in_any_build"] = sorted(dropped - set(report["dropped"]))
    (work / "author_ids.json").write_text(
        json.dumps({b: author_of[b] for b in sorted(taken)}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (work / "SELECTION.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("books", "garbage_views")}, ensure_ascii=False, indent=1))
    print("dropped:", len(report["dropped"]), "| drop ids in no build:", len(report["drop_ids_not_in_any_build"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
