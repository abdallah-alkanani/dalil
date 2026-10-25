"""Build the book catalog files from the 2026-10-16 audit (read-only on the audit).

    books.json              every clean book in the search DB, with its duplicate status
    needs_replacement.json  damaged books, and works we only have as broken PDF text
    not_embedded.json       clean material we have that is not in the search DB

Titles are cleaned by rule: title_ar is the work's title, edition holds what was
stripped from it (publisher, editor, "ط أخرى", ...), title_raw keeps the original.
A hand-written `notes` field survives re-runs.

Authors come from the registry `recipe/books/authors.jsonl`: author_id is the
registry id (old ids are mapped through `previous_ids`), author_ar its short name.
The author is the scholar whose words the book holds: a student who transcribed,
compiled, abridged or extracted them is a contributor, not the author. Someone
who adds their own scholarship (a sharh, a refutation, a biography) is the author.
For Shamela books, `contributors` lists the co-authors, editors and graders named
in the book's metadata (e.g. al-Albani's tahqiq of another author's work).

Previous outputs in --out are read first, so hand-written notes and replacement
candidates carry over.

    python -m dalil.catalog.books [--data DATA_DIR] [--out DATA_DIR/catalog]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dalil.paths import DATA_DIR, RECIPE_BOOKS

# Set by main() from --data / --out; the helpers below read them.
DATA = DATA_DIR
OUT = DATA_DIR / "catalog"
ENRICHED = ("notes",)


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_inputs() -> None:
    """The audit, the library index and the author registry, as module tables."""
    global library, quality, coverage, actions, member_of, planned_drops, held, albani_resolved, albani_verdicts
    global authors, renamed, by_alias, by_shamela
    audit = DATA / "catalog" / "audit"
    library = {r["book_id"]: r for r in jsonl(DATA / "catalog" / "library" / "INDEX.jsonl")}
    quality = {r["book_id"]: r for r in jsonl(audit / "quality_books.jsonl")}
    coverage = {r["book_id"]: r for r in jsonl(audit / "coverage_books.jsonl")}
    actions = {r["book_id"]: r for r in jsonl(audit / "actions_proposed.jsonl")}
    clusters = json.load(open(audit / "duplicates_clusters.json", encoding="utf-8"))["clusters"]
    member_of = {}
    for cl in clusters:
        for m in cl["members"]:
            member_of[m["id"]] = cl
        member_of.setdefault(cl["keeper"], cl)
    plan = OUT / "removal_plan"
    planned_drops = {line.strip() for line in open(plan / "drop_books.txt", encoding="utf-8")
                     if line.strip() and not line.startswith("#")} if (plan / "drop_books.txt").exists() else set()
    _held = json.load(open(plan / "held_albani.json", encoding="utf-8")) if (plan / "held_albani.json").exists() else {}
    held = {b["book_id"] for b in _held.get("books", [])}
    albani_resolved = bool(_held.get("resolved"))
    albani_verdicts = {v["book_id"]: v["evidence"] for v in json.load(open(audit / "albani_verdicts.json", encoding="utf-8"))["verdicts"]} \
        if (audit / "albani_verdicts.json").exists() else {}
    authors = {r["author_id"]: r for r in jsonl(RECIPE_BOOKS / "authors.jsonl")}
    renamed = {old: r["author_id"] for r in authors.values() for old in r["previous_ids"]}
    by_alias = {a: r["author_id"] for r in authors.values() for a in r["aliases_ar"]}
    by_shamela = {r["external_ids"]["shamela"]: r["author_id"] for r in authors.values() if "shamela" in r["external_ids"]}


AUTHOR_SUFFIXES = ("الألباني", "العثيمين", "ابن عثيمين", "ابن باز", "السعيدان", "ابن القيم", "ابن تيمية", "شيخ الإسلام ابن تيمية")
PUBLISHER_WORDS = ("عالم الفوائد", "العلمية", "الفكر", "دار ", "مكتبة", "المكتبة", "المكبتة", "الآثار", "غراس", "وهبة",
                   "القاسم", "ابن كثير", "ابن الجوزي", "الرسالة", "المعارف")
DROP_LABELS = ("نسخة مفهرسة", "مجمع")


def clean_title(raw: str | None, author_ar: str | None) -> tuple[str | None, str | None]:
    """(title, edition) from a raw catalog title; edition holds the stripped debris."""
    if not raw:
        return raw, None
    edition: list[str] = []

    def strip_debris(t: str) -> str:
        t = re.sub(r"\s+", " ", t.replace("_", " ")).strip(" -")
        changed = True
        while changed:
            changed = False
            for pat in (r"\(([^()]*(?:ط\.?\s|دار |مكتبة|المكتبة|المكبتة|نسخة|قطعة|طبعت)[^()]*)\)\s*$",  # parenthetical edition
                        r"\s+-?\s*(ط[- ]?أخرى)\s*$",                                    # ط أخرى
                        r"(?:\s+-)?\s+((?:ت|تحقيق)\s+\S.*?)$",                         # editor: "ت النشيري ..."
                        r"(?:\s+-)?\s+(ط\.?\s+\S.*?)$",                                # "ط عالم الفوائد", "- ط وهبة"
                        r"\s+-\s*(قطعة منه)\s*$"):
                m = re.search(pat, t)
                if m:
                    edition.append(m.group(1).strip())
                    t = t[:m.start()].strip(" -")
                    changed = True
            m = re.search(r"\s*-\s*([^-]+)$", t)
            if m:
                tail = m.group(1).strip()
                if (tail in DROP_LABELS or tail in AUTHOR_SUFFIXES or (author_ar and tail and tail in author_ar)
                        or any(tail.startswith(w) or tail == w.strip() for w in PUBLISHER_WORDS)):
                    if tail not in DROP_LABELS and tail not in AUTHOR_SUFFIXES and not (author_ar and tail in author_ar):
                        edition.append(tail)
                    t = t[:m.start()].strip(" -")
                    changed = True
        return t

    parts = [re.sub(r"\s*\(م\)\s*$", "", strip_debris(p)) for p in raw.split("::")]  # OpenITI "(م)" marker
    parts = [p for p in parts if p and not p.startswith("إهداء")]
    title = max(parts, key=len) if parts else raw.strip()
    seen: list[str] = []
    for e in edition:
        if e not in seen:
            seen.append(e)
    return title, "، ".join(seen) or None


def previous(name):
    path = OUT / name
    if not path.exists():
        return {}
    return {r["id"]: r for r in json.load(open(path, encoding="utf-8"))["items"]}


# Books whose source metadata names no author, or the wrong one.
BOOK_AUTHORS = {
    "tafsir_arabic::tanwir_al_miqbas": "ibn_ashur_1296_1393",
    "book:hadith:unknown_author:bc29bf92d78c": "al_albani_1332_1420",  # تراجعات الألباني: his words, compiled by others
    "archive:quran_irab_bahjat_salih": "bahjat_abd_al_wahid_salih",
    "archive:saaid_manzuma_sunniya_doc": "walid_alsaeedan",
    # Compiled from the scholar's own books; the compiler is a contributor.
    "book:fiqh:shamela_author_2315:7dc75a54be3b": "ibn_uthaymeen_1347_1421",   # from الشرح الممتع
    "book:fiqh:shamela_author_2315:ce36c4c25178": "ibn_uthaymeen_1347_1421",   # from الأصول من علم الأصول
    "book:hadith:shamela_author_2315:4971e14222ef": "ibn_uthaymeen_1347_1421",  # from مصطلح الحديث
    "book:fiqh:shamela_author_2510:17571dc33c6a": "al_albani_1332_1420",       # from al-Albani's books
}
TITLES = {"tafsir_arabic::tanwir_al_miqbas": "التحرير والتنوير"}
# Contributors the metadata names only inside the title (no "role: name" line).
KHANFAR = [{"name_ar": "حازم خنفر", "role": "جمع وإعداد", "author_id": None}]
EXTRA_CONTRIBUTORS = {
    "book:hadith:shamela_author_230:3a61afdd6a00": [
        {"name_ar": "محمد ناصر الدين الألباني", "role": "تخريج (ظلال الجنة)", "author_id": "al_albani_1332_1420"}],
    "book:fiqh:shamela_author_2315:7dc75a54be3b": KHANFAR,
    "book:fiqh:shamela_author_2315:ce36c4c25178": KHANFAR,
    "book:hadith:shamela_author_2315:4971e14222ef": KHANFAR,
    "book:fiqh:shamela_author_2510:17571dc33c6a": [
        {"name_ar": "أبو مالك محمد بن حامد بن عبد الوهاب", "role": "استخلاص", "author_id": None}],
}
CONTRIBUTOR_ROLES = r"(المحقق|تحقيق|حققه|تخريج|شرح وتعليق|تعليق|اعتنى به|إعداد|ضبط)"


def author_of(row):
    aid = BOOK_AUTHORS.get(row["id"], renamed.get(row.get("author_id"), row.get("author_id")))
    if aid is not None and aid not in authors:
        raise SystemExit(f"{row['id']}: author {aid} is not in authors.jsonl")
    return aid


def contributors(row):
    """Co-authors and editors/graders from a Shamela book's own metadata."""
    if row.get("source") != "shamela4" or not row.get("source_path"):
        return []
    meta = DATA / Path(row["source_path"]).parent / "book_metadata.json"
    if not meta.exists():
        return []
    m = json.load(open(meta, encoding="utf-8"))
    out = [{"name_ar": a["name_ar"], "role": "coauthor", "author_id": by_shamela.get(str(a["author_id"]))}
           for a in m["authors"] if a["role"] != "author"]
    for line in re.split(r"[\r\n]+", m.get("betaka_text") or ""):
        hit = re.match(rf"\s*{CONTRIBUTOR_ROLES}[^:]*:\s*(.+)", line)
        if hit:
            name = re.sub(r"\s*[\[(].*$", "", hit.group(2)).strip()
            out.append({"name_ar": name, "role": hit.group(1), "author_id": by_alias.get(name)})
    return out


def base_row(book_id):
    lib = library.get(book_id, {})
    cov = coverage.get(book_id, {})
    q = quality.get(book_id, {})
    return {
        "id": book_id,
        "title_ar": lib.get("title_ar") or q.get("title") or cov.get("title"),
        "author_ar": lib.get("author_name_ar") or q.get("author") or cov.get("author"),
        "author_id": lib.get("author_id"),
        "category": lib.get("category") or q.get("category"),
        "subcategory": lib.get("subcategory") or q.get("subcategory"),
        "book_type": lib.get("book_type"),
        "source": lib.get("source_corpus") or q.get("source_corpus"),
        "file_format": lib.get("file_format") or q.get("file_format"),
        "source_url": lib.get("source_url"),
        "source_path": lib.get("source_path"),
        "views_in_db": cov.get("views_in_db"),
        "parents_in_db": cov.get("parents_in_db"),
    }


def with_enrichment(row, old):
    prev = old.get(row["id"], {})
    row["title_raw"] = row["title_ar"]
    row["title_ar"], row["edition"] = clean_title(row["title_ar"], row.get("author_ar"))
    row["title_ar"] = TITLES.get(row["id"], row["title_ar"])
    row["author_id"] = author_of(row)
    reg = authors.get(row["author_id"], {})
    row["author_ar"] = reg.get("name_ar_short")
    row["author_died_ah"] = reg.get("death_hijri")
    row["contributors"] = contributors(row) + EXTRA_CONTRIBUTORS.get(row["id"], [])
    for key in ENRICHED:
        row[key] = prev.get(key, prev.get("notes_en"))
    return row


KEY_ORDER = ("id", "title_ar", "edition", "author_ar", "author_died_ah", "category", "subcategory", "book_type",
             "source", "file_format", "views_in_db", "parents_in_db", "quality", "quality_note", "decision",
             "decision_note", "duplicate_cluster", "duplicate_reason", "status", "damage", "notes", "title_raw", "author_id",
             "contributors")
MISLABELS = {"tafsir_arabic::tanwir_al_miqbas": "كان مسمى خطأ «تنوير المقباس» للكلبي؛ النص هو التحرير والتنوير لابن عاشور. "
                                                "معرّف الكتاب في قاعدة البحث وعنوانه فيها ما زالا بالاسم القديم حتى يعاد تضمينه"}


def write(name, description, items):
    for r in items:
        if r["id"] in MISLABELS:
            r["notes"] = MISLABELS[r["id"]]
    items = [{**{k: r[k] for k in KEY_ORDER if k in r}, **{k: v for k, v in r.items() if k not in KEY_ORDER}} for r in items]
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump({"description": description, "count": len(items), "items": items}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{name}: {len(items)}")


# ── books.json: clean books in the DB ────────────────────────────────────────
def write_books() -> None:
    old = previous("books.json")
    books = []
    for book_id, q in sorted(quality.items(), key=lambda kv: (kv[1].get("category") or "", kv[0])):
        if q["verdict"] not in ("clean", "minor issues", "minor"):
            continue
        row = with_enrichment(base_row(book_id), old)
        row["quality"] = "clean" if q["verdict"] == "clean" else "minor_issues"
        row["quality_note"] = None if q["verdict"] == "clean" else q.get("reason")
        action = actions.get(book_id, {}).get("action")
        cl = member_of.get(book_id)
        row["duplicate_cluster"] = cl["id"] if cl else None
        if book_id in planned_drops:
            row["decision"] = "drop"
        elif not albani_resolved and (book_id in held or action in ("drop_duplicate", "review_duplicate")):
            row["decision"] = "held_albani_check" if "al_albani" in book_id or book_id in held else "review_duplicate"
        else:
            row["decision"] = "keep"
            if action in ("drop_duplicate", "review_duplicate") and albani_verdicts.get(book_id):
                row["decision_note"] = "kept after the entry-level al-Albani grading check: " + albani_verdicts[book_id][:200]
        row["duplicate_reason"] = actions.get(book_id, {}).get("reason") if action else None
        books.append(row)
    write("books.json", "Clean books in the search DB. decision: keep / drop (approved, see removal_plan/) / "
          "held_albani_check (al-Albani books waiting for the grading check).", books)


PDF_ONLY = [
    ("saaid_pdf:76", "متن المنظومة الميمية"),
    ("saaid_pdf:77", "شرح المنظومة الميمية"),
    ("saaid_pdf:78", "متن المنظومة السنية"),
    ("saaid_pdf:79", "شرح المنظومة السنية"),
    ("saaid_pdf:86", "قواعد تعظيم الأدلة وحراستها"),
    ("saaid_pdf:87", "تفصيل التأصيل ج1"),
    ("saaid_pdf:88", "تفصيل التأصيل ج2"),
    ("saaid_pdf:103", "تربية الملكة الفقهية"),
    ("saaid_pdf:104", "خلاصة القواعد الأصولية"),
    ("saaid_pdf:114", "قواعد أصول الفقه من خلال متن الورقات"),
]


# ── needs_replacement.json ───────────────────────────────────────────────────
def write_needs_replacement() -> None:
    old = previous("needs_replacement.json")
    needs = []
    for book_id, q in sorted(quality.items()):
        if q["verdict"] != "damaged":
            continue
        row = with_enrichment(base_row(book_id), old)
        cl = member_of.get(book_id)
        clean_alternatives = []
        if cl:
            clean_alternatives = [m["id"] for m in cl["members"] + [{"id": cl["keeper"]}]
                                  if m["id"] != book_id and quality.get(m["id"], {}).get("verdict") != "damaged"]
        row.update({
            "status": "in_db_damaged",
            "damage": q.get("reason"),
            "excerpts": (q.get("excerpts") or [])[:2],
            "clean_edition_in_db": sorted(set(clean_alternatives)),
            "replacement_candidates": old.get(book_id, {}).get("replacement_candidates", []),
        })
        needs.append(row)

    for pid, title in PDF_ONLY:
        prev = old.get(pid, {})
        row = {"id": pid, "title_ar": title, "author_ar": "وليد بن راشد السعيدان", "author_id": "walid_alsaeedan",
               "category": "fiqh_usul" if "أصول" in title or "تأصيل" in title or "قواعد" in title else "fiqh",
               "source": "saaid_pdf", "file_format": "pdf_text",
               "source_path": "archive/corpus_source_archive/*/walid_alsaeedan_taseel (PDF copies)",
               "views_in_db": 0, "parents_in_db": 0}
        row = with_enrichment(row, old)
        row.update({
            "status": "not_in_db_pdf_only",
            "damage": "only copy is PDF-extracted text (ligature/font-encoding damage); never indexed",
            "excerpts": [],
            "clean_edition_in_db": [],
            "replacement_candidates": prev.get("replacement_candidates", []),
        })
        if pid == "saaid_pdf:78":
            row["notes"] = row["notes"] or "A clean .doc of the matn exists locally; see not_embedded.json."
        needs.append(row)
    write("needs_replacement.json", "Damaged books in the DB and works we only have as broken PDF text. "
          "replacement_candidates are clean online sources found for them.", needs)


ARCHIVE_GOOD = [
    ("archive:quran_irab_bahjat_salih", "الإعراب المفصل لكتاب الله المرتل", "بهجت عبد الواحد صالح",
     "archive/corpus_source_archive/native_online_candidates/quran_irab_full/"
     "150578__الإعراب_المفصل_لكتاب_الله_المرتل__بهجت_عبد_الواحد_صالح"),
    ("archive:quran_irab_dar_al_sahaba", "إعراب القرآن الكريم", "دار الصحابة",
     "archive/corpus_source_archive/native_online_candidates/quran_irab_full/86__إعراب_القرآن_الكريم__دار_الصحابة"),
    ("archive:saaid_manzuma_sunniya_doc", "متن المنظومة السنية", "وليد بن راشد السعيدان",
     "archive/corpus_source_archive/native_online_candidates/walid_alsaeedan_taseel/extracted/79_alt_saaid_manzuma_sunniya"),
]


# ── not_embedded.json ────────────────────────────────────────────────────────
def write_not_embedded() -> None:
    old = previous("not_embedded.json")
    items = []
    for book_id, cov in sorted(coverage.items()):
        if cov.get("views_in_db") or not cov.get("in_catalog"):
            continue  # indexed, or a candidate id that never became a catalogued book
        row = with_enrichment(base_row(book_id), old)
        row["status"] = cov.get("status")
        row["what_exists"] = cov.get("not_indexed_reason") or cov.get("status")
        row["views_extracted"] = cov.get("views_extracted")
        items.append(row)
    for aid, title, author, path in ARCHIVE_GOOD:
        row = with_enrichment({"id": aid, "title_ar": title, "author_ar": author, "author_id": None,
                               "category": "tafsir" if "إعراب" in title else "fiqh_usul",
                               "source": "archive", "file_format": "txt/pages.json" if "إعراب" in title else "doc",
                               "source_path": path, "views_in_db": 0, "parents_in_db": 0}, old)
        row["status"] = "clean_source_not_extracted"
        row["what_exists"] = "clean digital text in the archive; never parsed, chunked or embedded"
        row["views_extracted"] = 0
        items.append(row)
    write("not_embedded.json", "Clean material we have that is not in the search DB.", items)


def main(argv: list[str] | None = None) -> int:
    global DATA, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="data directory (catalog/audit, catalog/library)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "catalog", help="where the catalog files are read and written")
    args = ap.parse_args(argv)
    DATA, OUT = args.data, args.out
    load_inputs()
    write_books()
    write_needs_replacement()
    write_not_embedded()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
