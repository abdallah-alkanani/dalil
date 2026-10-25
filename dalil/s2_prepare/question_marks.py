"""Put back what Ketabonline 92317 (إتحاف أهل الألباب) lost to '?'.

The package was converted from a Word file whose Quran brackets and honorifics
were glyphs of a symbol font, and each became a plain '?' (2,155 of them). Two
passes, each from evidence rather than guesswork:

1. Honorifics. The saaid.net copy of the same book (a .doc converted to text, in
   books/original/ beside it) has the honorifics as words but lost its Quran text. Each '?' is
   located there by the words around it; when the words between are an
   honorific, that honorific replaces it.
2. Quran. Of the '?' left, each consecutive pair whose text between is Quran
   becomes ﴿ … ﴾: at least 80% of its three-word windows (the whole text, if
   shorter) occur in the Quran's consonant skeleton (`skel`, with doubled
   letters squeezed, as Uthmani writes الليل ٱلَّيۡل), so quotes that join two
   verses or skip words still count. The gap between two quotes ("، وقال تعالى:")
   is never a quote. After "تعالى:", "قوله:" or "سورة …:" the book itself says a
   quote follows, so there one word, or one matching window in a misquoted or
   misspelt verse (جنهم for جهنم), is enough.
3. Right after a word that can only mean the Prophet (النبي, رسول الله, ...), or
   right before his words ("قوله ?: ((", "عنه ? أنه قال: (("), a '?' is
   صلى الله عليه وسلم, as in every such case pass 1 resolved.

Page text and table-of-contents titles (the question headings) are both repaired;
a Quran pair never spans two titles. A '?' none of these explains is left as it is
and listed in the report. The original
download is not touched: the repaired package goes to the book's folder under
build/books/, which dalil.s2_prepare.ketabonline then reads, and the counts to
question_marks_92317.json beside it.

    python -m dalil.s2_prepare.question_marks [--original DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from dalil.paths import BUILD_BOOKS_DIR, DATA_DIR, ORIGINAL_DIR, RECIPE_BOOKS

QURAN = ORIGINAL_DIR / "القرآن الكريم" / "quran-ar.json"
PACKAGE = "92317.data.zip"
OLD_BOOK = "saaid:14006::walid_alsaeedan::إتحاف_أهل_الألباب_بمعرفة_التوحيد_والعقيدة_في_سؤال_وجواب"
HONORIFICS = ("صلى الله عليه وسلم", "عليه الصلاة والسلام", "عليه السلام", "عليهم السلام",
              "رضي الله عنه", "رضي الله عنها", "رضي الله عنهما", "رضي الله عنهم", "رضي الله عنهن",
              "رحمه الله", "رحمها الله", "رحمهم الله", "رحمه الله تعالى", "رحمهم الله تعالى",
              "عز وجل", "جل وعلا", "جل جلاله", "سبحانه وتعالى", "تبارك وتعالى", "سبحانه")
CONTEXT_WORDS = 4
MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
SKEL_DROP = re.compile(r"[اأإآٱءؤئوىي]|[^ء-ي]")


def skel(text: str) -> str:
    """Consonant skeleton: no marks, no alif/hamza/waw/ya, no spaces (survives Uthmani vs imla'i spelling)."""
    return SKEL_DROP.sub("", MARKS.sub("", text).replace("ة", "ه"))


def quran_skeleton() -> str:
    surahs = json.loads(QURAN.read_text(encoding="utf-8"))
    return "|".join("".join(skel(v["text"]) for v in s["verses"]) for s in surahs)


def words(text: str) -> list[str]:
    text = MARKS.sub("", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    return re.findall(r"[ء-ي]+", text)


PROPHET = re.compile(r"(?:النبي|للنبي|رسول الله|لرسول الله|برسول الله|الرسول|نبيه|رسوله|محمد|المصدوق|الحبيب)\s*$")
HIS_WORDS = re.compile(r"\s*:?\s*(?:أنه قال\s*:?\s*)?\(\(")
INTRODUCED = re.compile(r"(?:تعالى|قوله|سورة|الآية)[^.:،؟?]{0,30}:?\s*$")


def squeeze(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def is_quran(span: str, qskel: str, introduced: bool) -> bool:
    stripped = span.strip()
    if "((" in span or stripped[:1] in "،," or stripped[-1:] == ":":
        return False
    if len(skel(span)) < (2 if introduced else 5):
        return False
    w = words(span)
    windows = [squeeze(skel(" ".join(w[i:i + 3]))) for i in range(max(1, len(w) - 2))]
    hits = sum(x in qskel for x in windows)
    return hits >= 1 if introduced else hits >= 0.8 * len(windows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path, default=ORIGINAL_DIR, help="the downloads (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=BUILD_BOOKS_DIR, help="the repaired copies; the copy goes to the same <author>/<book>/ (default: %(default)s)")
    args = ap.parse_args(argv)
    src = next(args.original.rglob(PACKAGE))
    dst = args.out / src.relative_to(args.original)
    with zipfile.ZipFile(src) as z:
        member = next(n for n in z.namelist() if n.endswith(".json"))
        data = json.loads(z.read(member))

    # One string for the whole book (pages, then index titles, each title after a
    # NUL so no pair crosses it), tags blanked to spaces so offsets stay valid.
    slots = [(p, "content") for p in data["pages"]] + [(t, "title") for t in data["index"]]
    contents = [str(row.get(key) or "") for row, key in slots]
    starts, pos = [], 0
    for c in contents:
        starts.append(pos)
        pos += len(c) + 1
    book = "".join(("\0" if key == "title" else "\n") * bool(i) + re.sub(r"<[^>]*>", lambda m: " " * len(m.group()), c)
                   for i, ((_, key), c) in enumerate(zip(slots, contents)))
    marks = [m.start() for m in re.finditer(r"\?", book)]
    fill: dict[int, str] = {}

    # The saaid.net copy's source file: the same words, in the same order, as the
    # parents the August stores built from it (checked 2026-10-17: 175,256 words).
    old_row = next(r for r in map(json.loads, RECIPE_BOOKS.joinpath("core_profiles.jsonl").open(encoding="utf-8"))
                   if r["book_id"] == OLD_BOOK)
    old_text = (DATA_DIR / old_row["source_paths"][0]).read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    old = " ".join(words(old_text))
    honorific = {" ".join(words(h)): h for h in HONORIFICS}
    found: Counter = Counter()
    unresolved = []
    for k, at in enumerate(marks):
        before = book[marks[k - 1] + 1 if k else 0:at]
        after = book[at + 1:marks[k + 1] if k + 1 < len(marks) else len(book)]
        left, right = words(before)[-CONTEXT_WORDS:], words(after)[:CONTEXT_WORDS]
        fills = set()
        if len(left) + len(right) >= CONTEXT_WORDS:
            pattern = re.escape(" ".join(left)) + r" ?((?:[ء-ي]+ ?){0,4}?) ?" + re.escape(" ".join(right))
            fills = {m.group(1).strip() for m in re.finditer(pattern, old)}
        hits = {honorific[f] for f in fills if f in honorific}
        if len(hits) == 1:
            fill[at] = hits.pop()
            found[fill[at]] += 1

    qskel = squeeze(quran_skeleton())
    rest = [at for at in marks if at not in fill]
    i = 0
    while i + 1 < len(rest):
        a, b = rest[i], rest[i + 1]
        if b - a <= 1500 and "\0" not in book[a:b] and is_quran(book[a + 1:b], qskel, bool(INTRODUCED.search(book[max(0, a - 40):a]))):
            fill[a], fill[b] = "﴿", "﴾"
            i += 2
        else:
            i += 1

    for at in marks:
        if at not in fill and (PROPHET.search(book[max(0, at - 20):at]) or HIS_WORDS.match(book, at + 1)):
            fill[at] = "صلى الله عليه وسلم"
            found["صلى الله عليه وسلم (after a word for the Prophet)"] += 1
        elif at not in fill:
            unresolved.append(" ".join(book[max(0, at - 70):at + 50].replace("\0", "\n").split()))

    for (row, key), start, text in zip(slots, starts, contents):
        here = sorted((at - start, s) for at, s in fill.items() if start <= at < start + len(text))
        for off, s in reversed(here):
            if s not in "﴿﴾":
                s = ("" if off == 0 or text[off - 1].isspace() else " ") + s
                s += " " if text[off + 1:off + 2].isalpha() else ""
            text = text[:off] + s + text[off + 1:]
        if here:
            row[key] = text

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, json.dumps(data, ensure_ascii=False))
    report = {
        "question_marks": len(marks),
        "quran_brackets": sum(1 for s in fill.values() if s in "﴿﴾"),
        "honorifics": dict(found.most_common()),
        "unresolved": len(unresolved),
        "unresolved_examples": unresolved[:40],
    }
    (dst.parent / "question_marks_92317.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "unresolved_examples"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
