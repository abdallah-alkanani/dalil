"""Scan retrieval views for degenerate text: character-run garbage, Latin
contamination, digit floods. Aggregated per book so whole broken books surface.

    python -m dalil.s5_audit.junk_scan \\
        data/build/corpora/collection/retrieval_views.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
from multiprocessing import Pool

# 8+ of the SAME character in a row: 'mmmmmmmm', '99999999', '.........'
RUN_RE = re.compile(r"(.)\1{7,}")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
DIGIT_RUN_RE = re.compile(r"[0-9٠-٩]{12,}")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def work(block: list[str]):
    stats = collections.Counter()
    books = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list)
    for line in block:
        r = json.loads(line)
        t = r.get("display_text") or ""
        if not t:
            continue
        bid = r.get("book_id")
        n = len(t)
        arabic = len(ARABIC_RE.findall(t))
        latin = len(LATIN_RE.findall(t))
        stats["views"] += 1

        m = RUN_RE.search(t)
        if m:
            stats["char_run"] += 1; books[bid]["char_run"] += 1
            if len(ex["char_run"]) < 6:
                ex["char_run"].append((bid, t[max(0, m.start()-40):m.start()+90]))
        # Latin-dominant text (transliteration dumps, code, mojibake)
        if n >= 40 and latin > arabic and latin / n > 0.30:
            stats["latin_dominant"] += 1; books[bid]["latin_dominant"] += 1
            if len(ex["latin_dominant"]) < 6:
                ex["latin_dominant"].append((bid, t[:130]))
        # any substantial Latin words inside otherwise Arabic prose
        elif LATIN_WORD_RE.search(t):
            stats["latin_words"] += 1; books[bid]["latin_words"] += 1
            if len(ex["latin_words"]) < 6:
                mm = LATIN_WORD_RE.search(t)
                ex["latin_words"].append((bid, t[max(0, mm.start()-50):mm.start()+80]))
        m = DIGIT_RUN_RE.search(t)
        if m:
            stats["digit_flood"] += 1; books[bid]["digit_flood"] += 1
            if len(ex["digit_flood"]) < 6:
                ex["digit_flood"].append((bid, t[max(0, m.start()-40):m.start()+90]))
        # very low Arabic density in a long chunk
        if n >= 200 and arabic / n < 0.35:
            stats["low_arabic"] += 1; books[bid]["low_arabic"] += 1
            if len(ex["low_arabic"]) < 6:
                ex["low_arabic"].append((bid, t[:130]))
    return stats, {k: dict(v) for k, v in books.items()}, dict(ex)


def blocks(path, size):
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line)
            if len(buf) >= size:
                yield buf; buf = []
    if buf:
        yield buf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--block", type=int, default=8000)
    a = ap.parse_args(argv)

    stats = collections.Counter()
    books = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list)
    with Pool(a.workers) as pool:
        for s, b, e in pool.imap_unordered(work, blocks(a.views, a.block)):
            stats.update(s)
            for k, v in b.items():
                books[k].update(v)
            for k, v in e.items():
                if len(ex[k]) < 6:
                    ex[k].extend(v)

    n = stats["views"]
    print(f"=== JUNK SCAN over {n:,} views ===")
    for k in ("char_run", "latin_dominant", "latin_words", "digit_flood", "low_arabic"):
        # max(1, n): an empty or all-blank file reports 0% instead of dividing by zero
        print(f"  {k:16} {stats[k]:>8,} views ({100*stats[k]/max(1, n):6.3f}%)")
    print()
    for k in ("char_run", "latin_dominant", "latin_words", "digit_flood", "low_arabic"):
        worst = sorted(books.items(), key=lambda kv: -kv[1][k])[:5]
        worst = [(b, c[k]) for b, c in worst if c[k] > 0]
        if not worst:
            continue
        print(f"--- worst books for {k}")
        for b, c in worst:
            print(f"    {c:>7,}  {b}")
        for b, s in ex[k][:3]:
            print(f"      eg: {s[:120]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
