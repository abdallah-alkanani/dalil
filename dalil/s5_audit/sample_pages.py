"""Sample random 'pages' from random books out of the structured corpus.

Two passes: one to learn the byte offset of every parent row per book, then a
second that seeks directly to the chosen rows. That keeps memory flat over a
multi-GB parents.jsonl instead of holding the corpus in RAM.

The pages feed ``dalil.s5_audit.text_quality``:

    python -m dalil.s5_audit.sample_pages \\
        data/build/corpora/collection/parents.jsonl \\
        --books 5 --pages-per-book 3 --out pages.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parents")
    ap.add_argument("--books", type=int, default=25, help="how many random books")
    ap.add_argument("--pages-per-book", type=int, default=1)
    ap.add_argument("--chars", type=int, default=4000, help="target page size")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--only-books", default="", help="comma-separated book_ids to target")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    offsets = collections.defaultdict(list)
    with open(a.parents, "rb") as f:
        pos = f.tell()
        for line in f:
            if line.strip():
                bid = json.loads(line)["book_id"]
                offsets[bid].append(pos)
            pos = f.tell()

    only = {b for b in a.only_books.split(",") if b}
    books = sorted(only) if only else rng.sample(sorted(offsets), min(a.books, len(offsets)))

    samples = []
    with open(a.parents, "rb") as f:
        for bid in books:
            rows = offsets[bid]
            for _ in range(a.pages_per_book):
                start = rng.randrange(len(rows))
                text, used, title = [], 0, None
                for off in rows[start:start + 40]:
                    f.seek(off)
                    r = json.loads(f.readline())
                    if title is None:
                        title = " > ".join(r.get("section_path") or [])
                    t = r.get("text_for_synthesis") or ""
                    text.append(t)
                    used += len(t)
                    if used >= a.chars:
                        break
                # Cut on a whitespace boundary. A hard character cut makes every
                # sample look like it "ends mid-word", which the reader then reports
                # as corpus truncation when it is only an artifact of sampling.
                joined = "\n\n".join(text)
                if len(joined) > a.chars:
                    cut = joined.rfind(" ", 0, a.chars)
                    joined = joined[: cut if cut > a.chars // 2 else a.chars]
                samples.append({
                    "book_id": bid,
                    "section": title,
                    "parent_index": start,
                    "n_parents": len(rows),
                    "excerpt": True,
                    "text": joined,
                })

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    print(f"books_in_corpus={len(offsets)} sampled_books={len(books)} samples={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
