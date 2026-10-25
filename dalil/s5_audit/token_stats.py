"""Token-length distribution + HTML/markup artifact scan over retrieval views.

Parallel across all cores: the file is read in blocks and handed to a worker
pool, and each worker batch-encodes its block with the fast (Rust) tokenizer
rather than tokenizing one string at a time.

Needs ``transformers`` (imported in the workers, so ``--help`` works without it).

    python -m dalil.s5_audit.token_stats \\
        data/build/corpora/collection/retrieval_views.jsonl --limit 100000
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
from multiprocessing import Pool

MODEL = "Qwen/Qwen3-Embedding-8B"
REVISION = "4e423935c619ae4df87b646a3ce949610c66241c"

PATTERNS = {
    "html_tag":       r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]{0,200})?/?>",
    "html_entity":    r"&(?:[a-zA-Z]{2,10}|#\d{2,6}|#x[0-9a-fA-F]{2,6});",
    "angle_leftover": r"[<>]",
    "css_or_js":      r"(?:\{[^{}]*:[^{}]*;[^{}]*\}|function\s*\(|<script)",
    "url":            r"https?://\S+",
    "shamela_span":   r"data-type\s*=|class\s*=\s*[\"']",
    "replacement_ch": r"�",
    "pagev_marker":   r"PageV\d",
}

_tok = None
_compiled: dict[str, re.Pattern] = {}


def _init():
    global _tok, _compiled
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    _compiled = {k: re.compile(v) for k, v in PATTERNS.items()}


def _work(block: list[str]):
    texts, vtypes, books = [], [], []
    for line in block:
        r = json.loads(line)
        texts.append(r.get("embedding_text") or "")
        vtypes.append(r.get("view_type"))
        books.append(r.get("book_id"))
    enc = _tok(texts, add_special_tokens=False)["input_ids"]
    lens = [len(x) for x in enc]

    by_type = collections.defaultdict(list)
    for vt, n in zip(vtypes, lens):
        by_type[vt].append(n)
    hits = collections.Counter()
    hit_books = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for text, bid in zip(texts, books):
        for name, pat in _compiled.items():
            m = pat.search(text)
            if m:
                hits[name] += 1
                hit_books[name][bid] += 1
                if len(examples[name]) < 2:
                    s = max(0, m.start() - 60)
                    examples[name].append((bid, text[s:m.end() + 60].replace("\n", " ")))
    return lens, dict(by_type), hits, {k: dict(v) for k, v in hit_books.items()}, dict(examples)


def blocks(path: str, size: int, limit: int):
    buf, seen = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            buf.append(line)
            seen += 1
            if len(buf) >= size:
                yield buf
                buf = []
            if limit and seen >= limit:
                break
    if buf:
        yield buf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--block", type=int, default=5000)
    args = ap.parse_args(argv)
    # A worker whose initializer raises is replaced by another that raises, so a
    # missing transformers hangs the pool forever instead of failing. Check here.
    if importlib.util.find_spec("transformers") is None:
        ap.error("transformers is not installed in this environment")

    lens: list[int] = []
    by_type = collections.defaultdict(list)
    hits = collections.Counter()
    hit_books = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)

    with Pool(args.workers, initializer=_init) as pool:
        for l, bt, h, hb, ex in pool.imap_unordered(
                _work, blocks(args.views, args.block, args.limit)):
            lens.extend(l)
            for k, v in bt.items():
                by_type[k].extend(v)
            hits.update(h)
            for k, v in hb.items():
                hit_books[k].update(v)
            for k, v in ex.items():
                if len(examples[k]) < 3:
                    examples[k].extend(v)

    n = len(lens)
    print(f"=== TOKEN LENGTHS (Qwen3 tokenizer) over {n:,} views ===")
    if not n:
        # Percentiles, max and mean are undefined on nothing; say so rather than crash.
        print("  no views to measure")
        return 0
    lens.sort()
    def pct(p): return lens[min(n - 1, int(n * p / 100))]
    for p in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{p:<6} {pct(p):>7,}")
    print(f"  max     {lens[-1]:>7,}   mean {sum(lens)/n:.0f}")
    for cap in (512, 1024, 2048, 4096, 8192):
        over = sum(1 for x in lens if x > cap)
        print(f"  > {cap:<5}: {over:>8,} views ({100*over/n:6.3f}%)")

    print("\n=== BY VIEW TYPE (tokens) ===")
    for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        v.sort()
        print(f"  {str(k):22} n={len(v):>7,} p50={v[len(v)//2]:>5,} "
              f"p99={v[int(len(v)*0.99)]:>6,} max={v[-1]:>7,}")

    print(f"\n=== MARKUP / ARTIFACT SCAN over {n:,} views ===")
    for name in PATTERNS:
        c = hits[name]
        print(f"  {name:16} {c:>8,} views ({100*c/n:6.3f}%)  books={len(hit_books[name])}")
        for bid, cnt in hit_books[name].most_common(3):
            print(f"       {cnt:>7,}  {bid}")
        for bid, ex in examples[name][:2]:
            print(f"       eg: ...{ex[:150]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
