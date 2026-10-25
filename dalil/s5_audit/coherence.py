"""Check whether an argument spans several parents — and whether the later ones
are still findable on their own.

The failure this looks for is context fragmentation. An author opens a point in
parent N ("مسألة: حكم كذا"), reasons through parent N+1, and concludes in parent
N+2 — but never restates the subject. Search the topic and you retrieve parent N,
which states the question and not the answer. Parent N+2 holds the answer and is
unreachable, because none of the words a user would search appear in it.

Chunk-level quality checks cannot see this: each parent looks fine in isolation.
It only shows up when consecutive parents are read together, which is what this
does — it hands DeepSeek a window of N consecutive parents (ordered by
`order_in_book`) plus a sample of their child views, and asks whether the window
is one argument and whether each parent stands alone.

Findings feed two decisions:
  - a high spanning rate means the parent policy for that book cuts too finely
  - it also tells the query layer how far to expand around a hit (its neighbour
    expansion)

    python -m dalil.s5_audit.coherence \\
        data/build/corpora/collection/parents.jsonl \\
        --out coherence.json --window 3 --samples 20
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dalil.llm import TRANSPORTS, LLMError, get_transport

PROMPT = """You are auditing retrieval chunking for an Arabic Islamic corpus.

Below are %(n)d CONSECUTIVE parents from one book, in reading order. Each is a
separate retrieval unit: a search can return one without the others.

Judge ONE thing: does the author's argument run ACROSS these parents in a way
that makes a later parent impossible to find on its own?

Concretely — if the topic is named only in parent 1, and parent 3 carries the
ruling or conclusion without naming the topic, then someone searching that topic
retrieves parent 1 and never reaches parent 3. That is the failure.

Answer in STRICT JSON, no markdown fence:
{"spans_parents": true|false,
 "self_contained": [true|false per parent, in order],
 "unfindable_parents": [1-based indices that cannot be retrieved on their own],
 "topic_carrying_parent": <1-based index that names the subject, or null>,
 "recommended_window": <how many neighbours a retrieval hit should pull in, 0-3>,
 "reason": "<= 200 chars"}

BOOK: %(book)s

%(blocks)s"""


def load_parents(path: Path, book_filter: set[str] | None):
    by_book: dict[str, list[dict]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if book_filter and r["book_id"] not in book_filter:
                continue
            by_book[r["book_id"]].append(r)
    return by_book


def ask(window, book, transport, model, timeout, chars):
    blocks = []
    for i, p in enumerate(window, 1):
        t = " ".join((p.get("text_for_synthesis") or "").split())[:chars]
        blocks.append(f"--- PARENT {i} ({p.get('parent_type')}) ---\n{t}")
    prompt = PROMPT % {"n": len(window), "book": book, "blocks": "\n\n".join(blocks)}
    try:
        d = transport.ask_json(prompt, model=model, timeout=timeout)
    except LLMError as exc:
        return {"error": str(exc)[:160], "book": book}
    d["book"] = book
    d["parent_ids"] = [p["parent_id"] for p in window]
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parents", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=3, help="consecutive parents per check")
    ap.add_argument("--samples", type=int, default=20, help="windows to audit")
    ap.add_argument("--chars", type=int, default=1100, help="chars per parent in the prompt")
    ap.add_argument("--books", default="", help="comma-separated book_ids; default all")
    ap.add_argument("--transport", default="openrouter", choices=TRANSPORTS,
                    help="which path to the model")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the transport's DEFAULT_MODEL")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    transport = get_transport(args.transport)
    model = args.model or transport.DEFAULT_MODEL

    flt = {b for b in args.books.split(",") if b}
    by_book = load_parents(args.parents, flt or None)
    if not by_book:
        raise SystemExit("no parents matched")

    rng = random.Random(args.seed)
    windows = []
    for book, ps in by_book.items():
        # reading order: order_in_book when present, else file order
        ps.sort(key=lambda r: r.get("order_in_book") or 0)
        if len(ps) < args.window:
            continue
        for _ in range(max(1, args.samples // max(1, len(by_book)))):
            i = rng.randrange(len(ps) - args.window + 1)
            windows.append((ps[i:i + args.window], book))
    windows = windows[: args.samples]
    print(f"auditing {len(windows)} windows of {args.window} consecutive parents "
          f"across {len(by_book)} books", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, r in enumerate(ex.map(
                lambda w: ask(w[0], w[1], transport, model, args.timeout, args.chars), windows), 1):
            results.append(r)
            if "error" in r:
                print(f"  [{i}/{len(windows)}] {r['error']}", flush=True)
            else:
                print(f"  [{i}/{len(windows)}] spans={str(r.get('spans_parents')):5} "
                      f"unfindable={r.get('unfindable_parents')} "
                      f"window={r.get('recommended_window')}  {r['book'][-26:]}", flush=True)

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    good = [r for r in results if "error" not in r]
    spanning = [r for r in good if r.get("spans_parents")]
    unfind = sum(len(r.get("unfindable_parents") or []) for r in good)
    print(f"\nwindows audited     : {len(good)}")
    print(f"argument spans parents: {len(spanning)} ({100*len(spanning)/max(1,len(good)):.0f}%)")
    print(f"parents not findable alone: {unfind}")
    wins = [r.get("recommended_window") for r in good if isinstance(r.get("recommended_window"), int)]
    if wins:
        print(f"recommended neighbour window: median {sorted(wins)[len(wins)//2]}, max {max(wins)}")
    per = collections.Counter(r["book"] for r in spanning)
    if per:
        print("\nbooks whose parents fragment most:")
        for b, n in per.most_common(5):
            print(f"   {n:>3} windows  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
