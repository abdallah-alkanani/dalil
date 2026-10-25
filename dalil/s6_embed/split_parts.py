"""Merge the corpora's to_embed files into N balanced parts, one per GPU.

A text needed by several corpora is embedded once (dalil.s6_embed.reuse assemble
finds it by text). Views are dealt longest first, each to the part with the
fewest words so far.

    python -m dalil.s6_embed.split_parts TO_EMBED_DIR OUT_DIR N
"""
from __future__ import annotations

import argparse
import heapq
import json
from contextlib import ExitStack
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="directory of *.jsonl to_embed files (dalil.s6_embed.reuse plan)")
    ap.add_argument("out", type=Path, help="writes part_0.jsonl ... part_{N-1}.jsonl here")
    ap.add_argument("n", type=int, help="number of parts")
    args = ap.parse_args(argv)
    src, out, n = args.src, args.out, args.n
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for f in sorted(src.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    text = json.loads(line)["embedding_text"]
                    if text not in seen:
                        seen.add(text)
                        rows.append((len(text.split()), line if line.endswith("\n") else line + "\n"))
    rows.sort(key=lambda r: -r[0])
    heap = [(0, i) for i in range(n)]
    counts = [0] * n
    with ExitStack() as stack:
        files = [stack.enter_context(open(out / f"part_{i}.jsonl", "w", encoding="utf-8")) for i in range(n)]
        for words, line in rows:
            load, i = heapq.heappop(heap)
            files[i].write(line)
            counts[i] += 1
            heapq.heappush(heap, (load + words, i))
    print(json.dumps({"texts": len(rows), "parts": counts, "words": sorted(h[0] for h in heap)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
