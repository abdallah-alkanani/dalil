"""Sample parents a parser fix was supposed to repair and ask DeepSeek if it did.

A fix that makes the numbers move is not yet a fix that works. When the OpenITI
heading matcher started recognising `###|` headings, book-sized parents stopped
appearing — but that only proves the parser now cuts *somewhere*. Whether it cuts
in the right places is a question about the text, so it goes to a reader.

The reader is shown one parent together with the heading the parser assigned it,
and asked a structural question, not a quality one: is this exactly one complete
unit of the declared kind, does the heading belong to this body, does it start and
end where such a unit should? A wrong-but-clean parse — two chapters fused under
one heading, an answer separated from its question — reads as perfectly good
Arabic, so the usual text-quality audit cannot see it.

Sample by whatever the fix touched (`--parser-profile openiti_txt_parser`,
`--parent-type qanda`, `--flag takhrij_group`) and compare the verdict rate
against a run over parents the fix did not touch.

    python -m dalil.s5_audit.parse_check \\
        --parents data/build/corpora/collection/parents.jsonl \\
        --parser-profile openiti_txt_parser --out parse_check.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dalil.llm import TRANSPORTS, LLMError, get_transport

PROMPT = """You are checking whether a parser correctly divided a classical Arabic
Islamic book into retrieval units. Judge the STRUCTURE of the unit below, not its
religious content and not its spelling.

The parser claims this is ONE unit of type: %(ptype)s
The heading it assigned: %(heading)s

Answer these:
1. Does the HEADING actually belong to this BODY, or is it from a different section?
2. Is the body exactly ONE unit of that type, or were several separate units
   (several أبواب/فصول, several fatwas, several hadiths) fused into one?
3. Does the body start at a real beginning, or does it open in the middle of
   something whose start is missing?
4. Does it end at a real ending, or is the rest of it clearly missing?

If the type is qanda, the question and its own answer must be together.
If the type is hadith, the isnad and its matn must be together.

Return STRICT JSON only, no markdown fence:
{"verdict":"correct|fused|orphaned|wrong_heading|truncated",
 "reason":"<=140 chars, ENGLISH ONLY",
 "evidence":"<=120 chars quoted verbatim from the body"}

"correct" means the parser drew the right boundaries.
"fused" means several units were merged into one.
"orphaned" means it begins mid-unit with its opening missing.
"wrong_heading" means the heading belongs elsewhere.
"truncated" means it stops before the unit is finished.

BODY:
<<<
%(text)s
>>>"""


def check(row: dict, transport, model: str, timeout: int, max_words: int) -> dict:
    text = row["text_for_synthesis"]
    words = text.split()
    shown = text if len(words) <= max_words else (
        " ".join(words[: max_words // 2]) + "\n[…]\n" + " ".join(words[-max_words // 2:]))
    heading = (row.get("section_path") or ["(none)"])[-1]
    base = {"parent_id": row["parent_id"], "book_id": row["book_id"],
            "parent_type": row.get("parent_type"), "words": len(words),
            "parser_profile": row.get("parser_profile")}
    prompt = PROMPT % {"ptype": row.get("parent_type", "?"), "heading": heading, "text": shown}
    try:
        d = transport.ask_json(prompt, model=model, timeout=timeout)
    except LLMError as exc:
        return {**base, "verdict": "ERROR", "reason": str(exc)[:140], "evidence": ""}
    return {**base, "verdict": d.get("verdict", "ERROR"),
            "reason": (d.get("reason") or "")[:140],
            "evidence": (d.get("evidence") or "")[:120]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=30, help="parents to check")
    ap.add_argument("--parser-profile", help="only parents from this profile")
    ap.add_argument("--parent-type", help="only parents of this type")
    ap.add_argument("--flag", help="only parents carrying this quality_flag")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--max-words", type=int, default=2500,
                    help="longer bodies are shown head+tail only")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--transport", default="openrouter", choices=TRANSPORTS,
                    help="which path to the model")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the transport's DEFAULT_MODEL")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)
    transport = get_transport(args.transport)
    model = args.model or transport.DEFAULT_MODEL

    pool = []
    for line in args.parents.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if args.parser_profile and r.get("parser_profile") != args.parser_profile:
            continue
        if args.parent_type and r.get("parent_type") != args.parent_type:
            continue
        if args.flag and args.flag not in (r.get("quality_flags") or []):
            continue
        if len(r["text_for_synthesis"].split()) < args.min_words:
            continue
        pool.append(r)

    if not pool:
        print("no parents matched the filters", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(pool)
    rows = pool[: args.sample]
    print(f"{len(pool):,} parents matched; checking {len(rows)} "
          f"with {args.workers} workers", flush=True)

    done = [0]

    def work(r):
        v = check(r, transport, model, args.timeout, args.max_words)
        done[0] += 1
        print(f"  [{done[0]}/{len(rows)}] {v['verdict']:<13} {v['words']:>6,}w  "
              f"{v['reason'][:64]}", flush=True)
        return v

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, rows))

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    tally = collections.Counter(r["verdict"] for r in results)
    ok = tally.get("correct", 0)
    judged = sum(v for k, v in tally.items() if k != "ERROR")
    print(f"\nverdicts: {dict(tally)}")
    print(f"correct: {ok}/{judged} ({100 * ok / max(1, judged):.0f}% of judged)")
    for r in results:
        if r["verdict"] not in ("correct", "ERROR"):
            print(f"  ! {r['verdict']:<13} {r['parent_id'][-20:]}  {r['reason'][:80]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
