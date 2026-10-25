"""Decide how a book should be cut into retrieval parents, before extracting it.

Run this on a raw text file and it returns a `parent_policy` — the value that goes
into the book's parser profile and determines every parent boundary the extractor
will produce.

The judgement is the model's (DeepSeek by default); the cutting is not. This script
only recommends a policy, which deterministic code in `dalil.s4_extract` then
implements. That split keeps extraction reproducible and checkable while still
using a model for the one part that genuinely needs reading comprehension.

Why not just count markers? Because marker counting misses what it was not told to
look for. On the al-Saeedan books, counting `مسألة` / `س)` / `فصل` at line start
suggested one policy for three books; DeepSeek read the text and found `الفرع
الأول…` ordinals and `((كتاب الطهارة))` headings instead. One of those books would
otherwise have been emitted as a single 34,000-word parent.

The mechanical counts are still sent to the model — as evidence to verify, not as
the answer. The prompt states explicitly that the decision must come from the text.

Usage
-----
    python -m dalil.s3_structure.parent_policy --raw book.txt --title "..." --out policy.json
    python -m dalil.s3_structure.parent_policy --raw-dir books/raw/saaid/author/ --out policies.json
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

from dalil.llm import TRANSPORTS, LLMError, get_transport

POLICIES = {
    "heading_section_parent_v1": "short heading lines (باب/فصل/كتاب, possibly bracketed) delimit sections",
    "fatwa_masala_parent_v1": "each ruling opens with a marker such as 'مسألة'",
    "qanda_pair_parent_v1": "explicit question/answer pairs (س … ج …)",
    "numbered_item_parent_v1": "numbered items are the units (١- ٢- …, or 'الفرع الأول…')",
    "paragraph_group_parent_v1": "continuous prose with no reliable marker; group paragraphs",
    "monograph_prose_parent_v1": "one continuous argument with no internal divisions",
    "risala_collection_parent_v1": "a collection of separate short treatises",
}

# Mechanical evidence. Deliberately broad — the point is to hand the model things
# it might otherwise not think to look for, including markers we do not yet parse.
COUNTERS = {
    "line_start_masala": r"(?m)^\s*مسألة",
    "inline_masala": r"مسألة",
    "line_start_qanda": r"(?m)^\s*س\s*[0-9٠-٩]{0,4}\s*\)",
    "line_start_qaida": r"(?m)^\s*(?:ال)?قاعدة",
    "line_start_fasl_bab": r"(?m)^\s*(?:فصل|باب|كتاب|المبحث|تمهيد)",
    "bracketed_heading": r"\(\(\s*[^)]{3,60}\s*\)\)",
    "bracketed_structural": r"\(\(\s*[^)]{0,60}(?:كتاب|باب|فصل)[^)]{0,60}\)\)",
    "fara_ordinal": r"الفرع\s+ال[ء-ي]+",
    "numbered_line": r"(?m)^\s*[0-9٠-٩]{1,3}\s*[-)]",
}

PROMPT = """You are deciding how an Arabic Islamic book must be split into retrieval
"parents" (semantic units) for a production RAG system. This choice determines every
parent boundary, so it must be defensible from the text itself.

RULES (non-negotiable):
- Decide ONLY from the text windows below. Do NOT infer from the title or filename.
- The marker counts are mechanical evidence to VERIFY, not the answer. A marker can
  be frequent and still not be structural — e.g. '((...))' delimits headings in some
  books and quotes hadith in others, and 'مسألة' is often just an ordinary word.
- If the evidence does not support any policy, answer "needs_review" and say exactly
  what is missing. Never substitute a plausible guess.

Choose exactly one parent_policy:
%(policies)s

Return STRICT JSON only, no markdown fence:
{"parent_policy":"...","confidence":0.0-1.0,
 "evidence":"<quote the actual marker text you saw>",
 "canonical_parent_definition":"<one sentence: what exactly is one parent>",
 "audit_status":"approved_for_parser_implementation|needs_review",
 "risks":["..."]}

BOOK: %(title)s

MECHANICAL MARKER COUNTS (verify against the text; do not trust blindly):
%(stats)s

%(windows)s"""


def windows(text: str, n: int, chars: int) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    out = []
    for i in range(n):
        start = int(len(lines) * (i / max(1, n))) + (2 if i == 0 else 0)
        chunk = "\n".join(lines[start:start + 24])[:chars]
        out.append(f"WINDOW {i + 1} ({'opening' if i == 0 else f'{int(100*i/n)}% through'}):\n<<<\n{chunk}\n>>>")
    return "\n\n".join(out)


def audit_one(path: Path, title: str, transport: ModuleType, model: str, timeout: int,
              chars: int, n_windows: int, attempts: int = 3):
    """Ask the model, retrying on timeout or unparseable output.

    Long Arabic prompts make the endpoint slow and occasionally flaky, and a
    silent failure here would leave a book with no policy — which is worse than a
    slow answer. Each retry shortens the windows, since prompt length is the main
    driver of both timeouts and rambling non-JSON replies.
    """
    text = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
    stats = {k: len(re.findall(p, text)) for k, p in COUNTERS.items()}
    stats["lines"] = len([l for l in text.splitlines() if l.strip()])
    stats["words"] = len(text.split())

    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        # Shrink gently. Aggressive shrinking stops the timeouts but starves the
        # model of evidence: at 1200 chars/window it answered with 0.9 confidence,
        # at 800 the same books came back needs_review at 0.25.
        shrink = max(700, int(chars * (0.85 ** (attempt - 1))))
        prompt = PROMPT % {
            "policies": "\n".join(f"  - {k}: {v}" for k, v in POLICIES.items()),
            "title": title,
            "stats": json.dumps(stats, ensure_ascii=False),
            "windows": windows(text, n_windows, shrink),
        }
        try:
            d = transport.ask_json(prompt, model=model, timeout=timeout)
            break
        except LLMError as exc:
            last = f"{exc} (attempt {attempt}, {shrink} chars/window)"
    else:
        # Never guess a policy: an unaudited book must be visibly unaudited.
        return {"file": path.name, "title": title, "error": last,
                "parent_policy": None, "audit_status": "needs_review",
                "evidence": f"structure audit failed after {attempts} attempts: {last}",
                "stats": stats}
    if attempt > 1:
        d["risks"] = (d.get("risks") or []) + [f"answered on attempt {attempt}"]
    if d.get("parent_policy") not in POLICIES and d.get("audit_status") != "needs_review":
        d["risks"] = (d.get("risks") or []) + [f"unknown policy {d.get('parent_policy')!r}"]
        d["audit_status"] = "needs_review"
    d.update({"file": path.name, "title": title, "stats": stats})
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", type=Path, help="a single .txt book")
    src.add_argument("--raw-dir", type=Path, help="directory of .txt books")
    ap.add_argument("--title", default=None, help="title for --raw (defaults to filename)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transport", default="openrouter", choices=TRANSPORTS,
                    help="which path to the model (see dalil.llm)")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the transport's DEFAULT_MODEL")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--chars", type=int, default=1200,
                    help="chars per window; long Arabic prompts get slow, keep modest")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--attempts", type=int, default=3,
                    help="retries per book; each retry uses shorter windows")
    args = ap.parse_args(argv)
    transport = get_transport(args.transport)
    model = args.model or transport.DEFAULT_MODEL

    files = sorted(args.raw_dir.glob("*.txt")) if args.raw_dir else [args.raw]
    if not files:
        raise SystemExit("no .txt files found")
    jobs = [(f, args.title or f.stem) for f in files]
    print(f"auditing structure of {len(jobs)} book(s) with {model}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(
            lambda j: audit_one(j[0], j[1], transport, model, args.timeout, args.chars,
                                args.windows, args.attempts),
            jobs))

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'file':>14}  {'parent_policy':30} {'conf':>5}  status")
    for r in results:
        if "error" in r:
            print(f"{r['file'][:14]:>14}  {'ERROR: ' + r['error']:30} {'':>5}  -")
            continue
        print(f"{r['file'][:14]:>14}  {str(r.get('parent_policy'))[:30]:30} "
              f"{str(r.get('confidence'))[:5]:>5}  {r.get('audit_status')}")
    print(f"\nwrote {args.out}")
    print("Copy parent_policy into the book's line in "
          "metadata/structure_audit_*/recommended_parser_profiles_*.jsonl, then run "
          "dalil.s4_extract.")
    needs = [r for r in results if r.get("audit_status") == "needs_review"]
    if needs:
        print(f"\n{len(needs)} book(s) need review:")
        for r in needs:
            print(f"   {r['file']}: {str(r.get('evidence'))[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
