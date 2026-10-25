"""Fan out DeepSeek v4 Flash readers over sampled corpus pages.

Each reader gets one page of real Arabic text and reports, as strict JSON,
whether the text is usable for retrieval or damaged. Readers run concurrently
because each model call is an independent request.

Pages come from ``dalil.s5_audit.sample_pages``:

    python -m dalil.s5_audit.text_quality pages.json --out text_quality.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from dalil.llm import TRANSPORTS, LLMError, get_transport

PROMPT = """You are auditing text extracted from classical Arabic Islamic books for a retrieval system.

Read the PAGE below and judge ONLY the text quality. Do not judge the religious content, and do not summarise it.

Report these problems if present:
- markup_artifacts: leftover HTML/XML tags, entities like &nbsp;, CSS/JS, URLs
- encoding_issues: mojibake, replacement chars, reversed/garbled letters
- truncation: sentences or words cut off mid-way
- boilerplate: page numbers, publisher/site headers, footnote markers dominating the text
- structure_broken: heading and body clearly unrelated, or unrelated passages fused together
- duplication: the same passage repeated
- not_arabic_prose: mostly digits, symbols, indexes or tables of contents rather than prose

IMPORTANT: this PAGE is an EXCERPT cut from the middle of a book. It is EXPECTED to begin and end mid-sentence. Do NOT report that as truncation. Only report truncation if text is cut off INSIDE the excerpt, away from its two ends.

Return STRICT JSON only, no markdown fence, no commentary:
{"verdict":"clean|minor|broken","problems":["..."],"evidence":"<=160 chars quoted from the page","confidence":0.0-1.0}

verdict=clean means directly usable. minor means usable with cosmetic noise. broken means it would harm retrieval.

PAGE (book=%(book)s, section=%(section)s):
<<<
%(text)s
>>>"""


def audit(sample: dict, timeout: int, transport, model: str) -> dict:
    prompt = PROMPT % {"book": sample["book_id"], "section": (sample.get("section") or "")[:120],
                       "text": sample["text"]}
    try:
        v = transport.ask_json(prompt, model=model, timeout=timeout)
    except LLMError as exc:
        return {**{k: sample[k] for k in ("book_id", "section")}, "verdict": "error",
                "problems": ["llm_error"], "evidence": str(exc)[:160], "confidence": 0.0}
    v["book_id"] = sample["book_id"]
    v["section"] = sample.get("section")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--transport", default="openrouter", choices=TRANSPORTS,
                    help="which path to the model")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the transport's DEFAULT_MODEL")
    a = ap.parse_args(argv)
    transport = get_transport(a.transport)
    model = a.model or transport.DEFAULT_MODEL

    with open(a.samples, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"auditing {len(samples)} pages with {a.concurrency} concurrent "
          f"{model} readers", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, r in enumerate(ex.map(lambda s: audit(s, a.timeout, transport, model), samples), 1):
            results.append(r)
            print(f"  [{i}/{len(samples)}] {r['verdict']:7} {r['book_id'][-28:]:28} "
                  f"{','.join(r.get('problems') or [])[:60]}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\nverdicts:", counts)
    probs: dict[str, int] = {}
    for r in results:
        for p in r.get("problems") or []:
            probs[p] = probs.get(p, 0) + 1
    print("problems:", dict(sorted(probs.items(), key=lambda kv: -kv[1])))
    flagged = sorted({r["book_id"] for r in results if r["verdict"] == "broken"})
    print("flagged books:", len(flagged))
    for b in flagged:
        print("   ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
