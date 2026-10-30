"""Check candidate models for the source classifier and the query planner.

Runs the engine's own code paths (``detect_source_boost_llm`` and
``plan_query_variants``) with each model through OpenRouter.

- Classifier: the questions for which the keyword prior finds nothing, which
  are the only ones the LLM sees at query time. Correct when the model names
  the category of the question's topic label (topics with no source category,
  such as adab, count when the model answers "all" or one category).
- Planner: the first ``--planner-questions`` of those questions, with the
  critic. Valid when planner and critic both return usable JSON and every
  variant is Arabic.

    python -m dalil.s8_search.model_check --questions questions.jsonl \\
        --models anthropic/claude-haiku-5.5 qwen/qwen3.5-flash-02-23 --summary docs/model_check.md

``--questions`` is JSON Lines with ``question`` and ``topic`` fields.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dalil.llm.env import api_key
from dalil.paths import DB_DIR
from dalil.s8_search.engine import QueryEngine
from dalil.s8_search.sources import detect_source_boost_keywords

TOPIC_CATEGORY = {
    "fiqh": "fiqh", "aqeedah": "aqeedah", "hadith": "hadith", "tafsir": "tafsir",
    "quran_lookup": "quran", "seerah_history": "seerah", "usul_fiqh": "fiqh_usul",
}
ARABIC = re.compile(r"[؀-ۿ]")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def engine_for(model: str) -> QueryEngine:
    return QueryEngine(db_dir=DB_DIR, api_key=api_key("OPENROUTER_API_KEY"),
                       listwise_model=model, planner_model=model, planner_provider="")


def classify(model: str, row: dict) -> dict:
    engine = engine_for(model)
    engine._status = {}
    started = time.time()
    found = engine.detect_source_boost_llm(row["question"])
    seconds = time.time() - started
    status = engine._status.get("source_classifier", "")
    expected = TOPIC_CATEGORY.get(row.get("topic", ""))
    if not status.startswith("ok"):
        correct = False
    elif expected:
        correct = bool(found) and expected in found
    else:
        correct = found is None or len(found) == 1
    return {"question": row["question"], "topic": row.get("topic"), "model": model, "found": found,
            "status": status or "no answer", "correct": correct, "seconds": round(seconds, 2)}


def plan(model: str, row: dict) -> dict:
    engine = engine_for(model)
    started = time.time()
    result = engine.plan_query_variants(row["question"], max_queries=4, force_variants=True)
    seconds = time.time() - started
    variants = [v["query"] for v in result["queries"]]
    arabic = all(ARABIC.search(v) and not CJK.search(v) for v in variants)
    valid = not result.get("error") and not result.get("critic_error") and bool(variants) and arabic
    return {"question": row["question"], "model": model, "valid": valid, "variants": variants,
            "error": result.get("error") or result.get("critic_error"), "seconds": round(seconds, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--planner-questions", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="per-question results, JSON Lines")
    parser.add_argument("--summary", type=Path, help="Markdown summary")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    unmatched = [r for r in rows if not detect_source_boost_keywords(r["question"])]
    planner_rows = unmatched[:args.planner_questions]
    print(f"{len(rows)} questions, {len(unmatched)} without a keyword prior")

    table = ["| model | classifier correct | classifier failed | classifier median s | planner valid | planner median s |",
             "|---|---:|---:|---:|---:|---:|"]
    records = []
    for model in args.models:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            cls = list(pool.map(lambda r: classify(model, r), unmatched))
            pln = list(pool.map(lambda r: plan(model, r), planner_rows))
        records += [{"check": "classifier", **c} for c in cls] + [{"check": "planner", **p} for p in pln]
        failed = sum(not c["status"].startswith("ok") for c in cls)
        line = (f"| `{model}` | {sum(c['correct'] for c in cls)}/{len(cls)} | {failed} | "
                f"{statistics.median(c['seconds'] for c in cls):.1f} | {sum(p['valid'] for p in pln)}/{len(pln)} | "
                f"{statistics.median(p['seconds'] for p in pln):.1f} |")
        table.append(line)
        print(line, flush=True)

    if args.out:
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    if args.summary:
        args.summary.write_text("\n".join([
            "# Classifier and planner models",
            "",
            f"`python -m dalil.s8_search.model_check`: {len(unmatched)} questions without a keyword prior "
            f"(of {len(rows)}), planner on the first {len(planner_rows)} with the critic.",
            "",
            *table,
            "",
        ]), encoding="utf-8")


if __name__ == "__main__":
    main()
