"""Check candidate models for the source classifier and the query planner.

Runs the engine's own code paths (``detect_source_boost_llm`` and
``plan_query_variants``) with each model through OpenRouter.

- Classifier: the questions for which the keyword prior finds nothing, which
  are the only ones the LLM sees at query time. Correct when the model names the
  category of the question's topic label and at most ``--max-categories`` in all
  (topics with no source category, such as adab, count when it answers "all" or
  one category).
- Planner: the first ``--planner-questions`` of those questions, with the
  critic. Valid when planner and critic both return usable JSON and every
  variant is Arabic.

    python -m dalil.s8_search.model_check --questions questions.jsonl --source user \\
        --models google/gemma-4-26b-a4b-it google/gemini-3.1-flash-lite --summary docs/model_check.md

``--questions`` is JSON Lines with ``question`` and ``topic`` fields. The results
name each question by its line number in that file (``question_id``, from 1),
never by its text.
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


def classify(model: str, question_id: int, row: dict, max_categories: int) -> dict:
    started = time.time()
    found, status = engine_for(model).classify_source_type(row["question"])
    seconds = time.time() - started
    expected = TOPIC_CATEGORY.get(row.get("topic", ""))
    named = len(found or [])
    if status != "ok":
        correct = False
    elif expected:
        correct = bool(found) and expected in found and named <= max_categories
    else:
        correct = named <= 1
    return {"question_id": question_id, "topic": row.get("topic"), "model": model, "found": found, "named": named,
            "status": status, "correct": correct, "seconds": round(seconds, 2)}


def plan(model: str, question_id: int, row: dict) -> dict:
    engine = engine_for(model)
    started = time.time()
    result = engine.plan_query_variants(row["question"], max_queries=4, force_variants=True)
    seconds = time.time() - started
    variants = [v["query"] for v in result["queries"]]
    arabic = all(ARABIC.search(v) and not CJK.search(v) for v in variants)
    valid = not result.get("error") and not result.get("critic_error") and bool(variants) and arabic
    return {"question_id": question_id, "model": model, "valid": valid, "variants": len(variants),
            "error": result.get("error") or result.get("critic_error"), "seconds": round(seconds, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--source", help="only the questions whose source field is this, such as user")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--planner-questions", type=int, default=30)
    parser.add_argument("--max-categories", type=int, default=2,
                        help="an answer naming more source types than this is not correct")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="per-question results, JSON Lines")
    parser.add_argument("--summary", type=Path, help="Markdown summary")
    args = parser.parse_args()

    lines = [line for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [(i, r) for i, r in enumerate(map(json.loads, lines), 1) if not args.source or r.get("source") == args.source]
    unmatched = [(i, r) for i, r in rows if not detect_source_boost_keywords(r["question"])]
    planner_rows = unmatched[:args.planner_questions]
    print(f"{len(rows)} questions, {len(unmatched)} without a keyword prior")

    table = ["| model | classifier correct | types named, mean | classifier failed | classifier median s "
             "| planner valid | planner median s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    records = []
    for model in args.models:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            cls = list(pool.map(lambda x: classify(model, *x, args.max_categories), unmatched))
            pln = list(pool.map(lambda x: plan(model, *x), planner_rows))
        records += [{"check": "classifier", **c} for c in cls] + [{"check": "planner", **p} for p in pln]
        failed = sum(c["status"] != "ok" for c in cls)
        answered = [c["named"] for c in cls if c["status"] == "ok"]
        named = f"{statistics.mean(answered):.1f}" if answered else "-"
        line = (f"| `{model}` | {sum(c['correct'] for c in cls)}/{len(cls)} | {named} | {failed} | "
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
            f"`python -m dalil.s8_search.model_check`: the classifier on {len(unmatched)} questions without a "
            f"keyword prior (of {len(rows)}), correct when it names the question's source type and at most "
            f"{args.max_categories} in all; the planner on the first {len(planner_rows)}, with the critic.",
            "",
            *table,
            "",
        ]), encoding="utf-8")


if __name__ == "__main__":
    main()
