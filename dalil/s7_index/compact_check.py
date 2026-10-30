"""Compare compact two-stage dense search with exact search, question by question.

Each question is embedded once (OpenRouter, as at query time) and searched both
ways over the same stores: exact (``DenseIndex``, every vector) and compact
(``CompactDenseIndex``, the fp16 first-``dims`` copy, top ``rescore`` rescored
exactly). For each question it records how many of the exact top 10, 50 and 200
views the compact search also returns in its top 10, 50 and 200.

    python -m dalil.s7_index.compact_check --questions questions.jsonl \\
        --out docs/compact_check.jsonl --summary docs/compact_check.md

``--questions`` is JSON Lines with a ``question`` field. The results name each
question by its line number in that file (``question_id``, from 1), and keep the
``--group-by`` field, never the question.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dalil.llm.env import api_key
from dalil.paths import DB_DIR
from dalil.s7_index.compact import compact_path
from dalil.s8_search.dense import CompactDenseIndex, DenseIndex
from dalil.s8_search.engine import QueryEngine

DEPTHS = (10, 50, 200)


def load_questions(path: Path) -> list[tuple[int, dict]]:
    """(question_id, row) for every row with a question; the id is the row's line number."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(i, r) for i, r in enumerate(rows, 1) if (r.get("question") or "").strip()]


def embed_all(questions: list[str], db_dir: Path, workers: int) -> np.ndarray:
    engine = QueryEngine(db_dir=db_dir, api_key=api_key("OPENROUTER_API_KEY"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        vectors = list(pool.map(engine.embed_query, questions))
    return np.stack(vectors).astype(np.float32)


def overlap(exact: np.ndarray, compact: np.ndarray, k: int) -> int:
    return len(set(exact[:k].tolist()) & set(compact[:k].tolist()))


def summarise(rows: list[dict], label: str) -> list[str]:
    n = len(rows)
    out = [f"### {label} ({n} questions)", "", "| | top 10 | top 50 | top 200 |", "|---|---:|---:|---:|"]
    mean = [sum(r[f"overlap_{k}"] for r in rows) / (n * k) for k in DEPTHS]
    worst = [min(r[f"overlap_{k}"] for r in rows) for k in DEPTHS]
    full = [sum(r[f"overlap_{k}"] == k for r in rows) for k in DEPTHS]
    out.append("| mean overlap | " + " | ".join(f"{m:.2%}" for m in mean) + " |")
    out.append("| worst question | " + " | ".join(f"{w}/{k}" for w, k in zip(worst, DEPTHS)) + " |")
    out.append("| questions identical as a set | " + " | ".join(f"{f}/{n}" for f in full) + " |")
    out.append("| questions identical in order | " + " | ".join(
        f"{sum(r[f'same_order_{k}'] for r in rows)}/{n}" for k in DEPTHS) + " |")
    return out + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=DB_DIR)
    parser.add_argument("--dims", type=int, default=1024)
    parser.add_argument("--rescore", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8, help="parallel embedding requests")
    parser.add_argument("--group-by", default="source", help="field to break the summary down by")
    parser.add_argument("--out", type=Path, required=True, help="per-question results, JSON Lines")
    parser.add_argument("--summary", type=Path, help="Markdown summary")
    args = parser.parse_args()

    rows = load_questions(args.questions)
    started = time.time()
    queries = embed_all([r["question"] for _, r in rows], args.db_dir, args.workers)
    print(f"embedded {len(rows)} questions in {time.time() - started:.0f}s")

    full_path = args.db_dir / "dense_faiss_ar.index"
    k = max(DEPTHS)
    started = time.time()
    _, exact_ids = DenseIndex(full_path).search(queries, k)
    exact_s = time.time() - started
    started = time.time()
    compact = CompactDenseIndex(full_path, compact_path(args.db_dir, args.dims), rescore=args.rescore)
    _, compact_ids = compact.search(queries, k)
    compact_s = time.time() - started
    print(f"exact {exact_s:.0f}s, compact {compact_s:.0f}s for {len(rows)} questions")

    results = []
    for (question_id, row), e, c in zip(rows, exact_ids, compact_ids):
        result = {"question_id": question_id, **{k: row[k] for k in (args.group_by,) if k in row}}
        for depth in DEPTHS:
            result[f"overlap_{depth}"] = overlap(e, c, depth)
            result[f"same_order_{depth}"] = bool(np.array_equal(e[:depth], c[:depth]))
        results.append(result)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    if args.summary:
        lines = [
            "# Compact dense search against exact search",
            "",
            f"`python -m dalil.s7_index.compact_check`, {len(rows)} questions, compact index "
            f"{args.dims} dims fp16, rescore {args.rescore}. Overlap of the top k retrieval views; "
            f"per-question results in `{args.out.name}`.",
            "",
        ]
        lines += summarise(results, "all")
        groups = sorted({r.get(args.group_by) for r in results if r.get(args.group_by)})
        for g in groups if len(groups) > 1 else []:
            lines += summarise([r for r in results if r.get(args.group_by) == g], f"{args.group_by} = {g}")
        args.summary.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(summarise(results, "all")))


if __name__ == "__main__":
    main()
