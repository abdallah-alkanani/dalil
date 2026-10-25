"""Command-line search.

    python -m dalil.s8_search "ما حكم الربا؟" --top 10 --scores
    python -m dalil.s8_search "فضل الصدقة" --category hadith --json
    python -m dalil.s8_search "هل يجوز الجمع بين الصلاتين للمطر؟" --query-planning auto
    python -m dalil.s8_search --list-sources

Queries embed through OpenRouter (``OPENROUTER_API_KEY``); ``--retrieval bm25``
needs no key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dalil.llm.env import ENV_FILE, api_key, load_env_file
from dalil.paths import DB_DIR

from .api import (
    EMBED_MODEL,
    LISTWISE_MODEL,
    OPENROUTER_URL,
    PLANNER_MODEL,
    PLANNER_PROVIDER,
)
from .engine import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_MMR_DEPTH,
    DEFAULT_RRF_K,
    QueryEngine,
)
from .filters import normalize_filters
from .formatting import format_results
from .sources import parse_categories

resolve_api_key = api_key


def build_filters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return normalize_filters(
        {
            "subcategory": args.subcategory,
            "book_id": args.book_id,
            "parent_id": args.parent_id,
            "view_id": args.view_id,
            "source": args.source,
            "source_corpus": args.source_corpus,
            "view_type": args.view_type,
            "text_role": args.text_role,
            "work_id": args.work_id,
            "author_id": args.author_id,
            "parser_profile": args.parser_profile,
            "file_format": args.file_format,
            "quality_flag": args.quality_flag,
            "parent_type": args.parent_type,
            "parent_policy": args.parent_policy,
            "view_policy": args.view_policy,
            "citation_contains": args.citation_contains,
            "title": args.title,
            "section_contains": args.section_contains,
            "section_id": args.section_id,
            "quran_surah": args.quran_surah,
            "quran_ayah": args.quran_ayah,
            "hadith_collection": args.hadith_collection,
            "hadith_number": args.hadith_number,
            "hadith_book_number": args.hadith_book_number,
            "hadith_chapter_id": args.hadith_chapter_id,
            "source_record_id": args.source_record_id,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "min_source_line": args.min_source_line,
            "max_source_line": args.max_source_line,
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dalil search: hybrid retrieval with cited parent passages")
    parser.add_argument("question", nargs="?", help="Question to search")
    parser.add_argument("--db-dir", type=Path, default=DB_DIR, help="directory holding the search stores")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-url", default=OPENROUTER_URL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--listwise-model", default=LISTWISE_MODEL)
    parser.add_argument("--planner-model", default=PLANNER_MODEL)
    parser.add_argument("--planner-provider", default=PLANNER_PROVIDER)
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of results")
    parser.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=DEFAULT_BM25_WEIGHT)
    parser.add_argument(
        "--query-planning",
        choices=["off", "auto", "on"],
        default="off",
        help="intent-preserving sequential query planning (opt-in; use auto/on)",
    )
    parser.add_argument(
        "--max-query-variants", type=int, default=6,
        help="maximum total queries including the exact original",
    )
    parser.add_argument(
        "--variant-depth", type=int, default=0,
        help="candidate depth per variant (0 = top in lossless no-listwise mode; otherwise max(30, top*5))",
    )
    critic = parser.add_mutually_exclusive_group()
    critic.add_argument("--intent-critic", dest="intent_critic", action="store_true", default=True)
    critic.add_argument("--no-intent-critic", dest="intent_critic", action="store_false")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-depth", type=int, default=DEFAULT_MMR_DEPTH)
    parser.add_argument("--category", action="append", default=None, help="Filter category; repeat or comma-separate")
    parser.add_argument("--subcategory", action="append", default=None)
    parser.add_argument("--book-id", action="append", default=None)
    parser.add_argument("--parent-id", action="append", default=None)
    parser.add_argument("--view-id", action="append", default=None)
    parser.add_argument("--source", action="append", default=None, help="Child source field, usually api/local")
    parser.add_argument("--source-corpus", action="append", default=None)
    parser.add_argument("--view-type", action="append", default=None)
    parser.add_argument("--text-role", action="append", default=None)
    parser.add_argument("--work-id", action="append", default=None)
    parser.add_argument("--author-id", action="append", default=None)
    parser.add_argument("--parser-profile", action="append", default=None)
    parser.add_argument("--file-format", action="append", default=None)
    parser.add_argument("--quality-flag", action="append", default=None)
    parser.add_argument("--parent-type", action="append", default=None)
    parser.add_argument("--parent-policy", action="append", default=None)
    parser.add_argument("--view-policy", action="append", default=None)
    parser.add_argument("--citation-contains", action="append", default=None)
    parser.add_argument("--title", action="append", default=None, help="Substring match against book_id/section path")
    parser.add_argument("--section-contains", action="append", default=None)
    parser.add_argument("--section-id", action="append", default=None)
    parser.add_argument("--quran-surah", action="append", default=None)
    parser.add_argument("--quran-ayah", type=int, default=None)
    parser.add_argument("--hadith-collection", action="append", default=None)
    parser.add_argument("--hadith-number", action="append", default=None)
    parser.add_argument("--hadith-book-number", action="append", default=None)
    parser.add_argument("--hadith-chapter-id", action="append", default=None)
    parser.add_argument("--source-record-id", action="append", default=None)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--min-source-line", type=int, default=None)
    parser.add_argument("--max-source-line", type=int, default=None)
    parser.add_argument(
        "--boost",
        choices=["hadith", "quran", "tafsir", "fiqh", "fiqh_usul", "aqeedah", "seerah", "all"],
        default=None,
        help="Manual source prior",
    )
    listwise = parser.add_mutually_exclusive_group()
    listwise.add_argument("--listwise", dest="listwise", action="store_true", default=False)
    listwise.add_argument("--no-listwise", dest="listwise", action="store_false")
    parser.add_argument("--scores", action="store_true", help="Show retrieval scores")
    parser.add_argument(
        "--expand-neighbours", "--expand-neighbors", dest="expand_neighbours",
        type=int, default=0, metavar="N",
        help="also return the N parents before and after each hit, in reading order. "
             "Use when an argument may span several parents and the later ones do not "
             "restate the topic. Requires parents.order_in_book (dalil.s7_index.navigation).")
    parser.add_argument("--group-by-tier", action="store_true", help="Group display by source tier instead of rank order")
    parser.add_argument(
        "--parent-max-chars", type=int, default=12000, metavar="N",
        help="cap the parent text handed to the model, windowed around the matched "
             "passage (0 = no cap). Default ~12k chars ≈ 4k tokens: 95%% of parents "
             "are already smaller, and models measurably degrade past a few thousand "
             "tokens of context, so an unbounded parent hurts answers as well as cost.")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(args.env_file)

    api_key = resolve_api_key(args.api_key_env)
    engine = QueryEngine(
        db_dir=args.db_dir,
        api_key=api_key,
        openrouter_url=args.openrouter_url,
        embed_model=args.embed_model,
        listwise_model=args.listwise_model,
        planner_model=args.planner_model,
        planner_provider=args.planner_provider,
    )
    try:
        if args.list_sources:
            engine.validate_startup(require_dense=False)
            print(engine.list_sources())
            return 0

        if not args.question:
            raise SystemExit("question is required unless --list-sources is used")

        categories = parse_categories(args.category)
        if categories:
            print(f"  [Filter] Categories: {', '.join(categories)}", file=sys.stderr)
        filters = build_filters_from_args(args)

        print(f"\nQuery: {args.question}", file=sys.stderr)
        output = engine.query(
            args.question,
            n_results=args.top,
            use_listwise_rerank=args.listwise,
            mmr_lambda=args.mmr_lambda,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            retrieval_mode=args.retrieval,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            mmr_depth=args.mmr_depth,
            categories=categories,
            filters=filters,
            boost_override=args.boost,
            parent_max_chars=args.parent_max_chars,
            expand_neighbours=args.expand_neighbours,
            query_planning=args.query_planning,
            max_query_variants=args.max_query_variants,
            use_intent_critic=args.intent_critic,
            variant_depth=args.variant_depth,
        )

        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"\n{'=' * 80}")
            print(format_results(output, show_scores=args.scores, group_by_tier=args.group_by_tier))
        return 0
    except Exception as exc:
        if args.json:
            print(json.dumps({"results": [], "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.close()
