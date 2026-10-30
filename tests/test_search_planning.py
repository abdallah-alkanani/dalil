#!/usr/bin/env python3
"""Query planning, variant fusion, the engine's query flow, and the search CLI."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dalil.arabic.normalize import normalize_for_search
from dalil.s8_search import cli, planning
from dalil.s8_search import engine as search_engine


class ArabicSearchNormalizationTests(unittest.TestCase):
    def test_query_and_index_forms_share_required_normalization(self):
        self.assertEqual(normalize_for_search("إِنَّ ٱلصَّلَاةَ ١٢۳"), "ان الصلاة 123")

    def test_semantically_distinct_letters_are_not_folded(self):
        self.assertEqual(normalize_for_search("على علي رحمة رحمه"), "على علي رحمة رحمه")

    def test_source_cleanup_never_changes_display_input_in_place(self):
        original = "### | CHECK عنوان PageBeg متن"
        normalized = normalize_for_search(original)
        self.assertEqual(original, "### | CHECK عنوان PageBeg متن")
        self.assertEqual(normalized, "عنوان متن")


class WorkspaceIntegrationTests(unittest.TestCase):
    def test_readonly_sqlite_connection_sees_committed_wal_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.sqlite"
            writer = sqlite3.connect(path)
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE rows(value TEXT)")
            writer.commit()
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            writer.execute("INSERT INTO rows VALUES ('committed-in-wal')")
            writer.commit()
            reader = search_engine.open_sqlite_readonly(path)
            try:
                self.assertEqual(reader.execute("SELECT COUNT(*) FROM rows").fetchone()[0], 1)
            finally:
                reader.close()
                writer.close()

    def test_api_key_resolution_uses_secure_user_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / ".config" / "openrouter.key"
            key_path.parent.mkdir()
            key_path.write_text(" shared-key\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(Path, "home", return_value=Path(tmp)):
                    self.assertEqual(
                        cli.resolve_api_key("OPENROUTER_API_KEY"), "shared-key"
                    )

    def test_custom_api_key_env_does_not_fall_back_to_openrouter_default(self):
        with mock.patch.dict(os.environ, {"CUSTOM_RAG_KEY": " custom-key "}, clear=True):
            with mock.patch.object(Path, "home") as user_home:
                self.assertEqual(cli.resolve_api_key("CUSTOM_RAG_KEY"), "custom-key")
                user_home.assert_not_called()


class QueryPlanParsingTests(unittest.TestCase):
    def test_original_query_is_first_and_arabic_duplicates_are_removed(self):
        original = "أيهما أرجح في تفسير الآية: القول الأول أم القول الثاني؟"
        response = json.dumps(
            {
                "use_variants": True,
                "intent": "الترجيح بين القولين في تفسير الآية",
                "forbidden_adjacent": ["ما ترجمة صاحب القول الأول؟"],
                "queries": [
                    {"query": "ما معيار الترجيح بين القولين؟", "purpose": "criterion"},
                    {"query": "ما معيارُ الترجيح بين القولين؟", "purpose": "duplicate with diacritics"},
                    {"query": original, "purpose": "duplicate original"},
                ],
            },
            ensure_ascii=False,
        )

        plan = planning.parse_query_plan_response(response, original, max_queries=4)

        self.assertTrue(plan["use_variants"])
        self.assertEqual(plan["queries"][0]["query"], original)
        self.assertEqual(
            [item["query"] for item in plan["queries"]],
            [original, "ما معيار الترجيح بين القولين؟"],
        )
        self.assertEqual(plan["queries"][1]["purpose"], "criterion")

    def test_invalid_planner_response_falls_back_to_original_query(self):
        original = "ما حكم الربا؟"

        plan = planning.parse_query_plan_response("not json", original, max_queries=4)

        self.assertFalse(plan["use_variants"])
        self.assertEqual(plan["queries"], [{"query": original, "purpose": "original"}])
        self.assertIsNotNone(plan["error"])

    def test_string_false_and_non_string_query_are_rejected(self):
        original = "الأصل"
        responses = [
            json.dumps({"use_variants": "false", "queries": []}),
            json.dumps({"use_variants": True, "queries": [{"query": ["bad"], "purpose": "x"}]}),
        ]
        for response in responses:
            with self.subTest(response=response):
                plan = planning.parse_query_plan_response(response, original, max_queries=4)
                self.assertFalse(plan["use_variants"])
                self.assertEqual(plan["queries"], [{"query": original, "purpose": "original"}])
                self.assertIsNotNone(plan["error"])

    def test_fenced_json_is_accepted_and_query_limit_is_enforced(self):
        original = "قارن بين القولين"
        response = "```json\n" + json.dumps(
            {
                "use_variants": True,
                "intent": "comparison",
                "queries": [
                    {"query": "سؤال ١", "purpose": "one"},
                    {"query": "سؤال ٢", "purpose": "two"},
                    {"query": "سؤال ٣", "purpose": "three"},
                ],
            },
            ensure_ascii=False,
        ) + "\n```"

        plan = planning.parse_query_plan_response(response, original, max_queries=3)

        self.assertEqual([q["query"] for q in plan["queries"]], [original, "سؤال ١", "سؤال ٢"])


class QueryIntentCriticTests(unittest.TestCase):
    def test_adjacent_queries_are_removed_but_original_is_never_removed(self):
        variants = [
            {"query": "أيهما أرجح: القول الأول أم الثاني؟", "purpose": "original", "variant_id": "Q0"},
            {"query": "ما معيار الترجيح بين القولين؟", "purpose": "criterion", "variant_id": "Q1"},
            {"query": "ما ترجمة صاحب القول الأول؟", "purpose": "adjacent", "variant_id": "Q2"},
        ]
        response = json.dumps(
            {
                "decisions": [
                    {"variant_id": "Q0", "label": "drifted_adjacent"},
                    {"variant_id": "Q1", "label": "supporting_same_question"},
                    {"variant_id": "Q2", "label": "drifted_adjacent"},
                ]
            }
        )

        accepted, decisions = planning.filter_variants_with_critic_response(variants, response)

        self.assertEqual([v["variant_id"] for v in accepted], ["Q0", "Q1"])
        self.assertEqual(decisions["Q2"], "drifted_adjacent")

    def test_valid_partial_critic_response_drops_ungraded_generated_queries(self):
        variants = [
            {"query": "الأصل", "purpose": "original", "variant_id": "Q0"},
            {"query": "سؤال غير مقيم", "purpose": "support", "variant_id": "Q1"},
        ]
        response = json.dumps(
            {"decisions": [{"variant_id": "Q0", "label": "preserves_intent"}]}
        )

        accepted, decisions = planning.filter_variants_with_critic_response(variants, response)

        self.assertEqual([item["variant_id"] for item in accepted], ["Q0"])
        self.assertEqual(decisions, {})

    def test_invalid_critic_response_fails_closed_to_original_query(self):
        variants = [
            {"query": "الأصل", "purpose": "original", "variant_id": "Q0"},
            {"query": "سؤال داعم", "purpose": "support", "variant_id": "Q1"},
        ]

        accepted, decisions = planning.filter_variants_with_critic_response(variants, "bad json")

        self.assertEqual(accepted, variants[:1])
        self.assertEqual(decisions, {})


class QueryPlannerIntegrationTests(unittest.TestCase):
    def test_planner_and_critic_produce_intent_checked_variants(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.responses = [
                    json.dumps(
                        {
                            "use_variants": True,
                            "intent": "compare the two people",
                            "forbidden_adjacent": ["prayer validity only"],
                            "queries": [
                                {"query": "ما معيار المفاضلة؟", "purpose": "criterion"},
                                {"query": "هل تقبل الصلاة؟", "purpose": "adjacent"},
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "decisions": [
                                {"variant_id": "Q0", "label": "preserves_intent"},
                                {"variant_id": "Q1", "label": "supporting_same_question"},
                                {"variant_id": "Q2", "label": "drifted_adjacent"},
                            ]
                        }
                    ),
                ]
                self.prompts: list[str] = []

            def _planning_completion(self, prompt: str, max_tokens: int = 800) -> str:
                self.prompts.append(prompt)
                return self.responses.pop(0)

        engine = FakeEngine()

        plan = engine.plan_query_variants(
            "Which interpretation is better supported?",
            max_queries=4,
            use_critic=True,
            force_variants=True,
        )

        self.assertEqual([v["variant_id"] for v in plan["queries"]], ["Q0", "Q1"])
        self.assertEqual(plan["queries"][0]["query"], "Which interpretation is better supported?")
        self.assertEqual(plan["critic_decisions"]["Q2"], "drifted_adjacent")
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("explicitly requested expansion", engine.prompts[0].lower())
        self.assertIn("exact user question", engine.prompts[0].lower())
        self.assertIn("keep every named side", engine.prompts[0].lower())
        self.assertIn("evidence_dimensions", engine.prompts[0])
        self.assertIn("assumptions_unknowns", engine.prompts[0])
        self.assertIn("conditions_exceptions", engine.prompts[0])
        self.assertIn("governing_criterion", engine.prompts[0])
        self.assertIn("terminology_scope", engine.prompts[0])
        self.assertIn("drifted_adjacent", engine.prompts[1])
        self.assertIn("json", engine.prompts[1].lower())

    def test_planner_timeout_skips_expansion_and_keeps_original(self):
        class TimeoutEngine(search_engine.QueryEngine):
            def __init__(self):
                pass

            def _planning_completion(self, prompt: str, max_tokens: int = 800) -> str:
                raise search_engine.requests.Timeout("planner exceeded deadline")

        plan = TimeoutEngine().plan_query_variants("السؤال الأصلي", max_queries=6, use_critic=True)

        self.assertFalse(plan["use_variants"])
        self.assertEqual(plan["queries"], [{"query": "السؤال الأصلي", "purpose": "original", "variant_id": "Q0"}])
        self.assertIn("planner exceeded deadline", plan["error"])

    def test_critic_transport_failure_keeps_only_original_query(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.calls = 0

            def _planning_completion(self, prompt: str, max_tokens: int = 800) -> str:
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {
                            "use_variants": True,
                            "intent": "comparison",
                            "queries": [{"query": "سؤال مولد", "purpose": "governing_criterion"}],
                        },
                        ensure_ascii=False,
                    )
                raise RuntimeError("critic unavailable")

        plan = FakeEngine().plan_query_variants("الأصل", max_queries=3, use_critic=True)

        self.assertEqual([item["variant_id"] for item in plan["queries"]], ["Q0"])
        self.assertFalse(plan["use_variants"])
        self.assertIn("critic unavailable", plan["critic_error"])

    def test_simple_plan_does_not_call_critic(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.calls = 0

            def _planning_completion(self, prompt: str, max_tokens: int = 800) -> str:
                self.calls += 1
                return json.dumps({"use_variants": False, "intent": "exact lookup", "queries": []})

        engine = FakeEngine()

        plan = engine.plan_query_variants("آية الكرسي", max_queries=4, use_critic=True)

        self.assertEqual(engine.calls, 1)
        self.assertEqual([v["variant_id"] for v in plan["queries"]], ["Q0"])


class PlannedQueryEndToEndTests(unittest.TestCase):
    def test_variant_depth_controls_first_stage_retrieval_depth(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.requested_depths = []

            def validate_startup(self, *, require_dense=True):
                return None

            def compute_allowed_faiss_rows(self, filters):
                return None

            def plan_query_variants(self, question, max_queries=4, use_critic=True, force_variants=False):
                return {
                    "use_variants": True,
                    "intent": question,
                    "forbidden_adjacent": [],
                    "critic_decisions": {},
                    "queries": [
                        {"variant_id": "Q0", "query": question, "purpose": "original"},
                        {"variant_id": "Q1", "query": "support", "purpose": "evidence_dimensions"},
                    ],
                }

            def _retrieve_candidates_for_question(self, question, **kwargs):
                self.requested_depths.append(kwargs["n_results"])
                rows = [
                    {"parent_id": f"{question}-{i}", "hybrid_score": 1.0 / (i + 1), "semantic_sim": 1.0 / (i + 1)}
                    for i in range(kwargs["n_results"])
                ]
                return {"candidates": rows, "index": None, "qvec": None, "searched_k": len(rows)}

            def choose_boost_categories(self, question, override):
                return None

            def attach_neighbours(self, results, window):
                return None

        engine = FakeEngine()
        output = engine.query(
            "original", n_results=2, variant_depth=7, query_planning="on",
            use_listwise_rerank=False, mmr_depth=0,
        )

        self.assertEqual(engine.requested_depths, [7, 7])
        self.assertEqual(len(output["results"]), 14)

    def test_generated_variant_failure_preserves_successful_original_results(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                pass

            def validate_startup(self, *, require_dense=True):
                return None

            def compute_allowed_faiss_rows(self, filters):
                return None

            def plan_query_variants(self, question, max_queries=4, use_critic=True, force_variants=False):
                return {
                    "use_variants": True,
                    "intent": question,
                    "forbidden_adjacent": [],
                    "critic_decisions": {},
                    "queries": [
                        {"variant_id": "Q0", "query": question, "purpose": "original"},
                        {"variant_id": "Q1", "query": "fails", "purpose": "evidence_dimensions"},
                    ],
                }

            def _retrieve_candidates_for_question(self, question, **kwargs):
                if question == "fails":
                    raise RuntimeError("variant backend unavailable")
                return {
                    "candidates": [{"parent_id": "A", "hybrid_score": 1.0, "semantic_sim": 1.0}],
                    "index": None, "qvec": None, "searched_k": 1,
                }

            def choose_boost_categories(self, question, override):
                return None

            def attach_neighbours(self, results, window):
                return None

        output = FakeEngine().query(
            "original", n_results=1, query_planning="on",
            use_listwise_rerank=False, mmr_depth=0,
        )

        self.assertEqual([item["parent_id"] for item in output["results"]], ["A"])
        self.assertIn("Q1", output["variant_errors"])

    def test_planning_off_never_calls_planner_and_runs_exact_question_once(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.retrieval_calls = []
                self.retrieval_depths = []

            def validate_startup(self, *, require_dense=True):
                return None

            def compute_allowed_faiss_rows(self, filters):
                return None

            def plan_query_variants(self, *args, **kwargs):
                raise AssertionError("planner must not run when query_planning=off")

            def _retrieve_candidates_for_question(self, question, **kwargs):
                self.retrieval_calls.append(question)
                self.retrieval_depths.append(kwargs["n_results"])
                return {
                    "candidates": [
                        {"parent_id": "A", "hybrid_score": 1.0, "semantic_sim": 1.0}
                    ],
                    "index": None,
                    "qvec": None,
                    "searched_k": 1,
                }

            def choose_boost_categories(self, question, override):
                return None

            def attach_neighbours(self, results, window):
                return None

        engine = FakeEngine()
        output = engine.query(
            "exact question",
            n_results=1,
            use_listwise_rerank=False,
            mmr_depth=0,
            query_planning="off",
        )

        self.assertEqual(engine.retrieval_calls, ["exact question"])
        self.assertEqual(engine.retrieval_depths, [1])
        self.assertEqual(output["query_variants_executed"], 1)
        self.assertFalse(output["query_plan"]["use_variants"])

    def test_no_listwise_planned_query_preserves_top_n_from_each_variant(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                pass

            def validate_startup(self, *, require_dense=True):
                return None

            def compute_allowed_faiss_rows(self, filters):
                return None

            def plan_query_variants(self, question, max_queries=4, use_critic=True, force_variants=False):
                return {
                    "use_variants": True,
                    "intent": "two evidence axes",
                    "forbidden_adjacent": [],
                    "critic_decisions": {},
                    "queries": [
                        {"variant_id": "Q0", "query": "q0", "purpose": "original"},
                        {"variant_id": "Q1", "query": "q1", "purpose": "criterion"},
                    ],
                }

            def _retrieve_candidates_for_question(self, question, **kwargs):
                rows = {
                    "q0": [
                        {"parent_id": "A", "hybrid_score": 1.0, "semantic_sim": 1.0},
                        {"parent_id": "B", "hybrid_score": 0.9, "semantic_sim": 0.9},
                        {"parent_id": "DROP0", "hybrid_score": 0.1, "semantic_sim": 0.1},
                    ],
                    "q1": [
                        {"parent_id": "C", "hybrid_score": 1.0, "semantic_sim": 1.0},
                        {"parent_id": "D", "hybrid_score": 0.9, "semantic_sim": 0.9},
                        {"parent_id": "DROP1", "hybrid_score": 0.1, "semantic_sim": 0.1},
                    ],
                }[question]
                return {"candidates": rows, "index": None, "qvec": None, "searched_k": 3}

            def choose_boost_categories(self, question, override):
                return None

            def attach_neighbours(self, results, window):
                return None

        output = FakeEngine().query(
            "q0",
            n_results=2,
            use_listwise_rerank=False,
            mmr_depth=0,
            query_planning="on",
        )

        self.assertEqual({item["parent_id"] for item in output["results"]}, {"A", "B", "C", "D"})
        self.assertTrue(output["lossless_variant_pool"])

    def test_public_query_runs_planned_variants_sequentially_and_fuses_once(self):
        class FakeEngine(search_engine.QueryEngine):
            def __init__(self):
                self.calls: list[str] = []

            def validate_startup(self, require_dense: bool = True) -> None:
                return None

            def compute_allowed_faiss_rows(self, filters):
                return None

            def plan_query_variants(self, question, max_queries=4, use_critic=True, force_variants=False):
                return {
                    "use_variants": True,
                    "intent": "same comparison",
                    "forbidden_adjacent": [],
                    "critic_decisions": {"Q0": "preserves_intent", "Q1": "supporting_same_question"},
                    "queries": [
                        {"variant_id": "Q0", "query": question, "purpose": "original"},
                        {"variant_id": "Q1", "query": "criterion", "purpose": "criterion"},
                    ],
                }

            def _retrieve_candidates_for_question(self, question, **kwargs):
                self.calls.append(question)
                candidates = {
                    "original": [
                        {"parent_id": "A", "hybrid_score": 0.9, "semantic_sim": 0.9},
                        {"parent_id": "B", "hybrid_score": 0.8, "semantic_sim": 0.8},
                    ],
                    "criterion": [
                        {"parent_id": "B", "hybrid_score": 0.95, "semantic_sim": 0.95},
                        {"parent_id": "C", "hybrid_score": 0.7, "semantic_sim": 0.7},
                    ],
                }[question]
                return {
                    "candidates": candidates,
                    "index": None,
                    "qvec": None,
                    "searched_k": 2,
                    "bm25_parent_count": 0,
                }

            def choose_boost_categories(self, question, override):
                return None

            def attach_neighbours(self, results, window):
                return None

        engine = FakeEngine()

        output = engine.query(
            "original",
            n_results=3,
            use_listwise_rerank=False,
            mmr_depth=0,
            retrieval_mode="hybrid",
            query_planning="auto",
            max_query_variants=4,
        )

        self.assertEqual(engine.calls, ["original", "criterion"])
        self.assertEqual([item["parent_id"] for item in output["results"]], ["B", "A", "C"])
        self.assertEqual(output["query_plan"]["intent"], "same comparison")
        self.assertEqual(output["query_variants_executed"], 2)


class QueryPlannerTransportTests(unittest.TestCase):
    def test_planner_prefers_provider_with_fallbacks_and_never_retries(self):
        class FakeResponse:
            def json(self):
                return {"choices": [{"message": {"content": '{"use_variants": false}'}}]}

        engine = search_engine.QueryEngine.__new__(search_engine.QueryEngine)
        engine.api_key = "test-key"
        engine.openrouter_url = "https://example.invalid"
        engine.planner_model = "some/model"
        engine.planner_provider = "someprovider"

        with mock.patch.object(search_engine, "request_with_retries", return_value=FakeResponse()) as request:
            content = engine._planning_completion("return JSON")

        self.assertEqual(content, '{"use_variants": false}')
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 55)
        self.assertEqual(kwargs["retries"], 0)
        self.assertEqual(kwargs["json_body"]["model"], "some/model")
        self.assertEqual(
            kwargs["json_body"]["provider"],
            {"order": ["someprovider"], "allow_fallbacks": True},
        )

        engine.planner_provider = ""
        with mock.patch.object(search_engine, "request_with_retries", return_value=FakeResponse()) as request:
            engine._planning_completion("return JSON")
        self.assertNotIn("provider", request.call_args.kwargs["json_body"])


class QueryCliTests(unittest.TestCase):
    def test_cli_accepts_explicit_planning_and_sequential_controls(self):
        argv = [
            "dalil-search",
            "complex question",
            "--query-planning",
            "on",
            "--max-query-variants",
            "5",
            "--variant-depth",
            "70",
            "--no-intent-critic",
        ]
        with mock.patch("sys.argv", argv):
            args = cli.parse_args()

        self.assertEqual(args.query_planning, "on")
        self.assertEqual(args.max_query_variants, 5)
        self.assertEqual(args.variant_depth, 70)
        self.assertFalse(args.intent_critic)

    def test_cli_defaults_preserve_opt_in_network_behavior(self):
        with mock.patch("sys.argv", ["dalil-search", "simple question"]):
            args = cli.parse_args()

        self.assertEqual(args.query_planning, "off")
        self.assertFalse(args.listwise)
        self.assertEqual(args.max_query_variants, 6)
        self.assertEqual(args.planner_model, search_engine.PLANNER_MODEL)
        self.assertTrue(args.intent_critic)

    def test_list_sources_does_not_load_dense_index(self):
        engine = mock.Mock()
        engine.list_sources.return_value = "sources"
        with mock.patch("sys.argv", ["dalil-search", "--list-sources"]):
            with mock.patch.object(cli, "resolve_api_key", return_value=""):
                with mock.patch.object(cli, "QueryEngine", return_value=engine):
                    with mock.patch("builtins.print"):
                        self.assertEqual(cli.main(), 0)
        engine.validate_startup.assert_called_once_with(require_dense=False)
        engine.close.assert_called_once_with()


class SequentialRetrievalAndFusionTests(unittest.TestCase):
    def test_candidate_anchor_texts_prioritize_best_cross_query_evidence(self):
        candidate = {
            "matched_child_text": "weak original anchor",
            "matched_children": [
                {"variant_id": "Q0", "rank": 9, "text": "weak original anchor"},
                {"variant_id": "Q1", "rank": 1, "text": "decisive supporting anchor"},
            ],
        }

        self.assertEqual(
            planning.candidate_anchor_texts(candidate),
            ["decisive supporting anchor", "weak original anchor"],
        )

    def test_variants_execute_in_input_order_without_parallelism(self):
        calls: list[str] = []
        variants = [
            {"query": "original", "purpose": "original", "variant_id": "Q0"},
            {"query": "criterion", "purpose": "criterion", "variant_id": "Q1"},
            {"query": "hidden acts", "purpose": "hidden", "variant_id": "Q2"},
        ]

        def retrieve(variant):
            calls.append(variant["variant_id"])
            return [{"parent_id": variant["variant_id"], "weighted_score": 1.0}]

        results = planning.run_variants_sequentially(variants, retrieve)

        self.assertEqual(calls, ["Q0", "Q1", "Q2"])
        self.assertEqual(list(results), ["Q0", "Q1", "Q2"])

    def test_rrf_fusion_preserves_all_variant_child_anchors(self):
        variant_results = {
            "Q0": [
                {
                    "parent_id": "P",
                    "faiss_row": 10,
                    "matched_child_text": "first anchor",
                    "weighted_score": 0.9,
                }
            ],
            "Q1": [
                {
                    "parent_id": "P",
                    "faiss_row": 20,
                    "matched_child_text": "second anchor",
                    "weighted_score": 0.8,
                }
            ],
        }

        fused = planning.fuse_query_variant_rankings(variant_results)

        self.assertEqual(
            fused[0]["matched_children"],
            [
                {"variant_id": "Q0", "rank": 1, "faiss_row": 10, "text": "first anchor"},
                {"variant_id": "Q1", "rank": 1, "faiss_row": 20, "text": "second anchor"},
            ],
        )

    def test_rrf_fusion_rewards_consensus_and_preserves_query_provenance(self):
        variant_results = {
            "Q0": [
                {"parent_id": "A", "weighted_score": 0.9},
                {"parent_id": "B", "weighted_score": 0.8},
            ],
            "Q1": [
                {"parent_id": "B", "weighted_score": 0.95},
                {"parent_id": "C", "weighted_score": 0.7},
            ],
        }

        fused = planning.fuse_query_variant_rankings(variant_results, rrf_k=60)

        self.assertEqual([item["parent_id"] for item in fused], ["B", "A", "C"])
        by_id = {item["parent_id"]: item for item in fused}
        self.assertEqual(by_id["B"]["query_variant_ids"], ["Q0", "Q1"])
        self.assertEqual(by_id["B"]["rank_by_query"], {"Q0": 2, "Q1": 1})
        self.assertAlmostEqual(by_id["B"]["query_fusion_score"], 1 / 62 + 1 / 61)


if __name__ == "__main__":
    unittest.main()
