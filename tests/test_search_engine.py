"""Engine behaviour that needs no database, network or API key."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dalil.s8_search import engine as search_engine
from dalil.s8_search.dense import DenseIndex
from dalil.s8_search.filters import build_bm25_where, normalize_filters
from dalil.s8_search.ranking import normalize_arabic_for_search, sanitize_fts_query


def sample_child():
    return {
        "view_id": "view-1",
        "parent_id": "parent-1",
        "book_id": "quran_uthmani_ar",
        "category": "quran",
        "view_type": "ayah",
        "text_role": "source",
        "display_text": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
        "embedding_text": "القرآن الكريم إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
        "section_path": [],
        "section_path_ids": [],
        "refs": {"quran": {"surah": 1, "ayah": 1}},
        "metadata": {},
    }


def sample_parent():
    return {
        "parent_id": "parent-1",
        "book_id": "quran_uthmani_ar",
        "category": "quran",
        "citation": "اختبار 1:1",
        "section_path": [],
        "refs": {"quran": {"surah": 1, "ayah": 1}},
        "text_for_synthesis": "مقدمة طويلة لا تحتوي العبارة المطابقة في بدايتها",
        "text_original": "",
        "metadata": {},
    }


def test_arabic_search_normalization_preserves_source_and_unifies_variants():
    source = "إِنَّمَا الـأَعْمَالُ ١٢۳‏"
    normalized = normalize_arabic_for_search(source)
    assert source == "إِنَّمَا الـأَعْمَالُ ١٢۳‏", "normalization must not mutate evidence"
    assert normalized == "انما الاعمال 123"


def test_fts_query_uses_normalized_unique_terms():
    normalized = normalize_arabic_for_search("أَحْكَامُ الإيمان، احكام")
    assert sanitize_fts_query(normalized) == "احكام OR الايمان"


def test_result_keeps_matched_child_for_reranking_and_parent_for_synthesis():
    hit = {"child": sample_child(), "best_faiss_row": 7, "dense_score": 0.8, "child_scores": [0.8], "child_rows": [7]}
    result = search_engine.build_result_from_parent(sample_parent(), hit)
    assert result["matched_child_text"] == sample_child()["display_text"]
    assert result["text_for_synthesis"] == sample_parent()["text_for_synthesis"]


def test_bm25_only_query_does_not_embed_or_load_faiss(tmp_path):
    class LexicalOnlyEngine(search_engine.QueryEngine):
        def validate_startup(self, *, require_dense=True):
            assert not require_dense, "BM25-only startup must not require the dense index"

        def embed_query(self, question):
            raise AssertionError("BM25-only retrieval called the embedding API")

        def load_faiss_index(self):
            raise AssertionError("BM25-only retrieval loaded FAISS")

        def compute_allowed_faiss_rows(self, filters):
            return None

        def retrieve_bm25_parent_hits(self, **kwargs):
            return {
                "parent-1": {
                    "best_score": 1.0, "bm25_score": -2.0, "bm25_rank": 1, "best_faiss_row": 7,
                    "child": sample_child(), "child_scores": [1.0], "child_rows": [7],
                    "retrieval_sources": ["bm25"],
                }
            }

        def get_parents_batch(self, parent_ids):
            return {"parent-1": sample_parent()}

        def choose_boost_categories(self, question, override):
            return None

    engine = LexicalOnlyEngine(tmp_path, api_key="")
    output = engine.query("الأعمال بالنيات", n_results=1, retrieval_mode="bm25", use_listwise_rerank=False)
    assert len(output["results"]) == 1
    assert output["results"][0]["retrieval_sources"] == ["bm25"]


def test_listwise_prompt_uses_matched_child_not_parent_prefix(tmp_path, monkeypatch):
    captured = {}

    class Response:
        def json(self):
            return {"choices": [{"message": {"content": "1"}}]}

    def fake_request(*args, **kwargs):
        captured.update(kwargs["json_body"])
        return Response()

    monkeypatch.setattr(search_engine, "request_with_retries", fake_request)
    engine = search_engine.QueryEngine(tmp_path, api_key="test")
    candidate = {
        "parent_id": "parent-1",
        "citation": "اختبار",
        "matched_child_text": "النص المطابق الفعلي",
        "text_for_synthesis": "بداية الوالد غير المطابقة",
    }
    result = engine.llm_listwise_rerank("السؤال", [candidate], top_k=1)
    prompt = captured["messages"][0]["content"]
    assert "النص المطابق الفعلي" in prompt, "reranker must see the retrieved child"
    assert "بداية الوالد غير المطابقة" not in prompt, "parent prefix must not replace matched evidence"
    assert result[0]["parent_id"] == "parent-1"


def test_filter_builders_cover_exact_quran_lookup():
    filters = normalize_filters({"quran_surah": ["2"], "quran_ayah": 255})
    where_sql, params = build_bm25_where(filters)
    assert "quran_surah IN" in where_sql, "surah must be an indexed lexical filter"
    assert "quran_ayah_start" in where_sql, "ayah containment must be enforced"
    assert params[-2:] == [255, 255]


@pytest.mark.parametrize("ntotal", [5, 70_001])
def test_streamed_dense_search_matches_faiss(tmp_path, monkeypatch, ntotal):
    """Block-streamed exact search returns what IndexFlatIP.search returns,
    including across block boundaries and for exact ties.

    Scores are compared to float32 rounding: at small d, FAISS's multi-row
    kernels round differently depending on how rows are split across threads,
    and FAISS's own search changes in the last bit with the thread count. On the
    production index (d=4096) the streamed scores were bit-identical.
    """
    faiss = pytest.importorskip("faiss")
    monkeypatch.setattr("dalil.s8_search.dense.BLOCK_ROWS", 4096)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((ntotal, 16)).astype(np.float32)
    vectors[ntotal // 2] = vectors[1]  # an exact duplicate, so scores tie
    index = faiss.IndexFlatIP(16)
    index.add(vectors)
    path = tmp_path / "flat.index"
    faiss.write_index(index, str(path))

    queries = np.vstack([vectors[1], rng.standard_normal(16).astype(np.float32)])
    dense = DenseIndex(Path(path))
    k = min(50, ntotal)
    scores, ids = dense.search(queries, k)
    for qi in range(len(queries)):
        ref_scores, ref_ids = index.search(queries[qi:qi + 1], k)
        assert np.array_equal(ids[qi], ref_ids[0])
        np.testing.assert_allclose(scores[qi], ref_scores[0], rtol=1e-6)


def test_dense_only_candidates_rank_by_similarity():
    """Dense-only results have hybrid_score=None; ranking must not crash on it."""
    hit = {"child": sample_child(), "best_faiss_row": 7, "dense_score": 0.8, "child_scores": [0.8], "child_rows": [7]}
    dense_only = search_engine.build_result_from_parent(sample_parent(), hit)
    assert dense_only["hybrid_score"] is None
    assert search_engine.candidate_score(dense_only) == 0.8
    assert search_engine.candidate_score({"hybrid_score": 0.02, "semantic_sim": 0.9}) == 0.02
