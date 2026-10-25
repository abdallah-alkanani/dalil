"""Offline end-to-end: empty stores -> append a small corpus -> navigation -> search.

Vectors are random and the Qwen tokenizer is stubbed, so nothing is downloaded.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

pytest.importorskip("faiss")

from dalil.s7_index import append, navigation, schema  # noqa: E402
from dalil.s8_search.engine import QueryEngine  # noqa: E402

BOOK = "book:fiqh:test_author:000000000001"
PARAGRAPHS = [
    "باب الربا: الربا محرم بالكتاب والسنة والإجماع، وهو الزيادة في أشياء مخصوصة.",
    "باب الصدقة: فضل الصدقة عظيم، والصدقة تطفئ الخطيئة كما يطفئ الماء النار.",
    "باب الصلاة: صلاة الجماعة واجبة على الرجال القادرين في أصح قولي العلماء.",
]


class _StubTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def _write_corpus(root):
    parents, views = [], []
    for i, text in enumerate(PARAGRAPHS):
        pid, vid = f"parent:{BOOK}:{i:012d}", f"view:{BOOK}:{i:012d}:001"
        section = ["كتاب الاختبار", text.split(":")[0]]
        parents.append({
            "parent_id": pid, "book_id": BOOK, "category": "fiqh", "subcategory": None,
            "parent_type": "heading_section", "citation": f"كتاب الاختبار، {section[1]}",
            "section_path": section, "section_path_ids": [f"book:{BOOK}", f"s{i}"],
            "refs": {}, "child_view_ids": [vid], "text_for_synthesis": text, "text_original": text,
            "source_element_ids": [f"elem:{BOOK}:{i}"],
        })
        views.append({
            "view_id": vid, "parent_id": pid, "book_id": BOOK, "category": "fiqh", "subcategory": None,
            "view_type": "paragraph_pack", "text_role": "paragraph", "section_path": section,
            "section_path_ids": [f"book:{BOOK}", f"s{i}"], "refs": {}, "token_counts": [],
            "display_text": text, "embedding_text": f"الكتاب: كتاب الاختبار\nالنص: {text}",
            "source_corpus": "test", "work_id": "work:test", "author_id": "test_author",
            "file_format": "txt", "parser_profile": "test_parser", "quality_flags": [],
        })
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "parents.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in parents))
    elements = [{"element_id": e, "book_id": BOOK, "refs": {}} for p in parents for e in p["source_element_ids"]]
    (corpus / "elements.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in elements))
    (corpus / "retrieval_views.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in views))
    emb = root / "embeddings"
    emb.mkdir()
    vectors = np.random.default_rng(0).standard_normal((len(views), 4096)).astype(np.float16)
    np.save(emb / "shard_00000.npy", vectors)
    (emb / "shard_00000.ids.json").write_text(json.dumps([v["view_id"] for v in views]))
    return corpus, emb


def test_bootstrap_append_and_search(tmp_path, monkeypatch):
    db = tmp_path / "db"
    schema.create_stores(db)
    corpus, emb = _write_corpus(tmp_path)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubTokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    args = ["--corpus", str(corpus), "--embeddings", str(emb), "--db", str(db), "--no-backup"]
    assert append.main(args) == 0
    assert navigation.main(["--db", str(db), "--corpus", str(corpus), "--no-backup"]) == 0

    engine = QueryEngine(db, api_key="")
    try:
        engine.validate_startup(require_dense=True)
        lexical = engine.query("حكم الربا", retrieval_mode="bm25", n_results=1, boost_override="all")
        assert lexical["results"][0]["parent_id"].endswith("000000000000")

        target = engine.load_faiss_index().reconstruct(1)
        monkeypatch.setattr(engine, "embed_query", lambda q: target / np.linalg.norm(target))
        dense = engine.query("فضل الصدقة", retrieval_mode="dense", n_results=3, boost_override="all")
        assert dense["results"][0]["parent_id"].endswith("000000000001")
        hybrid = engine.query("فضل الصدقة", n_results=3, boost_override="all", expand_neighbours=1)
        assert hybrid["results"][0]["parent_id"].endswith("000000000001")
        assert {n["offset"] for n in hybrid["results"][0]["neighbours"]} == {-1, 1}
    finally:
        engine.close()


def test_prune_drops_books_and_views_into_a_new_store(tmp_path, monkeypatch):
    from dalil.s7_index import prune

    db = tmp_path / "db"
    schema.create_stores(db)
    corpus, emb = _write_corpus(tmp_path)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubTokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    assert append.main(["--corpus", str(corpus), "--embeddings", str(emb), "--db", str(db), "--no-backup"]) == 0
    old_vectors = QueryEngine(db, api_key="").load_faiss_index()

    # Drop view 0 (and so its parent); keep the rest.
    counts = prune.prune(db, tmp_path / "new", drop_books=[], drop_rows=[0])
    assert counts["views_after"] == 2 and counts["parents_after"] == 2

    engine = QueryEngine(tmp_path / "new", api_key="")
    try:
        engine.validate_startup(require_dense=True)
        new_vectors = engine.load_faiss_index()
        assert np.array_equal(new_vectors.reconstruct(0), old_vectors.reconstruct(1))
        assert np.array_equal(new_vectors.reconstruct(1), old_vectors.reconstruct(2))
        hits = engine.query("الربا", retrieval_mode="bm25", n_results=3, boost_override="all")["results"]
        assert all(not h["parent_id"].endswith("000000000000") for h in hits)
        lexical = engine.query("الصدقة", retrieval_mode="bm25", n_results=1, boost_override="all")
        assert lexical["results"][0]["parent_id"].endswith("000000000001")
    finally:
        engine.close()

    # A whole book.
    counts = prune.prune(db, tmp_path / "none", drop_books=[BOOK], drop_rows=[])
    assert counts["views_after"] == 0 and counts["parents_after"] == 0
