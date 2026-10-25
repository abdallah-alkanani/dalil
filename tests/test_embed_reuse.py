"""dalil.s6_embed.reuse: a rebuilt corpus takes stored vectors for unchanged texts and
new ones only for the rest, in the order dalil.s7_index.append reads views."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from test_end_to_end import _StubTokenizer, _write_corpus  # noqa: E402

from dalil.s6_embed import reuse  # noqa: E402
from dalil.s7_index import append, schema  # noqa: E402


def test_plan_and_assemble_reuse_unchanged_texts(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubTokenizer())))
    db = tmp_path / "db"
    schema.create_stores(db)
    (tmp_path / "old").mkdir()
    corpus, emb = _write_corpus(tmp_path / "old")
    assert append.main(["--corpus", str(corpus), "--embeddings", str(emb), "--db", str(db), "--no-backup"]) == 0

    # The rebuild renames every view, edits one text and adds a view (twice the same text).
    views = [json.loads(line) for line in (corpus / "retrieval_views.jsonl").read_text().splitlines()]
    rebuilt = tmp_path / "rebuilt"
    rebuilt.mkdir()
    new_views = []
    for i, v in enumerate(views + [dict(views[0]), dict(views[0])]):
        v = dict(v, view_id=f"view:rebuilt:{i}")
        if i == 1:
            v["embedding_text"] += " وزيادة"
        if i >= 3:
            v["embedding_text"] = "الكتاب: كتاب الاختبار\nالنص: نص جديد مكرر"
        new_views.append(v)
    (rebuilt / "retrieval_views.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in new_views))

    to_embed = tmp_path / "to_embed.jsonl"
    counts = reuse.plan(rebuilt, db, to_embed, set(), 3)
    assert counts == {"views": 5, "reused": 2, "to_embed_views": 3, "to_embed_texts": 2}
    planned = [json.loads(line)["view_id"] for line in to_embed.read_text().splitlines()]
    assert planned == ["view:rebuilt:1", "view:rebuilt:3"]

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    new_vecs = np.random.default_rng(1).standard_normal((2, 4096)).astype(np.float16)
    np.save(fresh / "shard_00000.npy", new_vecs)
    (fresh / "shard_00000.ids.json").write_text(json.dumps(planned))

    out = tmp_path / "embeddings"
    counts = reuse.assemble(rebuilt, db, [fresh], [to_embed], out, set(), 3)
    assert counts == {"views": 5, "reused": 2, "new": 3}
    assert json.loads((out / "shard_00000.ids.json").read_text()) == [v["view_id"] for v in new_views]
    vecs = np.load(out / "shard_00000.npy")
    index = faiss.read_index(str(db / "dense_faiss_ar.index"))
    np.testing.assert_array_equal(vecs[0], index.reconstruct(0))
    np.testing.assert_array_equal(vecs[2], index.reconstruct(2))
    np.testing.assert_array_equal(vecs[1], new_vecs[0].astype(np.float32))
    np.testing.assert_array_equal(vecs[3], vecs[4])


def test_assemble_fails_on_a_view_without_a_vector(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubTokenizer())))
    db = tmp_path / "db"
    schema.create_stores(db)
    (tmp_path / "old").mkdir()
    corpus, emb = _write_corpus(tmp_path / "old")
    assert append.main(["--corpus", str(corpus), "--embeddings", str(emb), "--db", str(db), "--no-backup"]) == 0
    rebuilt = tmp_path / "rebuilt"
    rebuilt.mkdir()
    v = json.loads((corpus / "retrieval_views.jsonl").read_text().splitlines()[0])
    v["embedding_text"] += " نص لم يضمن"
    (rebuilt / "retrieval_views.jsonl").write_text(json.dumps(v, ensure_ascii=False) + "\n")
    with pytest.raises(SystemExit, match="no vector"):
        reuse.assemble(rebuilt, db, [], [], tmp_path / "embeddings", set(), 3)
