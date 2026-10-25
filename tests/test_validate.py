import json

from dalil.s4_extract.validate import scan_range


def test_ranges_check_every_row_exactly_once(tmp_path):
    # Rows of uneven length, every third one invalid; however the file is cut into
    # byte ranges, the ranges together must see each row once, at its own line.
    (tmp_path / "isnad_edges.schema.json").write_text(
        json.dumps({"type": "object", "properties": {"n": {"type": "integer"}}}), encoding="utf-8")
    rows = [{"edge_id": f"edge:{i}", "n": "bad" if i % 3 == 0 else i, "pad": "ا" * (i % 7)} for i in range(200)]
    f = tmp_path / "isnad_edges.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n", encoding="utf-8")
    size = f.stat().st_size
    whole = scan_range((f, tmp_path, "isnad_edges", 0, size, 1))
    assert len(whole["schema"]) == 67
    assert [rid for _, rid in whole["ids"]] == [r["edge_id"] for r in rows]
    for step in (1, 7, 64, 1000):
        parts = [scan_range((f, tmp_path, "isnad_edges", start, start + step, 1)) for start in range(0, size, step)]
        assert [e for p in parts for e in p["schema"]] == whole["schema"]
        assert [rid for p in parts for _, rid in p["ids"]] == [rid for _, rid in whole["ids"]]
        assert sum(p["lines"] for p in parts) == whole["lines"] == 201   # the 200 rows and the blank line
