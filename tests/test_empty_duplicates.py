"""An empty parent that repeats the next parent's title is dropped with its views;
an empty chapter heading over different sections is kept."""
from __future__ import annotations

from dalil.s4_extract.collection import drop_empty_duplicates
from dalil.s4_extract.records import BuildState


def parent(pid: str, title: str, text: str) -> dict:
    return {"parent_id": pid, "section_path": ["كتاب", title], "text_original": text,
            "child_view_ids": [f"{pid}:title"]}


def test_drops_only_the_empty_copy_of_the_next_title():
    parents = [parent("p0", "الفصل الأول : في التوحيد", ""),
               parent("p1", "س/200 كم أقسام الصفات؟", ""),
               parent("p2", "س/٢٠٠ كم أقسام الصفات؟", "ج/ تنقسم إلى قسمين"),
               parent("p3", "الخاتمة", "")]
    state = BuildState([], parents, [{"view_id": f"p{i}:title"} for i in range(4)], [], [], [])
    drop_empty_duplicates(state)
    assert [p["parent_id"] for p in state.parents] == ["p0", "p2", "p3"]
    assert [v["view_id"] for v in state.views] == ["p0:title", "p2:title", "p3:title"]
