"""Shamela footnotes: parsing the field, assigning each footnote to a parent, and
adding them without changing anything that exists without them. Offline: the
fixtures are tiny Shamela books written to tmp_path.
"""
from __future__ import annotations

import json
import re

from dalil.s4_extract.records import BuildState
from dalil.s4_extract.sources.shamela import parse_shamela
from dalil.s4_extract.sources.shamela_footnotes import (
    assign_footnotes,
    split_footnote_field,
)


def title(text: str, n: int) -> str:
    return f"<span data-type='title' id=toc-{n}>{text}</span>"


# Page 1 holds two sections, each with a marker; page 2 continues the second
# section and the second footnote; page 3 is a new section with an unmarked note.
SECTIONED_PAGES = [
    {
        "body": title("باب الأول", 1) + "\rعن أبي هريرة قال قال رسول الله صلى الله عليه وسلم من صام رمضان (¬١) غفر له ما تقدم من ذنبه\r"
        + title("باب الثاني", 2) + "\rوعن ابن عمر أن النبي صلى الله عليه وسلم نهى عن بيع الثمار (¬٢) حتى يبدو صلاحها",
        "footnotes": "(¬١) ضعيف، أخرجه الترمذي.\r(¬٢) صحيح، متفق عليه، وله طرق==",
    },
    {
        "body": "وفيه أن البيع قبل بدو الصلاح لا يصح عند جمهور أهل العلم من الفقهاء والمحدثين",
        "footnotes": "= أخرى عند أحمد.",
    },
    {
        "body": title("باب الثالث", 3) + "\rوعن أنس قال كان النبي صلى الله عليه وسلم يتعوذ من الهم والحزن والعجز والكسل",
        "footnotes": "\r<s0> (ضعيف جدا)",
    },
]


def write_book(tmp_path, pages):
    path = tmp_path / "book.jsonl"  # one path: it is part of every element
    rows = [{"page_id": 100 + i, "sequence_num": i + 1, "page_num": i + 1, **p} for i, p in enumerate(pages)]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def profile_for(source_path: str, policy: str = "shamela_heading_section_parent_v1") -> dict:
    return {
        "book_id": "book:hadith:test_author:000000000001", "work_id": "work:test", "author_id": "test_author",
        "title_ar": "كتاب الاختبار", "category": "hadith", "subcategory": None,
        "source_corpus": "shamela4", "file_format": "shamela4_jsonl", "parser_profile": "shamela4_json_parser",
        "parent_policy": policy, "view_policy": "paragraph_pack_view_v1",
        "source_path": source_path, "auxiliary_paths": [],
    }


def build(profile: dict) -> BuildState:
    state = BuildState([], [], [], [], [], [])
    parse_shamela(profile, state)
    return state


def footnotes_of(parent: dict) -> list[str]:
    _, sep, notes = parent["text_for_synthesis"].partition("\n\nالحواشي:\n")
    return notes.split("\n") if sep else []


def test_split_footnote_field_formats():
    # Shamela numbering, continuation, a multi-line note, and the trailing '=='.
    cont, notes = split_footnote_field("= تتمة السابقة.\r(¬١) أول\rسطر ثان\r(¬٢) ثان==")
    assert cont == "تتمة السابقة."
    assert notes == [(("neg", "(", "١"), "أول سطر ثان"), (("neg", "(", "٢"), "ثان")]
    # Bare numbering, including '١.' and a number glued to the word.
    _, notes = split_footnote_field("١.سورة المؤمنون\r٢سورة غافر\r٣ سورة القصص")
    assert [m[2] for m, _ in notes] == ["١", "٢", "٣"]
    assert notes[0][1] == "سورة المؤمنون"
    # A number lower than the last one is a reference inside the note, not a new note.
    _, notes = split_footnote_field("(١) أخرجه أحمد\r(٣) قال الترمذي\r(٢) من الطبعة")
    assert [m[2] for m, _ in notes] == ["١", "٣"]
    assert notes[1][1] == "قال الترمذي (٢) من الطبعة"
    # Unmarked text is a note of its own; markup is dropped.
    assert split_footnote_field("\r<s0> (ضعيف)") == ("", [(None, "(ضعيف)")])
    # A '=' continuation line is never read as a marker.
    cont, notes = split_footnote_field("= (١٢١٦)\r(١) متفق عليه")
    assert cont == "(١٢١٦)" and notes == [(("bracket", "(", "١"), "متفق عليه")]


def test_footnotes_go_to_the_section_holding_their_marker(tmp_path):
    state = build(profile_for(write_book(tmp_path, SECTIONED_PAGES)))
    by_title = {p["section_path"][-1]: p for p in state.parents}
    first, second, third = by_title["باب الأول"], by_title["باب الثاني"], by_title["باب الثالث"]

    assert footnotes_of(first) == ["(١) [عند: «الله عليه وسلم من صام رمضان»] ضعيف، أخرجه الترمذي."]
    # The continuation on page 2 joins the footnote it continues.
    assert footnotes_of(second) == ["(٢) [عند: «عليه وسلم نهى عن بيع الثمار»] صحيح، متفق عليه، وله طرق أخرى عند أحمد."]
    # No marker: the parent holding the page.
    assert footnotes_of(third) == ["(ضعيف جدا)"]

    pages = [json.loads(l) for l in open(tmp_path / "book.jsonl", encoding="utf-8")]
    methods = [(n.marker is not None, n.method) for n in assign_footnotes(pages, state.parents)]
    assert methods == [(True, "marker"), (True, "marker"), (False, "page")]


def test_footnote_views_are_packed_and_linked(tmp_path):
    state = build(profile_for(write_book(tmp_path, SECTIONED_PAGES)))
    fn_views = [v for v in state.views if v["view_type"] == "footnote"]
    assert len(fn_views) == 3
    for v in fn_views:
        parent = next(p for p in state.parents if p["parent_id"] == v["parent_id"])
        key = "view:" + parent["parent_id"][len("parent:"):]
        assert re.fullmatch(re.escape(key) + r":fn:\d{3}", v["view_id"])
        assert v["text_role"] == "footnote"
        assert v["view_id"] in parent["child_view_ids"]
        assert v["display_text"] in parent["text_for_synthesis"]
        assert "نوع النص: footnote" in v["embedding_text"]
        others = [w["order_in_parent"] for w in state.views if w["parent_id"] == v["parent_id"] and w is not v]
        assert v["order_in_parent"] > max(others)


def test_existing_records_unchanged(tmp_path):
    """With footnotes, every element and view of the footnote-free build is byte-identical,
    and parents differ only by appended text_for_synthesis and extra child views."""
    for policy, pages in (("shamela_heading_section_parent_v1", SECTIONED_PAGES),
                          ("shamela_hadith_record_parent_v1", HADITH_PAGES)):
        with_fn = build(profile_for(write_book(tmp_path, pages), policy))
        without = build(profile_for(write_book(tmp_path, [{**p, "footnotes": None} for p in pages]), policy))
        dump = lambda rows: [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]

        assert dump(with_fn.elements) == dump(without.elements)
        n = len(without.views)
        assert dump(with_fn.views[:n]) == dump(without.views)
        assert all(v["view_type"] == "footnote" for v in with_fn.views[n:]) and with_fn.views[n:]
        assert len(with_fn.parents) == len(without.parents)
        for new, old in zip(with_fn.parents, without.parents):
            assert {k: v for k, v in new.items() if k not in ("text_for_synthesis", "child_view_ids")} == \
                   {k: v for k, v in old.items() if k not in ("text_for_synthesis", "child_view_ids")}
            assert new["child_view_ids"][:len(old["child_view_ids"])] == old["child_view_ids"]
            assert new["text_for_synthesis"].startswith(old["text_for_synthesis"])


def test_pages_of_dots_keep_their_footnotes_with_the_text_before_them(tmp_path):
    # القول المفيد: the body holds the matn, the footnote field the sharh, and a long
    # sharh runs on over pages whose body is only dots. Joined to that page, the
    # matn's closing period is eaten by the leader rule, so the page is found by
    # its probe without the edge words; the dots page goes to the same parent.
    pages = [
        {"body": title("باب الأول", 1) + "\rوقوله: ﴿وَقَضَى رَبُّكَ أَلَّا تَعْبُدُوا إِلَّا إِيَّاهُ﴾ [الإسراء: ٢٣] الآية.",
         "footnotes": "قوله: وقضى أي: أمر وشرع، وهذا قضاء شرعي"},
        {"body": "." * 40, "footnotes": "ومن فوائد الآية وجوب إفراد الله بالعبادة"},
        {"body": title("باب الثاني", 2) + "\rوقول الله تعالى: ﴿فَاعْلَمْ أَنَّهُ لَا إِلَهَ إِلَّا اللَّهُ﴾ الآية",
         "footnotes": "العلم هنا اليقين"},
    ]
    state = build(profile_for(write_book(tmp_path, pages)))
    by_title = {p["section_path"][-1]: p for p in state.parents}
    assert footnotes_of(by_title["باب الأول"]) == ["قوله: وقضى أي: أمر وشرع، وهذا قضاء شرعي",
                                                 "ومن فوائد الآية وجوب إفراد الله بالعبادة"]
    assert footnotes_of(by_title["باب الثاني"]) == ["العلم هنا اليقين"]
    rows = [json.loads(l) for l in open(tmp_path / "book.jsonl", encoding="utf-8")]
    assert [n.method for n in assign_footnotes(rows, state.parents)] == ["page", "neighbour", "page"]


# One hadith per page, its grade alone in the footnote field (ضعيف الجامع الصغير).
HADITH_PAGES = [
    {"body": title("١ ", 1) + "- آتي يوم القيامة باب الجنة فيفتح لي\r(ابن النجار) عن ابن عباس.",
     "footnotes": "\r<s0> (ضعيف)"},
    {"body": title("٢ ", 2) + "- أترعوا الطسوس وخالفوا المجوس\r(هب خط فر) عن ابن عمر.",
     "footnotes": "\r<s0> (موضوع)"},
]


def test_grade_reaches_its_hadith(tmp_path):
    state = build(profile_for(write_book(tmp_path, HADITH_PAGES), "shamela_hadith_record_parent_v1"))
    grades = {p["refs"]["hadith"]["hadith_number"]: footnotes_of(p) for p in state.parents}
    assert grades == {1: ["(ضعيف)"], 2: ["(موضوع)"]}


def test_shamela_pages_are_read_in_page_id_order_not_per_volume_sequence(tmp_path, monkeypatch):
    # Two volumes whose sequence_num both restart at 1: sorting on it interleaved them.
    import json as _json

    from dalil.s4_extract.sources import shamela

    rows = [{"page_id": 10 + i, "shamela_page_id": i, "sequence_num": s, "page_num": s, "part": v}
            for i, (v, s) in enumerate([("1", 1), ("1", 2), ("2", 1), ("2", 2)], start=1)]
    f = tmp_path / "pages.jsonl"
    f.write_text("\n".join(_json.dumps(r) for r in reversed(rows)), encoding="utf-8")
    monkeypatch.setattr(shamela, "DATA_DIR", tmp_path)
    pages = shamela.read_shamela_pages("pages.jsonl")
    assert [p["part"] for p in pages] == ["1", "1", "2", "2"]
    assert [p["_order"] for p in pages] == [0, 1, 2, 3]
