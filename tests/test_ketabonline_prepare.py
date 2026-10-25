"""Ketabonline preparation: which index heading a paragraph is filed under, and
text kept in g-table rows. Offline: the packages are small dicts shaped like
Ketabonline's JSON.
"""
from __future__ import annotations

import io
import json
from collections import Counter

from dalil.s2_prepare.ketabonline import emit_elements

SUMMARY = {"book_id": "ketabonline:test::1", "work_id": "ketabonline_work:test:1", "author_id": None,
           "language": "ar", "source_path": "raw/1.data.zip", "text_role": "book_text",
           "title_ar": "كتاب الاختبار", "package_id": 1, "review_flags": []}


def p(n: int, text: str) -> str:
    return f'<p class="g-paragraph g-rtl" id="p-{n}">{text}</p>'


def elements(pages: list[dict], index: list[dict]) -> list[dict]:
    out = io.StringIO()
    emit_elements(SUMMARY, {"pages": pages, "index": index, "parts": []}, out, set(), set(), Counter())
    return [json.loads(line) for line in out.getvalue().splitlines()]


def heading_of(els: list[dict], text: str) -> str:
    """Title of the index heading a paragraph is filed under."""
    titles = {e["refs"]["source_record"]["index_entry_ids"][0]: e["text_original"]
              for e in els if e["source_record_id"].startswith("index:")}
    para = next(e for e in els if e["element_type"] == "paragraph" and text in e["text_clean"])
    return titles[para["refs"]["source_record"]["index_entry_ids"][-1]]


def test_paragraphs_follow_the_heading_before_them_on_a_shared_page():
    # Page 2 holds the end of question 1's answer, then questions 2 and 3.
    # The index writes "س 2" where the page has "س ٢".
    pages = [
        {"id": 1, "page": 1, "content": p(1, "س ١: ما التوحيد؟") + p(2, "ج ١: إفراد الله بالعبادة")},
        {"id": 2, "page": 2, "content": p(1, "وهذا تمام جواب السؤال الأول")
         + p(2, "س ٢: ما الشرك؟") + p(3, "ج ٢: صرف العبادة لغير الله")
         + p(4, "س ٣: ما النفاق؟") + p(5, "ج ٣: إظهار الإسلام وإبطان الكفر")},
    ]
    index = [{"id": 11, "page_id": 1, "page": 1, "title": "س 1: ما التوحيد؟", "title_level": 1},
             {"id": 12, "page_id": 2, "page": 2, "title": "س 2: ما الشرك؟", "title_level": 1},
             {"id": 13, "page_id": 2, "page": 2, "title": "س 3: ما النفاق؟", "title_level": 1}]
    els = elements(pages, index)
    assert heading_of(els, "تمام جواب السؤال الأول") == "س 1: ما التوحيد؟"
    assert heading_of(els, "صرف العبادة لغير الله") == "س 2: ما الشرك؟"
    assert heading_of(els, "إظهار الإسلام") == "س 3: ما النفاق؟"


def test_an_entry_not_found_on_its_page_heads_the_whole_page_as_before():
    pages = [{"id": 1, "page": 1, "content": p(1, "نص في أول الصفحة") + p(2, "ونص بعده")}]
    index = [{"id": 11, "page_id": 1, "page": 1, "title": "عنوان لا يظهر في النص", "title_level": 1}]
    els = elements(pages, index)
    assert heading_of(els, "نص في أول الصفحة") == "عنوان لا يظهر في النص"


def test_g_table_rows_outside_paragraphs_are_kept():
    row = ('<div class="g-table"><div class="g-cell">قفا نبك من ذكرى حبيب ومنزل </div>'
           '<div class="g-border"> ... </div><div class="g-cell"> بسقط اللوى بين الدخول فحومل</div></div>')
    pages = [{"id": 1, "page": 1, "content": p(1, "قال الشاعر:") + row + p(2, "انتهى البيت")}]
    els = elements(pages, [])
    texts = [e["text_clean"] for e in els if e["element_type"] == "paragraph"]
    assert texts == ["قال الشاعر:", "قفا نبك من ذكرى حبيب ومنزل ... بسقط اللوى بين الدخول فحومل", "انتهى البيت"]


def test_headings_missing_from_the_index_head_the_text_after_them():
    # The index stops at the introduction; two questions follow only as titles in the text.
    from dalil.s4_extract.records import BuildState
    from dalil.s4_extract.sources.ketabonline import parse_ketabonline_book

    def title(t):
        return f'<span class="g-title g-rtl">{t}</span>'
    pages = [{"id": 1, "page": 1, "content": p(1, "مقدمة") + p(2, "الحمد لله وحده")},
             {"id": 2, "page": 2, "content": p(1, title("١ - ما حكم الأذان؟")) + p(2, "الأذان فرض كفاية")
              + p(3, title("٢ - ما حكم الإقامة؟")) + p(4, "الإقامة سنة مؤكدة")}]
    index = [{"id": 11, "page_id": 1, "page": 1, "title": "مقدمة", "title_level": 1}]
    profile = {**SUMMARY, "title_ar": "كتاب الاختبار", "category": "fiqh", "subcategory": None,
               "source_corpus": "ketabonline", "file_format": "ketabonline_json", "parser_profile": "ketabonline_json_parser",
               "parent_policy": "ketabonline_heading_section_parent_v1", "view_policy": "ketabonline_paragraph_pack_view_v1"}
    state = BuildState(elements=[], parents=[], views=[], quran_refs=[], isnad_edges=[], report_rows=[])
    parse_ketabonline_book(profile, elements(pages, index), state)
    body = {p["section_path"][-1]: p["text_original"] for p in state.parents}
    assert "الأذان فرض كفاية" in body["١ - ما حكم الأذان؟"]
    assert "الإقامة سنة مؤكدة" in body["٢ - ما حكم الإقامة؟"]
    assert "الأذان" not in body["مقدمة"]
