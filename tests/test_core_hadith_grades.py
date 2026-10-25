"""The core hadith parser shows the standard printed number and al-Albani's grade
where a confident mapping exists, and keeps ids keyed on the source number."""
from __future__ import annotations

import json

from dalil.s4_extract import core_corpus
from dalil.s4_extract.records import BuildState

BOOK = "hadith_api::sunan_test"
PROFILE = {"book_id": BOOK, "work_id": "sunan_test", "title_ar": "سنن الاختبار", "category": "hadith",
           "subcategory": "hadith_collection", "parser_profile": "hadith_json_parser", "parent_policy": "hadith_parent_v1",
           "view_policy": "hadith_multiview_v1", "source_paths": ["sunan_test.json"], "source_corpus": "hadith_api",
           "file_format": "json"}
HADITH = [{"id": 10, "idInBook": 300, "chapterId": 1, "bookId": 6,
           "arabic": "حدثنا فلان عن فلان عن النبي صلى الله عليه وسلم قال: طلب العلم فريضة على كل مسلم"},
          {"id": 11, "idInBook": 301, "chapterId": 1, "bookId": 6,
           "arabic": "حدثنا فلان عن فلان قال: كان رسول الله صلى الله عليه وسلم يكثر الذكر"}]


def build(tmp_path, monkeypatch, grades):
    (tmp_path / "sunan_test.json").write_text(json.dumps(
        {"hadiths": HADITH, "chapters": [{"id": 1, "arabic": "باب فضل العلم"}]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core_corpus, "CLEAN_ROOT", tmp_path)
    monkeypatch.setattr(core_corpus, "hadith_grades", lambda: grades)
    state = BuildState(elements=[], parents=[], views=[], quran_refs=[], isnad_edges=[], report_rows=[])
    core_corpus.parse_hadith(PROFILE, state)
    return state


def test_standard_number_and_grade_are_applied(tmp_path, monkeypatch):
    grades = {(BOOK, "300"): {"standard_number": 224, "albani_grade": "صحيح", "albani_grade_class": "sahih",
                              "albani_grade_raw": "صحيح، المشكاة (٢١٨)"}}
    graded, plain = build(tmp_path, monkeypatch, grades).parents
    assert graded["citation"].endswith("حديث 224")
    assert graded["section_path"][-1] == "حديث 224"
    assert graded["refs"]["hadith"]["hadith_number"] == "224"
    assert graded["refs"]["hadith"]["source_hadith_number"] == "300"
    assert graded["refs"]["hadith"]["albani_grade"] == "صحيح"
    assert graded["text_for_synthesis"].endswith("حكم الألباني: صحيح، المشكاة (٢١٨)")
    assert "حكم الألباني" not in graded["text_original"]
    # ids do not move with the number, and a hadith without a mapping is unchanged
    ungraded = build(tmp_path, monkeypatch, {}).parents[0]
    assert graded["parent_id"] == ungraded["parent_id"]
    assert plain["citation"].endswith("حديث 301") and "حكم الألباني" not in plain["text_for_synthesis"]
