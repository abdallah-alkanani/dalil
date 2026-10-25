"""Ibn Baz fatwa record datasets (JSONL).
"""
from __future__ import annotations

import json
from typing import Any

from dalil.paths import DATA_DIR

from ..quran_refs import extract_all_quran_refs
from ..records import (
    BuildState,
    add_quran_ref,
    citation,
    clean_text,
    default_refs,
    split_paragraph_packs,
    stable_hash,
)
from ..sections import (
    FATWA_ANSWER_WORDS,
    hold_flags,
    make_element_fixed,
    make_parent_fixed,
    make_view_fixed,
    sanitize_refs,
)
from ..text import sanitize_markup, strip_source_markers


def parse_ibn_baz(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["clean_source_path"] or profile["source_path"]
    records: list[dict[str, Any]] = []
    with (DATA_DIR / source_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    book = make_element_fixed(
        profile,
        element_id=f"elem:{profile['book_id']}:book",
        source_path=source_path,
        source_record_id="book",
        element_type="book",
        level=1,
        parent_element_id=None,
        ancestor_ids=[],
        order_in_parent=0,
        section_path=[profile.get("title_ar") or profile["book_id"]],
        section_path_ids=[f"book:{profile['book_id']}"],
        text="",
        refs=default_refs(),
    )
    state.elements.append(book)
    order = 0
    for rec in records:
        rid = str(rec.get("record_id") or rec.get("id") or order + 1)
        title = clean_text(strip_source_markers(sanitize_markup(rec.get("title_clean") or "")))
        question = clean_text(strip_source_markers(sanitize_markup(rec.get("question_clean") or "")))
        answer = clean_text(strip_source_markers(sanitize_markup(rec.get("answer_clean") or "")))
        if not title and not question and not answer:
            continue
        order += 1
        flags = hold_flags(profile)
        answerless = not answer
        if answerless:
            flags.append("answerless_record")
        if not question and answer:
            flags.append("questionless_record")
        refs = default_refs()
        refs["source_record"] = {
            "record_id": rid,
            "record_index": order,
            "page_id": None,
            "page_number": None,
            "page_index": None,
            "part_id": None,
            "part_name": None,
            "html_id": None,
            "html_classes": [],
            "index_entry_ids": [],
            "title_level": None,
            "page_field_keys": [],
            "source_package_id": None,
        }
        elem_id = f"elem:{profile['book_id']}:fatwa:{stable_hash(rid)}"
        parent_id = f"parent:{profile['book_id']}:fatwa:{stable_hash(rid)}"
        section_path = [profile.get("title_ar") or profile["book_id"]]
        section_ids = [f"book:{profile['book_id']}"]
        if title:
            section_path.append(title[:120])
            section_ids.append(f"fatwa:{rid}")
        full_text = "\n".join(x for x in (title, question, answer) if x)
        state.elements.append(
            make_element_fixed(
                profile,
                element_id=elem_id,
                source_path=source_path,
                source_record_id=f"fatwa:{rid}",
                element_type="fatwa",
                level=2,
                parent_element_id=book["element_id"],
                ancestor_ids=[book["element_id"]],
                order_in_parent=order,
                section_path=section_path,
                section_path_ids=section_ids,
                text=full_text,
                refs=refs,
                quality_flags=flags,
                parse_confidence=0.95,
            )
        )
        view_ids: list[str] = []
        if title:
            vid = f"view:{profile['book_id']}:fatwa:{stable_hash(rid)}:title"
            state.views.append(
                make_view_fixed(
                    profile, view_id=vid, parent_id=parent_id, view_type="fatwa_title",
                    text_role="title", order_in_parent=0, section_path=section_path,
                    section_path_ids=section_ids, refs=sanitize_refs(refs), text=title, quality_flags=flags,
                )
            )
            view_ids.append(vid)
        if question:
            vid = f"view:{profile['book_id']}:fatwa:{stable_hash(rid)}:question"
            state.views.append(
                make_view_fixed(
                    profile, view_id=vid, parent_id=parent_id, view_type="fatwa_question",
                    text_role="question", order_in_parent=1, section_path=section_path,
                    section_path_ids=section_ids, refs=sanitize_refs(refs), text=question, quality_flags=flags,
                )
            )
            view_ids.append(vid)
        if answer:
            parts = split_paragraph_packs(answer, FATWA_ANSWER_WORDS)
            for idx, part in enumerate(parts, start=1):
                vid = f"view:{profile['book_id']}:fatwa:{stable_hash(rid)}:answer:{idx:03d}"
                state.views.append(
                    make_view_fixed(
                        profile, view_id=vid, parent_id=parent_id,
                        view_type="fatwa_answer" if len(parts) == 1 else "fatwa_answer_clause",
                        text_role="answer", order_in_parent=2 + idx, section_path=section_path,
                        section_path_ids=section_ids, refs=sanitize_refs(refs), text=part,
                        quality_flags=flags + (["long_answer_clause_view"] if len(parts) > 1 else []),
                    )
                )
                view_ids.append(vid)
        state.parents.append(
            make_parent_fixed(
                profile,
                parent_id=parent_id,
                parent_type="fatwa",
                source_element_ids=[elem_id],
                ancestor_ids=[book["element_id"], elem_id],
                section_path=section_path,
                section_path_ids=section_ids,
                refs=sanitize_refs(refs),
                text=full_text,
                citation=citation(profile, title[:100] if title else f"فتوى {rid}"),
                child_view_ids=view_ids,
                quality_flags=flags,
            )
        )
        for surah, ayah_start, ayah_end in extract_all_quran_refs(full_text):
            add_quran_ref(
                state, source_kind="parent", source_id=parent_id, book_id=profile["book_id"],
                surah=surah, ayah_start=ayah_start, ayah_end=ayah_end,
                match_method="explicit_ref", confidence=0.75, matched_text=title,
            )
