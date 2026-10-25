"""Build the original 78-book core corpus (Quran, the hadith collections, the major
tafsir works) from their sources under ``data/books/original/`` into elements / parents / retrieval_views.

Source structure wins over token limits, and former physical delimiters are
stripped before semantic parsing. It does not embed or index.

    python -m dalil.s4_extract.core_corpus --output-root /tmp/core_build
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from dalil.paths import BUILD_DIR, DATA_DIR, HADITH_GRADES_PATH, RECIPE_BOOKS

from .quran_refs import extract_numeric_quran_ref
from .records import (
    BuildState,
    add_quran_ref,
    citation,
    clean_text,
    default_refs,
    make_element,
    make_parent,
    make_view,
    narrator_edges_from_sanad,
    read_jsonl,
    safe_num,
    split_hadith_sanad_matn,
    split_paragraph_packs,
    stable_hash,
    write_json,
    write_jsonl,
)

CLEAN_ROOT = DATA_DIR  # profiles' source_paths are relative to the data directory
PROFILE_PATH = RECIPE_BOOKS / "core_profiles.jsonl"
OUT_ROOT = BUILD_DIR / "structured" / "core"
FORMER_DELIMITER_RE = re.compile(r"^\s*(?:P" + r"ageV\S*.*|P" + r"ageV\s+.*)$")
OPENITI_HEADING_RE = re.compile(r"^\s*(?:###\s*.*|AUTO\s+.*|CHECK\s+\[.*\]|\$\s+.+|=+\s*.+)$")
ARABIC_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:كتاب|باب|فصل|قاعدة|مسألة|المسألة|القاعدة|الفرع|الدليل|الوجه)\b.*"
    r"|\{\s*(?:باب|فصل|كتاب)\b.*\}"
    r"|\[\s*[^]]{3,90}\s*\]"
    r")\s*$"
)
ORDINAL_WORDS = (
    "الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع",
    "الثامن", "التاسع", "العاشر", "الأولى", "الثانية", "الثالثة",
    "الرابعة", "الخامسة", "السادسة", "السابعة", "الثامنة", "التاسعة",
    "العاشرة",
)
NUMBERED_LINE_RE = re.compile(r"^\s*([0-9٠-٩]{1,4})\s*(?:[)\-.،]|[-—])\s*(.+)$")
IRAB_ANCHOR_RE = re.compile(r"(?:\[\[?\s*سورة\s*\((\d{1,3})\)\s*:\s*آية\s*(\d{1,3})\s*\]?\]|(?:قَالَ|قال|قوله|قَوْلُهُ|قَوْله)\s+تَعَالَى)")
BRACE_QURAN_RE = re.compile(r"[﴿{]([^﴾}]{2,400})[﴾}]")


def strip_former_delimiters(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if FORMER_DELIMITER_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def book_root(profile: dict[str, Any], source_path: str) -> dict[str, Any]:
    return make_element(
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


def split_sections_by_lines(text: str, profile: dict[str, Any]) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "مقدمة"
    current_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        is_heading = False
        if profile["parser_profile"] == "openiti_txt_parser":
            is_heading = bool(OPENITI_HEADING_RE.match(stripped) or ARABIC_HEADING_RE.match(stripped))
        elif profile["parser_profile"] == "saaid_txt_parser":
            is_heading = bool(ARABIC_HEADING_RE.match(stripped)) and len(stripped) < 140
        elif profile["parser_profile"] == "shamela_json_records_parser":
            is_heading = bool(stripped and len(stripped) < 100 and not stripped.endswith(("،", ".", "؟")))
        if is_heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return [(title, "\n".join(chunk).strip()) for title, chunk in sections if "\n".join(chunk).strip()]


def split_numbered_items(text: str) -> list[tuple[int, str]]:
    items: list[tuple[int, list[str]]] = []
    current_num: int | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        m = NUMBERED_LINE_RE.match(line)
        if m:
            if current_num is not None and current_lines:
                items.append((current_num, current_lines))
            current_num = safe_num(m.group(1))
            current_lines = [m.group(2).strip()]
        elif current_num is not None:
            current_lines.append(line.strip())
    if current_num is not None and current_lines:
        items.append((current_num, current_lines))
    return [(num, "\n".join(lines).strip()) for num, lines in items if "\n".join(lines).strip()]


def parse_quran(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_paths"][0]
    data = json.loads((CLEAN_ROOT / source_path).read_text(encoding="utf-8"))
    book = book_root(profile, source_path)
    state.elements.append(book)
    for surah in data:
        sid = int(surah["id"])
        surah_name = surah["name"]
        surah_id = f"elem:quran:surah:{sid}"
        state.elements.append(
            make_element(
                profile,
                element_id=surah_id,
                source_path=source_path,
                source_record_id=f"surah:{sid}",
                element_type="surah",
                level=2,
                parent_element_id=book["element_id"],
                ancestor_ids=[book["element_id"]],
                order_in_parent=sid,
                section_path=["القرآن", f"سورة {surah_name}"],
                section_path_ids=["quran", f"surah:{sid}"],
                text="",
                refs=default_refs(),
            )
        )
        for verse in surah.get("verses", []):
            ayah = int(verse["id"])
            text = verse.get("text", "")
            refs = default_refs()
            refs["quran"] = {"surah": sid, "ayah_start": ayah, "ayah_end": ayah}
            elem_id = f"elem:quran:{sid}:{ayah}"
            parent_id = f"parent:quran:{sid}:{ayah}"
            view_id = f"view:quran:{sid}:{ayah}:ayah_exact"
            section_path = ["القرآن", f"سورة {surah_name}", f"آية {ayah}"]
            section_ids = ["quran", f"surah:{sid}", f"ayah:{ayah}"]
            state.elements.append(
                make_element(
                    profile,
                    element_id=elem_id,
                    source_path=source_path,
                    source_record_id=f"surah:{sid}:ayah:{ayah}",
                    element_type="quran_ayah",
                    level=3,
                    parent_element_id=surah_id,
                    ancestor_ids=[book["element_id"], surah_id],
                    order_in_parent=ayah,
                    section_path=section_path,
                    section_path_ids=section_ids,
                    text=text,
                    refs=refs,
                )
            )
            state.views.append(
                make_view(
                    profile,
                    view_id=view_id,
                    parent_id=parent_id,
                    view_type="ayah_exact",
                    text_role="ayah",
                    order_in_parent=0,
                    section_path=section_path,
                    section_path_ids=section_ids,
                    refs=refs,
                    text=text,
                )
            )
            state.parents.append(
                make_parent(
                    profile,
                    parent_id=parent_id,
                    parent_type="quran_ayah",
                    source_element_ids=[elem_id],
                    ancestor_ids=[surah_id],
                    section_path=section_path,
                    section_path_ids=section_ids,
                    refs=refs,
                    text=text,
                    citation=citation(profile, f"سورة {surah_name}، آية {ayah}"),
                    child_view_ids=[view_id],
                )
            )
            add_quran_ref(
                state,
                source_kind="parent",
                source_id=parent_id,
                book_id=profile["book_id"],
                surah=sid,
                ayah_start=ayah,
                ayah_end=ayah,
                match_method="source_structural_ref",
                confidence=1.0,
                matched_text=text,
            )


@lru_cache(maxsize=1)
def hadith_grades(path: Path = HADITH_GRADES_PATH) -> dict[tuple[str, str], dict[str, Any]]:
    """(book_id, source hadith number) -> the confident row of ``sunan_grades.jsonl``.

    The source numbers its hadith with a running id that is not the printed number
    (Ibn Majah is 266 ahead for most of the book; Nasa'i differs in 5,229 places).
    A confident row gives the standard number, found by matching the text against
    al-Albani's graded Sunan, and his grade. Rows that are not confident are left out.
    """
    if not path.exists():
        return {}
    return {(r["book_id"], str(r["hadith_api_number"])): r for r in read_jsonl(path) if r.get("status") == "confident"}


def parse_hadith(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_paths"][0]
    path = CLEAN_ROOT / source_path
    book = book_root(profile, source_path)
    state.elements.append(book)
    records: list[dict[str, Any]] = []
    if profile["parser_profile"] == "musnad_csv_parser":
        with path.open(encoding="utf-8") as f:
            for idx, row in enumerate(csv.reader(f), start=1):
                if len(row) >= 2:
                    records.append({"id": row[0], "arabic": " ".join(row[1:]), "chapter": None})
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "hadiths" in data:
            chapters = {str(c.get("id")): c for c in data.get("chapters", [])}
            for h in data.get("hadiths", []):
                cid = str(h.get("chapterId"))
                records.append({**h, "chapter": chapters.get(cid)})
        elif isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                text = item.get("arabic") or item.get("hadith") or item.get("text", "")
                records.append({"id": item.get("id", idx), "idInBook": idx, "arabic": text, "chapter": None})
    for order, rec in enumerate(records, start=1):
        hid = str(rec.get("idInBook") or rec.get("id") or order)
        text = rec.get("arabic") or rec.get("hadith") or ""
        if not clean_text(text):
            continue
        sanad, matn, flags = split_hadith_sanad_matn(text)
        refs = default_refs()
        refs["hadith"] = {
            "collection": profile.get("work_id") or profile["book_id"],
            "book_number": safe_num(rec.get("bookId"), None),
            "chapter_id": rec.get("chapterId"),
            "hadith_number": hid,
            "id_in_book": rec.get("idInBook"),
        }
        # Ids stay keyed on the source number; what is shown and filtered is the printed one.
        number = hid
        grade = hadith_grades().get((profile["book_id"], hid))
        if grade:
            nums = [grade["standard_number"]] if grade.get("standard_number") is not None else grade.get("standard_numbers") or []
            number = "، ".join(str(n) for n in nums) or hid
            refs["hadith"].update(hadith_number=number, source_hadith_number=hid,
                                  albani_grade=grade.get("albani_grade"), albani_grade_class=grade.get("albani_grade_class"))
        chapter_title = ""
        if rec.get("chapter"):
            chapter_title = rec["chapter"].get("arabic") or rec["chapter"].get("title") or ""
        section_path = [profile.get("title_ar") or profile["book_id"]]
        section_ids = [f"book:{profile['book_id']}"]
        if chapter_title:
            section_path.append(chapter_title)
            section_ids.append(f"chapter:{rec.get('chapterId')}")
        section_path.append(f"حديث {number}")
        section_ids.append(f"hadith:{hid}")
        elem_id = f"elem:{profile['book_id']}:hadith:{stable_hash(hid)}"
        parent_id = f"parent:{profile['book_id']}:hadith:{stable_hash(hid)}"
        views: list[dict[str, Any]] = []
        full_view_id = f"view:{profile['book_id']}:hadith:{stable_hash(hid)}:full"
        if len(clean_text(text).split()) <= 800:
            views.append(make_view(profile, view_id=full_view_id, parent_id=parent_id, view_type="hadith_full", text_role="full", order_in_parent=0, section_path=section_path, section_path_ids=section_ids, refs=refs, text=text, quality_flags=flags))
        else:
            for idx, part in enumerate(split_paragraph_packs(text, 360), start=1):
                views.append(make_view(profile, view_id=f"{full_view_id}:clause:{idx:03d}", parent_id=parent_id, view_type="hadith_clause", text_role="full", order_in_parent=idx, section_path=section_path, section_path_ids=section_ids, refs=refs, text=part, quality_flags=flags + ["long_hadith_full_clause_view"]))
        if sanad:
            sanad_base_id = f"view:{profile['book_id']}:hadith:{stable_hash(hid)}:sanad"
            sanad_parts = split_paragraph_packs(sanad, 360) if len(clean_text(sanad).split()) > 800 else [sanad]
            for idx, part in enumerate(sanad_parts, start=1):
                suffix = f":{idx:03d}" if len(sanad_parts) > 1 else ""
                views.append(make_view(profile, view_id=f"{sanad_base_id}{suffix}", parent_id=parent_id, view_type="hadith_sanad", text_role="sanad", order_in_parent=100 + idx, section_path=section_path, section_path_ids=section_ids, refs=refs, text=part, quality_flags=flags + (["long_sanad_clause_view"] if len(sanad_parts) > 1 else [])))
        if matn:
            matn_view_id = f"view:{profile['book_id']}:hadith:{stable_hash(hid)}:matn"
            if len(clean_text(matn).split()) <= 800:
                views.append(make_view(profile, view_id=matn_view_id, parent_id=parent_id, view_type="hadith_matn", text_role="matn", order_in_parent=200, section_path=section_path, section_path_ids=section_ids, refs=refs, text=matn, quality_flags=flags))
            else:
                for idx, part in enumerate(split_paragraph_packs(matn, 360), start=1):
                    views.append(make_view(profile, view_id=f"{matn_view_id}:clause:{idx:03d}", parent_id=parent_id, view_type="hadith_clause", text_role="matn", order_in_parent=200 + idx, section_path=section_path, section_path_ids=section_ids, refs=refs, text=part, quality_flags=flags + ["long_matn_clause_view"]))
        state.elements.append(make_element(profile, element_id=elem_id, source_path=source_path, source_record_id=f"hadith:{hid}", element_type="hadith", level=2, parent_element_id=book["element_id"], ancestor_ids=[book["element_id"]], order_in_parent=order, section_path=section_path, section_path_ids=section_ids, text=text, refs=refs, quality_flags=flags, parse_confidence=0.9 if flags else 1.0))
        state.views.extend(views)
        parent = make_parent(profile, parent_id=parent_id, parent_type="hadith", source_element_ids=[elem_id], ancestor_ids=[book["element_id"]], section_path=section_path, section_path_ids=section_ids, refs=refs, text=text, citation=citation(profile, f"حديث {number}"), child_view_ids=[v["view_id"] for v in views], quality_flags=flags)
        if grade and grade.get("albani_grade_raw"):
            # His whole verdict, references and caveats included ("صحيح ... لكن ذكر بلال فيه منكر").
            parent["text_for_synthesis"] += f"\n\nحكم الألباني: {clean_text(grade['albani_grade_raw'])}"
        state.parents.append(parent)
        sanad_view = next((v for v in views if v["view_type"] == "hadith_sanad"), None)
        if sanad_view:
            state.isnad_edges.extend(narrator_edges_from_sanad(sanad, parent_id=parent_id, profile=profile, view_id=sanad_view["view_id"]))


def parse_tafsir(profile: dict[str, Any], state: BuildState) -> None:
    book = book_root(profile, profile["source_paths"][0])
    state.elements.append(book)
    order = 0
    for source_path in profile["source_paths"]:
        data = json.loads((CLEAN_ROOT / source_path).read_text(encoding="utf-8"))
        for rec in data.get("ayahs", []):
            order += 1
            surah = int(rec.get("surah"))
            ayah = int(rec.get("ayah"))
            text = rec.get("text", "")
            if not clean_text(text):
                continue
            refs = default_refs()
            refs["quran"] = {"surah": surah, "ayah_start": ayah, "ayah_end": ayah}
            elem_id = f"elem:{profile['book_id']}:tafsir:{surah}:{ayah}"
            section_path = [profile.get("title_ar") or profile["book_id"], f"سورة {surah}", f"آية {ayah}"]
            section_ids = [f"book:{profile['book_id']}", f"surah:{surah}", f"ayah:{ayah}"]
            state.elements.append(make_element(profile, element_id=elem_id, source_path=source_path, source_record_id=f"surah:{surah}:ayah:{ayah}", element_type="tafsir_ayah_commentary", level=2, parent_element_id=book["element_id"], ancestor_ids=[book["element_id"]], order_in_parent=order, section_path=section_path, section_path_ids=section_ids, text=text, refs=refs))
            parent_segments = split_paragraph_packs(text, 1400)
            multi_parent = len(parent_segments) > 1
            for seg_idx, segment in enumerate(parent_segments, start=1):
                parent_suffix = f":seg:{seg_idx:03d}" if multi_parent else ""
                parent_id = f"parent:{profile['book_id']}:tafsir:{surah}:{ayah}{parent_suffix}"
                seg_section_path = section_path + ([f"مقطع {seg_idx}"] if multi_parent else [])
                seg_section_ids = section_ids + ([f"segment:{seg_idx}"] if multi_parent else [])
                view_ids = []
                for idx, part in enumerate(split_paragraph_packs(segment, 360), start=1):
                    view_type = "tafsir_ayah_commentary" if not multi_parent and idx == 1 and len(segment.split()) <= 360 else "tafsir_paragraph"
                    view_id = f"view:{profile['book_id']}:tafsir:{surah}:{ayah}{parent_suffix}:{idx:03d}"
                    state.views.append(make_view(profile, view_id=view_id, parent_id=parent_id, view_type=view_type, text_role="commentary", order_in_parent=idx, section_path=seg_section_path, section_path_ids=seg_section_ids, refs=refs, text=part, quality_flags=["long_tafsir_parent_segment"] if multi_parent else []))
                    view_ids.append(view_id)
                state.parents.append(make_parent(profile, parent_id=parent_id, parent_type="tafsir_ayah_commentary", source_element_ids=[elem_id], ancestor_ids=[book["element_id"], elem_id], section_path=seg_section_path, section_path_ids=seg_section_ids, refs=refs, text=segment, citation=citation(profile, f"سورة {surah}، آية {ayah}" + (f"، مقطع {seg_idx}" if multi_parent else "")), child_view_ids=view_ids, quality_flags=["long_tafsir_parent_segment"] if multi_parent else []))
                add_quran_ref(state, source_kind="parent", source_id=parent_id, book_id=profile["book_id"], surah=surah, ayah_start=ayah, ayah_end=ayah, match_method="source_structural_ref", confidence=1.0, matched_text=segment)


def parse_text_profile(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_paths"][0]
    text = strip_former_delimiters((CLEAN_ROOT / source_path).read_text(encoding="utf-8", errors="replace"))
    book = book_root(profile, source_path)
    state.elements.append(book)
    if profile["parent_policy"] == "numbered_item_parent_v1":
        units = [(f"مسألة {num}", body, f"numbered:{num}") for num, body in split_numbered_items(text)]
        parent_type = "numbered_item"
        element_type = "numbered_item"
    elif profile["parent_policy"] == "irab_ayah_commentary_parent_v1":
        units = split_irab_units(text)
        parent_type = "irab_ayah_commentary"
        element_type = "irab_ayah_commentary"
    else:
        sections = split_sections_by_lines(text, profile)
        if len(sections) <= 1:
            sections = [(f"مقطع {idx}", part) for idx, part in enumerate(split_paragraph_packs(text, 900), start=1)]
        units = [(title, body, f"section:{idx}") for idx, (title, body) in enumerate(sections, start=1)]
        parent_type = "heading_section" if profile["parent_policy"] == "heading_section_parent_v1" else "paragraph_group"
        element_type = "heading" if parent_type == "heading_section" else "paragraph_group"
    for order, unit in enumerate(units, start=1):
        title, body, unit_id = unit
        if not clean_text(body):
            continue
        refs = default_refs()
        qref = extract_numeric_quran_ref(title) or extract_numeric_quran_ref(body[:500])
        if qref:
            refs["quran"] = {"surah": qref[0], "ayah_start": qref[1], "ayah_end": qref[2]}
        elem_id = f"elem:{profile['book_id']}:{stable_hash(unit_id + str(order))}"
        section_path = [profile.get("title_ar") or profile["book_id"], clean_text(title)[:120]]
        section_ids = [f"book:{profile['book_id']}", unit_id]
        flags = []
        parent_segments = split_paragraph_packs(body, 1400)
        multi_parent = len(parent_segments) > 1
        if multi_parent:
            flags.append("long_parent_segment")
        state.elements.append(make_element(profile, element_id=elem_id, source_path=source_path, source_record_id=unit_id, element_type=element_type, level=2, parent_element_id=book["element_id"], ancestor_ids=[book["element_id"]], order_in_parent=order, section_path=section_path, section_path_ids=section_ids, text=body, refs=refs, quality_flags=flags, parse_confidence=0.85))
        for seg_idx, segment in enumerate(parent_segments, start=1):
            parent_suffix = f":seg:{seg_idx:03d}" if multi_parent else ""
            segment_flags = flags[:] if multi_parent else []
            parent_id = f"parent:{profile['book_id']}:{stable_hash(unit_id + str(order))}{parent_suffix}"
            seg_section_path = section_path + ([f"مقطع {seg_idx}"] if multi_parent else [])
            seg_section_ids = section_ids + ([f"segment:{seg_idx}"] if multi_parent else [])
            view_ids = []
            parts = split_paragraph_packs(segment, 420)
            if len(parts) > 1:
                segment_flags.append("parent_split_into_multiple_views")
            for idx, part in enumerate(parts, start=1):
                view_type = "commentary_paragraph" if parent_type == "irab_ayah_commentary" else "paragraph_pack"
                text_role = "commentary" if parent_type == "irab_ayah_commentary" else "paragraph"
                if parent_type == "numbered_item":
                    view_type = "numbered_item_view"
                    text_role = "numbered_item"
                view_id = f"view:{profile['book_id']}:{stable_hash(unit_id + str(order))}{parent_suffix}:{idx:03d}"
                state.views.append(make_view(profile, view_id=view_id, parent_id=parent_id, view_type=view_type, text_role=text_role, order_in_parent=idx, section_path=seg_section_path, section_path_ids=seg_section_ids, refs=refs, text=part, quality_flags=segment_flags))
                view_ids.append(view_id)
            state.parents.append(make_parent(profile, parent_id=parent_id, parent_type=parent_type, source_element_ids=[elem_id], ancestor_ids=[book["element_id"], elem_id], section_path=seg_section_path, section_path_ids=seg_section_ids, refs=refs, text=segment, citation=citation(profile, clean_text(title)[:100] + (f"، مقطع {seg_idx}" if multi_parent else "")), child_view_ids=view_ids, quality_flags=segment_flags))
            if refs["quran"]:
                q = refs["quran"]
                add_quran_ref(state, source_kind="parent", source_id=parent_id, book_id=profile["book_id"], surah=q["surah"], ayah_start=q["ayah_start"], ayah_end=q["ayah_end"], match_method="explicit_ref", confidence=0.75, matched_text=title)


def split_irab_units(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    units: list[tuple[str, list[str], str]] = []
    current_title = "مقدمة"
    current_id = "intro"
    current_lines: list[str] = []
    current_surah = None
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("[") and "سورة" in stripped and "آية" in stripped:
            if current_lines:
                units.append((current_title, current_lines, current_id))
            title = stripped
            qref = extract_numeric_quran_ref(stripped)
            current_title = title
            current_id = f"ayah:{qref[0]}:{qref[1]}" if qref else f"anchor:{idx}"
            current_lines = [line]
            continue
        if ("سورة" in stripped or "سُورَة" in stripped) and len(stripped) < 80:
            current_surah = stripped
        if IRAB_ANCHOR_RE.search(stripped) and current_lines and len(current_lines) > 2:
            units.append((current_title, current_lines, current_id))
            current_title = f"{current_surah or 'آية'} - {stripped[:80]}"
            current_id = f"anchor:{idx}"
            current_lines = [line]
            continue
        current_lines.append(line)
    if current_lines:
        units.append((current_title, current_lines, current_id))
    return [(title, "\n".join(body).strip(), unit_id) for title, body, unit_id in units if "\n".join(body).strip()]


def parse_shamela_json_records(profile: dict[str, Any], state: BuildState) -> None:
    book = book_root(profile, profile["source_paths"][0])
    state.elements.append(book)
    grouped: list[tuple[str, list[str], list[str]]] = []
    current_title = ""
    current_parts: list[str] = []
    current_paths: list[str] = []
    for source_path in profile["source_paths"]:
        data = json.loads((CLEAN_ROOT / source_path).read_text(encoding="utf-8"))
        title = clean_text(data.get("title") or "بلا عنوان")
        paragraphs = [clean_text(p) for p in data.get("paragraphs", []) if clean_text(p)]
        if current_parts and title != current_title:
            grouped.append((current_title, current_parts, current_paths))
            current_parts = []
            current_paths = []
        current_title = title
        current_parts.extend(paragraphs)
        current_paths.append(source_path)
    if current_parts:
        grouped.append((current_title, current_parts, current_paths))
    order = 0
    for title, paragraphs, paths in grouped:
        for pack_idx, pack in enumerate(split_paragraph_packs("\n".join(paragraphs), 900), start=1):
            order += 1
            unit_id = f"record_group:{order}:{stable_hash(title)}:{pack_idx}"
            elem_id = f"elem:{profile['book_id']}:{stable_hash(unit_id)}"
            parent_id = f"parent:{profile['book_id']}:{stable_hash(unit_id)}"
            refs = default_refs()
            refs["source_record"] = {"record_id": ",".join(Path(p).stem for p in paths[:5]), "record_index": order}
            section_path = [profile.get("title_ar") or profile["book_id"], title]
            section_ids = [f"book:{profile['book_id']}", f"title:{stable_hash(title)}"]
            view_id = f"view:{profile['book_id']}:{stable_hash(unit_id)}:001"
            state.elements.append(make_element(profile, element_id=elem_id, source_path=paths[0], source_record_id=unit_id, element_type="paragraph_group", level=2, parent_element_id=book["element_id"], ancestor_ids=[book["element_id"]], order_in_parent=order, section_path=section_path, section_path_ids=section_ids, text=pack, refs=refs))
            state.views.append(make_view(profile, view_id=view_id, parent_id=parent_id, view_type="paragraph_pack", text_role="paragraph", order_in_parent=1, section_path=section_path, section_path_ids=section_ids, refs=refs, text=pack))
            state.parents.append(make_parent(profile, parent_id=parent_id, parent_type="heading_section", source_element_ids=[elem_id], ancestor_ids=[book["element_id"]], section_path=section_path, section_path_ids=section_ids, refs=refs, text=pack, citation=citation(profile, title), child_view_ids=[view_id]))


def parse_one(profile: dict[str, Any], state: BuildState) -> None:
    before = (len(state.elements), len(state.parents), len(state.views))
    parser = profile["parser_profile"]
    if parser == "quran_json_parser":
        parse_quran(profile, state)
    elif parser in {"hadith_json_parser", "hadith_flat_json_parser", "musnad_csv_parser"}:
        parse_hadith(profile, state)
    elif parser == "tafsir_json_parser":
        parse_tafsir(profile, state)
    elif parser == "shamela_json_records_parser":
        parse_shamela_json_records(profile, state)
    else:
        parse_text_profile(profile, state)
    after = (len(state.elements), len(state.parents), len(state.views))
    state.report_rows.append(
        {
            "book_id": profile["book_id"],
            "parser_profile": parser,
            "elements": after[0] - before[0],
            "parents": after[1] - before[1],
            "retrieval_views": after[2] - before[2],
        }
    )


def validate_profiles(profiles: list[dict[str, Any]]) -> None:
    for profile in profiles:
        for source_path in profile.get("source_paths", []):
            path = CLEAN_ROOT / source_path
            if not path.exists():
                raise FileNotFoundError(f"{profile['book_id']} source missing: {source_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Allow replacing existing structured JSONL outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N parser profile rows.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="books parsed in parallel")
    parser.add_argument("--book-id", action="append", help="Only process a specific book_id. May be repeated.")
    return parser.parse_args(argv)


def parse_book(profile: dict[str, Any]) -> BuildState:
    state = BuildState([], [], [], [], [], [])
    parse_one(profile, state)
    return state


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    out_root: Path = args.output_root
    profiles = read_jsonl(PROFILE_PATH)
    if args.book_id:
        wanted = set(args.book_id)
        profiles = [p for p in profiles if p["book_id"] in wanted]
    if args.limit:
        profiles = profiles[: args.limit]
    validate_profiles(profiles)
    output_files = [
        out_root / name
        for name in ("elements.jsonl", "parents.jsonl", "retrieval_views.jsonl")
    ]
    if not args.force and any(path.exists() for path in output_files):
        raise RuntimeError(
            f"Refusing to overwrite existing structured corpus at {out_root}; "
            "pass --output-root for a new build or --force intentionally."
        )
    # Books are parsed in parallel, each into its own state (no book reads another's
    # rows), and joined in profile order: the output does not depend on --workers.
    state = BuildState([], [], [], [], [], [])
    with Pool(args.workers) as pool:
        for part in pool.imap(parse_book, profiles):
            for name in ("elements", "parents", "views", "quran_refs", "isnad_edges", "report_rows"):
                getattr(state, name).extend(getattr(part, name))
    out_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_root / "elements.jsonl", state.elements)
    write_jsonl(out_root / "parents.jsonl", state.parents)
    write_jsonl(out_root / "retrieval_views.jsonl", state.views)
    write_jsonl(out_root / "quran_ref_index.jsonl", state.quran_refs)
    write_jsonl(out_root / "isnad_edges.jsonl", state.isnad_edges)
    summary = {
        "schema": "dalil.parser_coverage_report.v1",
        "books_processed": len(profiles),
        "total_elements": len(state.elements),
        "total_parents": len(state.parents),
        "total_retrieval_views": len(state.views),
        "total_quran_refs": len(state.quran_refs),
        "total_isnad_edges": len(state.isnad_edges),
        "by_parser_profile": dict(Counter(row["parser_profile"] for row in state.report_rows)),
        "rows": state.report_rows,
    }
    write_json(out_root / "parser_coverage_report.json", summary)
    print(json.dumps({k: summary[k] for k in ("books_processed", "total_elements", "total_parents", "total_retrieval_views", "total_quran_refs", "total_isnad_edges")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
