"""Ketabonline books, from the pre-extracted ``structured/ketabonline/elements.jsonl``.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..records import BuildState, clean_text, default_refs, safe_num
from ..sections import (
    PARAGRAPH_PACK_WORDS,
    emit_section,
    hold_flags,
    make_element_fixed,
)
from ..text import sanitize_markup, strip_source_markers

TAKHRIJ_PARA_RE = re.compile(r"^\s*[0-9٠-٩]{1,5}\s*[-–—]\s*[\(\[]")


def parse_ketabonline_book(
    profile: dict[str, Any],
    elements: list[dict[str, Any]],
    state: BuildState,
) -> None:
    book_el = next(e for e in elements if e["element_type"] == "book")
    chapters = [e for e in elements if e["element_type"] == "chapter"]
    headings = [e for e in elements if e["element_type"] == "heading"]
    paragraphs = [e for e in elements if e["element_type"] == "paragraph"]
    policy = profile["parent_policy"]

    # re-emit book element with audit policies
    book = make_element_fixed(
        profile,
        element_id=book_el["element_id"],
        source_path=profile["source_path"],
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

    # re-emit chapters
    for ch in chapters:
        state.elements.append(
            make_element_fixed(
                profile,
                element_id=ch["element_id"],
                source_path=profile["source_path"],
                source_record_id=ch.get("source_record_id", "chapter"),
                element_type="chapter",
                level=2,
                parent_element_id=book["element_id"],
                ancestor_ids=[book["element_id"]],
                order_in_parent=ch.get("order_in_parent", 0),
                section_path=ch.get("section_path", []),
                section_path_ids=ch.get("section_path_ids", []),
                text=strip_source_markers(sanitize_markup(ch.get("text_clean", ""))),
                refs=ch.get("refs", default_refs()),
                quality_flags=hold_flags(profile),
            )
        )

    # map index_entry_id -> heading (deepest title_level wins)
    idx_to_heading: dict[Any, tuple[int, dict[str, Any]]] = {}
    for h in headings:
        ids = h.get("refs", {}).get("source_record", {}).get("index_entry_ids") or []
        tl = safe_num(h.get("refs", {}).get("source_record", {}).get("title_level"), 0)
        for i in ids:
            cur = idx_to_heading.get(i)
            if cur is None or tl >= cur[0]:
                idx_to_heading[i] = (tl, h)

    # assign paragraphs to headings, walking the elements in page order. A heading
    # that is in the text but not in the book's index (a waypoint: كتاب/باب titles,
    # numbered questions, the questions past the point where an index stops) heads
    # the paragraphs after it until the next heading; otherwise they would all fall
    # to the index entry above it and the waypoint would stay empty.
    heading_paras: dict[str, list[dict[str, Any]]] = defaultdict(list)
    book_paras: list[dict[str, Any]] = []
    para_parent: dict[str, dict[str, Any] | None] = {}
    last_heading: dict[str, Any] | None = None
    waypoint: dict[str, Any] | None = None
    for p in elements:
        if p["element_type"] == "heading":
            waypoint = p if "html_title_waypoint" in (p.get("quality_flags") or []) else None
            continue
        if p["element_type"] != "paragraph":
            continue
        ids = p.get("refs", {}).get("source_record", {}).get("index_entry_ids") or []
        target = None
        best_tl = -1
        for i in ids:
            if i in idx_to_heading:
                tl, h = idx_to_heading[i]
                if tl > best_tl:
                    best_tl = tl
                    target = h
        if waypoint is not None:
            target = waypoint
        if target is None:
            target = last_heading
        if target is not None:
            heading_paras[target["element_id"]].append(p)
            last_heading = target
        else:
            book_paras.append(p)
        para_parent[p["element_id"]] = target

    # re-emit headings + paragraphs with updated parent links
    for h in headings:
        state.elements.append(
            make_element_fixed(
                profile,
                element_id=h["element_id"],
                source_path=profile["source_path"],
                source_record_id=h.get("source_record_id", "heading"),
                element_type="heading",
                level=2,
                parent_element_id=book["element_id"],
                ancestor_ids=[book["element_id"]],
                order_in_parent=h.get("order_in_parent", 0),
                section_path=h.get("section_path", []),
                section_path_ids=h.get("section_path_ids", []),
                text=strip_source_markers(sanitize_markup(h.get("text_clean", ""))),
                refs=h.get("refs", default_refs()),
                quality_flags=hold_flags(profile),
            )
        )
    for p in paragraphs:
        parent_el = para_parent.get(p["element_id"])
        state.elements.append(
            make_element_fixed(
                profile,
                element_id=p["element_id"],
                source_path=profile["source_path"],
                source_record_id=p.get("source_record_id", "paragraph"),
                element_type="paragraph",
                level=3,
                parent_element_id=parent_el["element_id"] if parent_el else book["element_id"],
                ancestor_ids=[book["element_id"]] + ([parent_el["element_id"]] if parent_el else []),
                order_in_parent=p.get("order_in_parent", 0),
                section_path=p.get("section_path", []),
                section_path_ids=p.get("section_path_ids", []),
                text=strip_source_markers(sanitize_markup(p.get("text_clean", ""))),
                refs=p.get("refs", default_refs()),
                quality_flags=hold_flags(profile),
            )
        )

    order = 0

    def emit(title: str, body: str, unit_id: str, parent_type: str, element_type: str,
             view_types: tuple[str, str], text_roles: tuple[str, str],
             refs: dict[str, Any] | None = None, pack_words: int = PARAGRAPH_PACK_WORDS,
             hadith_split: bool = False, qanda_pair: bool = False,
             existing_element: dict[str, Any] | None = None) -> None:
        nonlocal order
        order += 1
        emit_section(
            profile, state, book=book, source_path=profile["source_path"],
            unit_id=unit_id, title=title, body=body, order=order,
            parent_type=parent_type, element_type=element_type,
            view_types=view_types, text_roles=text_roles, refs=refs,
            pack_words=pack_words, hadith_split=hadith_split, qanda_pair=qanda_pair,
            existing_element=existing_element,
        )

    if policy == "ketabonline_takhrij_record_parent_v1":
        # A takhrij record OPENS with 'N - ( verdict )' in its own paragraph and
        # continues through the following paragraphs until the next such marker.
        # Treating each matching paragraph as a whole record instead produced
        # 6,461 header-only parents with a median of 5 words, and swept every
        # body paragraph into a single 604,000-word blob — the hadith text was
        # present but unusable. Group from marker to marker instead.
        records: list[tuple[str, list[str]]] = []
        preface: list[str] = []
        header: str | None = None
        body: list[str] = []
        for p in paragraphs:
            text = (p.get("text_clean") or "").strip()
            if not text:
                continue
            if TAKHRIJ_PARA_RE.match(text):
                if header is not None:
                    records.append((header, body))
                header, body = text, [text]
            elif header is not None:
                body.append(text)
            else:
                preface.append(text)
        if header is not None:
            records.append((header, body))

        if preface:
            emit(profile.get("title_ar") or "", "\n".join(preface), "takhrij:preface",
                 "paragraph_group", "paragraph_group",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
        for idx, (head, paras) in enumerate(records, start=1):
            emit(head[:120], "\n".join(paras), f"takhrij:{idx}",
                 "takhrij_record", "takhrij_record",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
        return

    for h in headings:
        paras = heading_paras.get(h["element_id"], [])
        body = "\n".join(strip_source_markers(sanitize_markup(p.get("text_clean", ""))) for p in paras)
        title = h.get("text_clean", "")
        if not clean_text(body) and not clean_text(title):
            continue
        refs = default_refs()
        if policy in ("ketabonline_hadith_record_parent_v1", "ketabonline_hadith_section_parent_v1"):
            m = re.match(r"^\s*([0-9٠-٩]{1,5})", title)
            if m and "html_title_waypoint" not in (h.get("quality_flags") or []):  # "١- كتاب الطهارة" is not hadith 1
                refs["hadith"] = {
                    "collection": profile.get("work_id") or profile["book_id"],
                    "book_number": None,
                    "chapter_id": None,
                    "hadith_number": safe_num(m.group(1)),
                    "id_in_book": None,
                }
            emit(title or "حديث", body, f"hadith:{h['element_id']}",
                 "hadith", "hadith",
                 ("heading_title", "hadith_full"), ("heading", "full"),
                 refs=refs, hadith_split=True, existing_element=h)
        elif policy == "ketabonline_qa_section_parent_v1":
            emit(title or "سؤال", body, f"qanda:{h['element_id']}",
                 "qanda", "qanda",
                 ("heading_title", "qanda_pair"), ("heading", "paragraph"),
                 qanda_pair=True, existing_element=h)
        elif policy == "ketabonline_tape_section_parent_v1":
            emit(title or "شريط", body, f"tape:{h['element_id']}",
                 "tape_section", "tape_section",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 existing_element=h)
        elif policy == "ketabonline_bab_section_parent_v1":
            emit(title or "باب", body, f"bab:{h['element_id']}",
                 "bab_section", "bab_section",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 existing_element=h)
        else:  # ketabonline_heading_section_parent_v1 and fallback
            emit(title or "مقدمة", body, f"section:{h['element_id']}",
                 "heading_section", "heading",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 existing_element=h)

    if book_paras:
        body = "\n".join(strip_source_markers(sanitize_markup(p.get("text_clean", ""))) for p in book_paras)
        if clean_text(body):
            emit(profile.get("title_ar") or "", body, "book:orphan_paras",
                 "paragraph_group", "paragraph_group",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
