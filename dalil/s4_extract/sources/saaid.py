"""saaid.net TXT books (al-Saeedan): masa'il, Q&A and ordinal-heading structure.
"""
from __future__ import annotations

import re
from typing import Any

from dalil.paths import DATA_DIR

from ..qanda import split_qanda_pair
from ..records import BuildState, clean_text, default_refs
from ..sections import emit_section, make_element_fixed

# These books arrive as .doc (converted to .txt), one file per book. They carry
# no markup at all, so structure has to come from the author's own conventions:
# 'مسألة :-' introduces each ruling, and the Q&A volumes use 'س<n>)' / 'ج)'.
SAAID_MASALA_RE = re.compile(r"(?m)^\s*(مسألة\s*:?\s*-?\s*)")
SAAID_GROUP_WORDS = 700
SAAID_Q_RE = re.compile(r"(?m)^\s*س\s*([0-9٠-٩]{0,4})\s*\)")
# 'الفرع الأول:-' … 'الفرع الحادي والثلاثون:-' number the branches of an argument.
SAAID_FARA_RE = re.compile(r"(\(?\s*الفرع\s+ال[ء-ي]+(?:\s+و?ال[ء-ي]+)?\s*\)?\s*:?\s*-?\s*)")
# '((كتاب الطهارة))' style headings. The ((...)) wrapper is NOT reliable on its
# own: several of these books use it for hadith quotation instead. Requiring a
# structural word and a short span separates the two — without this guard, every
# quoted hadith becomes a spurious section break.
SAAID_BRACKET_HEAD_RE = re.compile(
    r"(?m)^\s*(\(\(\s*(?=[^)]{0,60}(?:كتاب|باب|فصل|مقدمة|خاتمة))[^)]{3,60}\)\)\s*)"
)


def parse_saaid(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_path"]
    text = (DATA_DIR / source_path).read_text(encoding="utf-8", errors="replace").lstrip("﻿")
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
    policy = profile["parent_policy"]
    order = 0

    def emit(title, body, unit_id, parent_type, element_type, views, roles, **kw):
        nonlocal order
        order += 1
        emit_section(
            profile, state, book=book, source_path=source_path, unit_id=unit_id,
            title=title, body=body, order=order, parent_type=parent_type,
            element_type=element_type, view_types=views, text_roles=roles, **kw,
        )

    if policy == "qanda_pair_parent_v1":
        starts = [m.start() for m in SAAID_Q_RE.finditer(text)]
        if starts:
            if clean_text(text[: starts[0]]):
                emit(profile.get("title_ar") or "", text[: starts[0]], "saaid:preface",
                     "heading_section", "heading",
                     ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
            for i, s in enumerate(starts):
                chunk = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)]
                question, _ = split_qanda_pair(chunk)
                emit(clean_text(question)[:150] or "سؤال", chunk, f"saaid:q:{i + 1}",
                     "qanda", "qanda", ("heading_title", "qanda_pair"),
                     ("heading", "paragraph"), qanda_pair=True)
            return

    if policy == "heading_section_parent_v1":
        parts = SAAID_BRACKET_HEAD_RE.split(text)
        if len(parts) > 1:
            if clean_text(parts[0]):
                emit(profile.get("title_ar") or "", parts[0], "saaid:preface",
                     "heading_section", "heading",
                     ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
            for i in range(1, len(parts), 2):
                head = clean_text(parts[i]).strip("() ")
                body = parts[i + 1] if i + 1 < len(parts) else ""
                emit(head or "فصل", body, f"saaid:head:{(i + 1) // 2}",
                     "heading_section", "heading",
                     ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
            return

    if policy == "numbered_item_parent_v1":
        parts = SAAID_FARA_RE.split(text)
        if len(parts) > 1:
            if clean_text(parts[0]):
                emit(profile.get("title_ar") or "", parts[0], "saaid:preface",
                     "heading_section", "heading",
                     ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
            for i in range(1, len(parts), 2):
                head = clean_text(parts[i]).strip("() :-")
                body = parts[i + 1] if i + 1 < len(parts) else ""
                emit(head or "فرع", body, f"saaid:fara:{(i + 1) // 2}",
                     "numbered_item", "numbered_item",
                     ("heading_title", "numbered_item_view"), ("heading", "numbered_item"))
            return

    # masala-delimited prose, when the author uses 'مسألة' to open each ruling
    parts = SAAID_MASALA_RE.split(text)
    if len(parts) > 1:
        if clean_text(parts[0]):
            emit(profile.get("title_ar") or "", parts[0], "saaid:preface",
                 "heading_section", "heading",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
        for i in range(1, len(parts), 2):
            body = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            emit("مسألة", body, f"saaid:masala:{(i + 1) // 2}",
                 "fatwa_masala", "fatwa_masala",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
        return

    # No structural marker at all. Emitting the whole book as one parent would
    # produce a 30k-word unit that is useless for retrieval, so fall back to
    # fixed-size paragraph groups: still coherent, still citable, and each one
    # small enough that its embedding means something.
    paras = [p for p in re.split(r"\n\s*\n|\n", text) if clean_text(p)]
    group: list[str] = []
    words = 0
    groups: list[str] = []
    for p in paras:
        group.append(p)
        words += len(p.split())
        if words >= SAAID_GROUP_WORDS:
            groups.append("\n".join(group))
            group, words = [], 0
    if group:
        groups.append("\n".join(group))
    for idx, body in enumerate(groups, start=1):
        emit(profile.get("title_ar") or "", body, f"saaid:group:{idx}",
             "paragraph_group", "paragraph_group",
             ("heading_title", "paragraph_pack"), ("heading", "paragraph"))
