"""OpenITI TXT books.
"""
from __future__ import annotations

import re
from typing import Any

from dalil.paths import DATA_DIR

from ..records import BuildState, clean_text, default_refs, stable_hash
from ..sections import TAFSIR_PACK_WORDS, emit_section, make_element_fixed

PAGEV_RE = re.compile(r"^\s*PageV\S*\s*@?\s*$")
PAGEV_INLINE_RE = re.compile(r"\s*PageV\S*\s*@?\s*")
OPENITI_HEADING_BAR_RE = re.compile(r"^\s*#\s*\|\s*(.*)$")
OPENITI_HASH3_RE = re.compile(r"^\s*###\s*\|?\s*(.*)$")
OPENITI_PROSE_RE = re.compile(r"^\s*#\s+(.*)$")
OPENITI_CONT_RE = re.compile(r"^\s*~~\s*(.*)$")
OPENITI_NUMBERED_RE = re.compile(r"^\s*#\s*([0-9٠-٩]{1,4})\s*[)\-.،]?\s*(.*)$")
MASALA_RE = re.compile(r"مسألة\s*([0-9٠-٩]{1,4})\s*\]?\s*:?\s*")
SURA_HEADING_RE = re.compile(r"سورة")


def read_openiti_lines(source_path: str, excluded_page_ids: set[str] | None = None) -> list[str]:
    text = (DATA_DIR / source_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if l.strip() == "#META#Header#End#":
            start = i + 1
            break
    lines = lines[start:]
    # strip standalone PageV lines and inline PageV markers at line ends;
    # drop lines belonging to excluded pages (empty-after-cleaning records)
    out: list[str] = []
    for l in lines:
        s = l.strip()
        if PAGEV_RE.match(s):
            continue
        if excluded_page_ids:
            m = re.search(r"PageV\S+", s)
            if m and m.group(0) in excluded_page_ids:
                continue
        s = PAGEV_INLINE_RE.sub("", s)
        out.append(s)
    return out


def openiti_paragraphs(lines: list[str]) -> list[str]:
    """Group '# prose' lines with their '~~' continuations into paragraphs."""
    paras: list[str] = []
    current: list[str] = []
    for l in lines:
        s = l.strip()
        if OPENITI_PROSE_RE.match(s) or OPENITI_HEADING_BAR_RE.match(s) or OPENITI_HASH3_RE.match(s):
            if current:
                paras.append("\n".join(current))
            current = [s]
        elif OPENITI_CONT_RE.match(s):
            current.append(OPENITI_CONT_RE.match(s).group(1).strip())
        elif s:
            current.append(s)
    if current:
        paras.append("\n".join(current))
    return [p for p in paras if clean_text(p)]


def parse_openiti(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_path"]
    excluded_ids = profile.get("_excluded_record_ids", set())
    lines = read_openiti_lines(source_path, excluded_ids)
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

    if policy == "heading_section_parent_v1":
        sections: list[tuple[str, list[str]]] = []
        current_title: str | None = None
        current: list[str] = []
        # OpenITI books mark headings either as '# | عنوان' or as '### عنوان'.
        # Matching only the first form left 80 books with zero detected sections,
        # so each collapsed into a single parent — one of them 592,551 words with
        # 2,385 '###' headings sitting unread in the source.
        for l in lines:
            stripped = l.strip()
            m = OPENITI_HEADING_BAR_RE.match(stripped) or OPENITI_HASH3_RE.match(stripped)
            if m:
                if current_title is not None or current:
                    sections.append((current_title or "مقدمة", current))
                current_title = m.group(1).strip()
                current = []
            else:
                current.append(l)
        if current_title is not None or current:
            sections.append((current_title or "مقدمة", current))
        for title, body_lines in sections:
            order += 1
            body = "\n".join(openiti_paragraphs(body_lines))
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id=f"section:{order}:{stable_hash(title)}",
                title=title, body=body, order=order,
                parent_type="heading_section", element_type="heading",
                view_types=("heading_title", "paragraph_pack"),
                text_roles=("heading", "paragraph"),
            )

    elif policy == "numbered_item_parent_v1":
        items: list[tuple[str, str]] = []
        current_num: str | None = None
        current: list[str] = []
        for l in lines:
            m = OPENITI_NUMBERED_RE.match(l.strip())
            if m:
                if current_num is not None and current:
                    items.append((current_num, "\n".join(openiti_paragraphs(current))))
                current_num = m.group(1)
                current = [m.group(2).strip()]
            else:
                current.append(l)
        if current_num is not None and current:
            items.append((current_num, "\n".join(openiti_paragraphs(current))))
        for num, body in items:
            order += 1
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id=f"numbered:{num}", title=f"مسألة {num}", body=body, order=order,
                parent_type="numbered_item", element_type="numbered_item",
                view_types=("heading_title", "numbered_item_view"),
                text_roles=("heading", "numbered_item"),
            )

    elif policy in ("monograph_prose_parent_v1", "whole_book_parent_v1"):
        body = "\n".join(openiti_paragraphs(lines))
        order += 1
        emit_section(
            profile, state, book=book, source_path=source_path,
            unit_id="book:whole", title=profile.get("title_ar") or "", body=body, order=order,
            parent_type="paragraph_group", element_type="paragraph_group",
            view_types=("heading_title", "paragraph_pack"),
            text_roles=("heading", "paragraph"),
        )

    elif policy == "poem_verse_parent_v1":
        # stanzas via '### | فصل' headings; verses are '# ...' lines with hemistich separator
        stanzas: list[tuple[str, list[str]]] = []
        current_title: str | None = None
        current: list[str] = []
        for l in lines:
            m = OPENITI_HASH3_RE.match(l.strip())
            if m:
                if current_title is not None or current:
                    stanzas.append((current_title or "القصيدة", current))
                current_title = m.group(1).strip()
                current = []
            else:
                current.append(l)
        if current_title is not None or current:
            stanzas.append((current_title or "القصيدة", current))
        for title, body_lines in stanzas:
            order += 1
            body = "\n".join(openiti_paragraphs(body_lines))
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id=f"stanza:{order}:{stable_hash(title)}",
                title=title, body=body, order=order,
                parent_type="poem_verse", element_type="poem_verse",
                view_types=("heading_title", "poem_verse"),
                text_roles=("heading", "verse"),
                pack_words=80,
            )

    elif policy == "fatwa_masala_parent_v1":
        text = "\n".join(lines)
        parts = MASALA_RE.split(text)
        # parts: [pre, num1, body1, num2, body2, ...]
        if parts and clean_text(parts[0]):
            order += 1
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id="fatwa:intro", title="مقدمة", body=parts[0], order=order,
                parent_type="fatwa_masala", element_type="fatwa_masala",
                view_types=("heading_title", "paragraph_pack"),
                text_roles=("heading", "paragraph"),
            )
        for i in range(1, len(parts), 2):
            num = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            order += 1
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id=f"fatwa:{num}", title=f"مسألة {num}", body=body, order=order,
                parent_type="fatwa_masala", element_type="fatwa_masala",
                view_types=("heading_title", "paragraph_pack"),
                text_roles=("heading", "paragraph"),
            )

    elif policy == "tafsir_sura_ayat_parent_v1":
        # '### | سورة X' top-level; '[سورة X (N) : الآيات a الى b]' blocks inside
        suras: list[tuple[str, list[str]]] = []
        current_title: str | None = None
        current: list[str] = []
        for l in lines:
            m = OPENITI_HASH3_RE.match(l.strip())
            if m and SURA_HEADING_RE.search(m.group(1)):
                if current_title is not None or current:
                    suras.append((current_title or "مقدمة", current))
                current_title = m.group(1).strip()
                current = []
            else:
                current.append(l)
        if current_title is not None or current:
            suras.append((current_title or "مقدمة", current))
        for sura_title, sura_lines in suras:
            order += 1
            body = "\n".join(openiti_paragraphs(sura_lines))
            emit_section(
                profile, state, book=book, source_path=source_path,
                unit_id=f"sura:{order}:{stable_hash(sura_title)}",
                title=sura_title, body=body, order=order,
                parent_type="tafsir_ayah_commentary", element_type="tafsir_ayah_commentary",
                view_types=("heading_title", "tafsir_paragraph"),
                text_roles=("heading", "commentary"),
                pack_words=TAFSIR_PACK_WORDS,
            )

    else:
        # fallback: whole book paragraph packs
        body = "\n".join(openiti_paragraphs(lines))
        order += 1
        emit_section(
            profile, state, book=book, source_path=source_path,
            unit_id="book:fallback", title=profile.get("title_ar") or "", body=body, order=order,
            parent_type="paragraph_group", element_type="paragraph_group",
            view_types=("heading_title", "paragraph_pack"),
            text_roles=("heading", "paragraph"),
        )
