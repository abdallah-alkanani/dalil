"""Shamela 4 books (pages.jsonl + toc.jsonl).
"""
from __future__ import annotations

import json
import re
from typing import Any

from dalil.paths import DATA_DIR

from ..qanda import split_qanda_units
from ..quran_refs import extract_ayah_ref, extract_numeric_quran_ref, surah_from_title
from ..records import BuildState, clean_text, default_refs, safe_num, stable_hash
from ..sections import (
    PARAGRAPH_PACK_WORDS,
    TAFSIR_PACK_WORDS,
    emit_section,
    make_element_fixed,
)
from ..text import strip_html
from .shamela_footnotes import attach_shamela_footnotes

SHAMELA_TITLE_TAG_RE = re.compile(
    r"<span[^>]*data-type=[\"']title[\"'][^>]*>(.*?)</span>", re.S
)
SHAMELA_TOC_ID_RE = re.compile(r"id\s*=\s*[\"']?toc-(\d+)")
NUMBERED_LINE_RE = re.compile(r"^\s*([0-9٠-٩]{1,4})\s*(?:[)\-.،]|[-—])\s*(.+)$")
HADITH_NUMBER_RE = re.compile(r"(?<![0-9٠-٩/])([0-9٠-٩]{1,5}) ?- ")


def read_shamela_pages(source_path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with (DATA_DIR / source_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    # Reading order is shamela_page_id. sequence_num restarts in every part of a
    # multi-volume book, so sorting on it shuffled the volumes page by page (السلسلة
    # الضعيفة: 14 parts, 3,327 part switches); in some books it holds hadith numbers
    # (شرح كتاب التوحيد: 7371). The old key stays as the tie-break for sources
    # without shamela_page_id. `_order` is the page's position in that order.
    out.sort(key=lambda p: (safe_num(p.get("shamela_page_id")), safe_num(p.get("sequence_num")),
                            safe_num(p.get("page_num"))))
    for i, page in enumerate(out):
        page["_order"] = i
    return out


def read_shamela_toc(aux_paths: list[str]) -> list[dict[str, Any]]:
    for p in aux_paths or []:
        if p.endswith("toc.jsonl"):
            out: list[dict[str, Any]] = []
            with (DATA_DIR / p).open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
            return out
    return []


def shamela_segments(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split raw page bodies at title tags into (title, text, page_id, seq) segments."""
    segments: list[dict[str, Any]] = []
    for page in pages:
        body = page.get("body") or ""
        parts = SHAMELA_TITLE_TAG_RE.split(body)
        seq = page.get("sequence_num")
        page_id = page.get("page_id")
        if parts[0].strip():
            segments.append({"title": None, "text": strip_html(parts[0]), "page_id": page_id, "seq": seq})
        for i in range(1, len(parts), 2):
            title = strip_html(parts[i]).strip()
            text = strip_html(parts[i + 1]) if i + 1 < len(parts) else ""
            segments.append({"title": title or None, "text": text, "page_id": page_id, "seq": seq})
    return segments


def page_parts(body: str) -> list[str]:
    """A page body in reading order, split at title tags as `shamela_segments` splits it
    (the titles are parts too)."""
    return [strip_html(p) for p in SHAMELA_TITLE_TAG_RE.split(body or "")]


def group_segments_by_title(segments: list[dict[str, Any]]) -> list[tuple[str | None, str, list[Any]]]:
    """Group segments into (title, text, page_ids) sections; a new title starts a new section."""
    sections: list[tuple[str | None, list[str], list[Any]]] = []
    current_title: str | None = None
    current: list[str] = []
    current_pages: list[Any] = []
    for seg in segments:
        if seg["title"] is not None:
            if current_title is not None or current:
                sections.append((current_title, current, current_pages))
            current_title = seg["title"]
            current = [seg["text"]]
            current_pages = [seg["page_id"]]
        else:
            current.append(seg["text"])
            if seg["page_id"] not in current_pages:
                current_pages.append(seg["page_id"])
    if current_title is not None or current:
        sections.append((current_title, current, current_pages))
    return [(t, "\n".join(c), pages) for t, c, pages in sections]


def split_numbered_units(text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, list[str]]] = []
    current_num: str | None = None
    current: list[str] = []
    for line in (text or "").splitlines():
        m = NUMBERED_LINE_RE.match(line.strip())
        if m:
            if current_num is not None and current:
                items.append((current_num, current))
            current_num = m.group(1)
            current = [m.group(2).strip()]
        elif current_num is not None:
            current.append(line.strip())
    if current_num is not None and current:
        items.append((current_num, current))
    return [(n, "\n".join(c)) for n, c in items if clean_text("\n".join(c))]


def split_numbered_hadith(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Split a run of numbered hadith (`١٢ - «...» ... ١٣ - «...»`) at its numbers.

    Hadith collections such as صحيح الجامع keep a whole chapter under one title, the
    entries run together without line breaks, and numbers also occur inside an entry
    (`الصحيحة ٦٨٥`, a زيادة number in `١٤ - ٦ - «`). A number therefore starts an
    entry only if it follows the previous entry's number by 1 to 3 (a few may be
    missing from the source). Returns the text before the first entry and the
    entries; with fewer than two, the text is not such a run and comes back whole.
    """
    found = [(m.start(), safe_num(m.group(1))) for m in HADITH_NUMBER_RE.finditer(text or "")]
    for i, (start, num) in enumerate(found):
        chain = [(start, num)]
        for pos, n in found[i + 1:]:
            if chain[-1][1] < n <= chain[-1][1] + 3:
                chain.append((pos, n))
        if len(chain) >= 2:
            break
    else:
        return text, []
    ends = [pos for pos, _ in chain[1:]] + [len(text)]
    entries = [(n, text[pos:end]) for (pos, n), end in zip(chain, ends)]
    return text[:chain[0][0]], entries


def parse_shamela(profile: dict[str, Any], state: BuildState) -> None:
    source_path = profile["source_path"]
    pages = read_shamela_pages(source_path)
    toc = read_shamela_toc(profile.get("auxiliary_paths") or [])
    excluded_ids = profile.get("_excluded_record_ids", set())
    review_ids = profile.get("_review_record_ids", set())
    if excluded_ids:
        pages = [pg for pg in pages if str(pg.get("page_id")) not in excluded_ids]
    first_parent, first_view = len(state.parents), len(state.views)
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
    segments = shamela_segments(pages)
    order = 0

    def emit(title: str, body: str, unit_id: str, parent_type: str, element_type: str,
             view_types: tuple[str, str], text_roles: tuple[str, str],
             refs: dict[str, Any] | None = None, pack_words: int = PARAGRAPH_PACK_WORDS,
             hadith_split: bool = False, qanda_pair: bool = False,
             extra_flags: list[str] | None = None) -> None:
        nonlocal order
        order += 1
        emit_section(
            profile, state, book=book, source_path=source_path,
            unit_id=unit_id, title=title, body=body, order=order,
            parent_type=parent_type, element_type=element_type,
            view_types=view_types, text_roles=text_roles, refs=refs,
            pack_words=pack_words, hadith_split=hadith_split, qanda_pair=qanda_pair,
            extra_flags=extra_flags,
        )

    def review_flags(page_ids: list[Any]) -> list[str]:
        flags: list[str] = []
        if any(str(pid) in review_ids for pid in page_ids):
            flags.append("low_arabic_pages_present")
        return flags

    if policy == "shamela_heading_section_parent_v1":
        for title, body, page_ids in group_segments_by_title(segments):
            if not clean_text(body) and not clean_text(title or ""):
                continue
            emit(title or "مقدمة", body, f"section:{order + 1}:{stable_hash(title or '')}",
                 "heading_section", "heading",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 extra_flags=review_flags(page_ids))

    elif policy in ("shamela_hadith_record_parent_v1", "shamela_indexed_hadith_parent_v1",
                    "shamela_bab_hadith_parent_v1"):
        def hadith_refs(number: int | None) -> dict[str, Any]:
            refs = default_refs()
            if number is not None:
                refs["hadith"] = {
                    "collection": profile.get("work_id") or profile["book_id"],
                    "book_number": None,
                    "chapter_id": None,
                    "hadith_number": number,
                    "id_in_book": None,
                }
            return refs

        for title, body, page_ids in group_segments_by_title(segments):
            if not clean_text(body) and not clean_text(title or ""):
                continue
            # Only books whose profile says their hadith carry index numbers, and not
            # their preface: elsewhere a numbered list (إرواء الغليل's commentary
            # `١ - أما حديث أبى هريرة`, ضعيف الترغيب's preface) would be cut as hadith.
            indexed = policy == "shamela_indexed_hadith_parent_v1" and not re.match(r"\s*(?:ال)?مقدم", title or "")
            preface, entries = split_numbered_hadith(body) if indexed else (body, [])
            if not entries:
                m = re.match(r"^\s*([0-9٠-٩]{1,5})", title or "")
                emit(title or "حديث", body, f"hadith:{order + 1}:{stable_hash(title or '')}",
                     "hadith", "hadith",
                     ("heading_title", "hadith_full"), ("heading", "full"),
                     refs=hadith_refs(safe_num(m.group(1)) if m else None), hadith_split=True,
                     extra_flags=review_flags(page_ids))
                continue
            # One parent per numbered hadith; the chapter title (a باب, or Ibn Hibban's
            # ذكر ...) leads each one's title, and text before the first number keeps it.
            if clean_text(preface):
                emit(title or "مقدمة", preface, f"section:{order + 1}:{stable_hash(title or '')}",
                     "heading_section", "heading",
                     ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                     extra_flags=review_flags(page_ids))
            for number, entry in entries:
                entry_title = f"{clean_text(title)[:100]}، {number}" if clean_text(title or "") else f"حديث {number}"
                emit(entry_title, entry, f"hadith:{order + 1}:{stable_hash(entry_title)}",
                     "hadith", "hadith",
                     ("heading_title", "hadith_full"), ("heading", "full"),
                     refs=hadith_refs(number), hadith_split=True,
                     extra_flags=review_flags(page_ids))

    elif policy in ("qanda_pair_parent_v1", "shamela_qa_section_parent_v1",
                    "shamela_qanda_per_question_parent_v1"):
        for title, body, page_ids in group_segments_by_title(segments):
            units = split_qanda_units(body)
            if units:
                for qtitle, qbody in units:
                    emit(qtitle, qbody, f"qanda:{order + 1}:{stable_hash(qtitle)}",
                         "qanda", "qanda",
                         ("heading_title", "qanda_pair"), ("heading", "paragraph"),
                         qanda_pair=True, extra_flags=review_flags(page_ids))
            elif clean_text(body) or clean_text(title or ""):
                emit(title or "سؤال", body, f"qanda:{order + 1}:{stable_hash(title or '')}",
                     "qanda", "qanda",
                     ("heading_title", "qanda_pair"), ("heading", "paragraph"),
                     qanda_pair=True, extra_flags=review_flags(page_ids))

    elif policy in ("tafsir_ayah_parent_v1", "tafsir_ayah_block_parent_v1"):
        for title, body, page_ids in group_segments_by_title(segments):
            if not clean_text(body) and not clean_text(title or ""):
                continue
            refs = default_refs()
            qref = extract_numeric_quran_ref(title or "")
            if qref:
                refs["quran"] = {"surah": qref[0], "ayah_start": qref[1], "ayah_end": qref[2]}
            emit(title or "آية", body, f"ayah:{order + 1}:{stable_hash(title or '')}",
                 "tafsir_ayah_commentary", "tafsir_ayah_commentary",
                 ("heading_title", "tafsir_paragraph"), ("heading", "commentary"),
                 refs=refs, pack_words=TAFSIR_PACK_WORDS, extra_flags=review_flags(page_ids))

    elif policy == "shamela_toc_anchored_section_parent_v1":
        # TOC entries anchored by page_id define section boundaries when the
        # body has no title tags. Sections run from one TOC page to the next.
        page_by_id = {pg["page_id"]: pg for pg in pages}
        toc_ordered = sorted(
            (t for t in toc if t.get("page_id") in page_by_id),
            key=lambda t: (page_by_id[t["page_id"]]["_order"], safe_num(t.get("shamela_title_id"))),
        )
        if toc_ordered:
            for i, t in enumerate(toc_ordered):
                start = page_by_id[t["page_id"]]["_order"]
                end = page_by_id[toc_ordered[i + 1]["page_id"]]["_order"] if i + 1 < len(toc_ordered) else None
                body_pages = [pg for pg in pages if pg["_order"] >= start and (end is None or pg["_order"] < end)]
                body = "\n".join(strip_html(pg.get("body") or "") for pg in body_pages)
                title = clean_text(t.get("title_text") or "")
                refs = default_refs()
                qref = extract_numeric_quran_ref(title)
                if qref:
                    refs["quran"] = {"surah": qref[0], "ayah_start": qref[1], "ayah_end": qref[2]}
                else:
                    aref = extract_ayah_ref(title)
                    surah = surah_from_title(profile.get("title_ar") or "")
                    if aref and surah:
                        refs["quran"] = {"surah": surah, "ayah_start": aref[1], "ayah_end": aref[2]}
                emit(title or "آية", body, f"toc:{t.get('title_id')}",
                     "tafsir_ayah_commentary", "tafsir_ayah_commentary",
                     ("heading_title", "tafsir_paragraph"), ("heading", "commentary"),
                     refs=refs, pack_words=TAFSIR_PACK_WORDS,
                     extra_flags=review_flags([pg["page_id"] for pg in body_pages]))
        else:
            for title, body, page_ids in group_segments_by_title(segments):
                if not clean_text(body) and not clean_text(title or ""):
                    continue
                emit(title or "آية", body, f"ayah:{order + 1}:{stable_hash(title or '')}",
                     "tafsir_ayah_commentary", "tafsir_ayah_commentary",
                     ("heading_title", "tafsir_paragraph"), ("heading", "commentary"),
                     pack_words=TAFSIR_PACK_WORDS, extra_flags=review_flags(page_ids))

    elif policy in ("numbered_item_parent_v1", "shamela_numbered_item_parent_v1",
                    "matn_numbered_point_parent_v1", "poem_numbered_verse_parent_v1"):
        full_text = "\n".join(seg["text"] for seg in segments)
        items = split_numbered_units(full_text)
        if items:
            for num, body in items:
                emit(f"مسألة {num}", body, f"numbered:{num}",
                     "numbered_item", "numbered_item",
                     ("heading_title", "numbered_item_view"), ("heading", "numbered_item"))
        else:
            for title, body, page_ids in group_segments_by_title(segments):
                if clean_text(body) or clean_text(title or ""):
                    emit(title or "مقدمة", body, f"section:{order + 1}:{stable_hash(title or '')}",
                         "heading_section", "heading",
                         ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                         extra_flags=review_flags(page_ids))

    elif policy == "shamela_lesson_per_page_parent_v1":
        for seg in segments:
            if not clean_text(seg["text"]):
                continue
            title = seg["title"] or f"درس {seg['seq']}"
            emit(title, seg["text"], f"lesson:{seg['seq']}:{seg['page_id']}",
                 "lesson", "lesson",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 extra_flags=review_flags([seg["page_id"]]))

    elif policy == "risala_collection_parent_v1":
        for title, body, page_ids in group_segments_by_title(segments):
            if not clean_text(body) and not clean_text(title or ""):
                continue
            emit(title or "رسالة", body, f"risala:{order + 1}:{stable_hash(title or '')}",
                 "risala", "risala",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 extra_flags=review_flags(page_ids))

    elif policy == "shamela_single_unit_parent_v1":
        full_text = "\n".join(seg["text"] for seg in segments)
        emit(profile.get("title_ar") or "", full_text, "treatise:whole",
             "treatise", "treatise",
             ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
             extra_flags=review_flags([seg["page_id"] for seg in segments]))

    else:
        for title, body, page_ids in group_segments_by_title(segments):
            if not clean_text(body) and not clean_text(title or ""):
                continue
            emit(title or "مقدمة", body, f"section:{order + 1}:{stable_hash(title or '')}",
                 "heading_section", "heading",
                 ("heading_title", "paragraph_pack"), ("heading", "paragraph"),
                 extra_flags=review_flags(page_ids))

    # Footnotes come last and only add: views of their own, and text appended to
    # the parents they annotate. Nothing emitted above depends on them.
    state.footnote_counts.update(
        attach_shamela_footnotes(profile, state, pages, first_parent, first_view))
