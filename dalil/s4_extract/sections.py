"""One semantic unit (a section, hadith, fatwa, Q&A pair) -> one parent, its
source element, and the retrieval views that get embedded.
"""
from __future__ import annotations

from typing import Any

from .boundaries import (
    BOUNDARY_FLAG_APPLIED,
    BOUNDARY_FLAG_PENDING,
    PARENT_SPLIT_ABOVE,
    PARENT_TARGET_WORDS,
    inline_heading_cuts,
    merge_cuts,
    section_marker_cuts,
    split_at_anchors,
    thin_cuts,
)
from .dialogue import DIALOGUE_TRIGGER_WORDS, split_speaker_turns
from .qanda import MAX_QUESTION_CHARS, split_qanda_pair
from .quran_refs import extract_all_quran_refs
from .records import (
    BuildState,
    add_quran_ref,
    citation,
    clean_text,
    default_refs,
    join_torn_articles,
    make_element,
    make_parent,
    make_view,
    narrator_edges_from_sanad,
    split_hadith_parts,
    split_paragraph_packs,
    stable_hash,
)
from .text import sanitize_markup, strip_source_markers

PARAGRAPH_PACK_WORDS = 256
FATWA_ANSWER_WORDS = 800
TAFSIR_PACK_WORDS = 420


def make_element_fixed(profile, **kw):
    el = make_element(profile, **kw)
    el["author_id"] = profile.get("author_id")
    return el


def make_parent_fixed(profile, **kw):
    p = make_parent(profile, **kw)
    p["author_id"] = profile.get("author_id")
    return p


def make_view_fixed(profile, **kw):
    v = make_view(profile, **kw)
    v["author_id"] = profile.get("author_id")
    return v


def sanitize_refs(refs: dict[str, Any]) -> dict[str, Any]:
    """Parents/views schemas allow only record_id/record_index in refs.source_record."""
    out = dict(refs)
    sr = refs.get("source_record")
    if isinstance(sr, dict):
        out["source_record"] = {
            "record_id": str(sr.get("record_id")) if sr.get("record_id") is not None else None,
            "record_index": sr.get("record_index"),
        }
    return out


def hold_flags(profile: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if profile.get("audit_status") == "hold":
        flags.append("book_level_hold")
    if profile.get("audit_status") == "needs_review":
        flags.append("needs_review")
    return flags


def emit_section(
    profile: dict[str, Any],
    state: BuildState,
    *,
    book: dict[str, Any],
    source_path: str,
    unit_id: str,
    title: str,
    body: str,
    order: int,
    parent_type: str,
    element_type: str,
    view_types: tuple[str, str],
    text_roles: tuple[str, str],
    refs: dict[str, Any] | None = None,
    extra_flags: list[str] | None = None,
    pack_words: int = PARAGRAPH_PACK_WORDS,
    hadith_split: bool = False,
    qanda_pair: bool = False,
    existing_element: dict[str, Any] | None = None,
) -> None:
    """Emit one semantic parent (section) with its element and views.

    Views: a heading/title view plus paragraph-pack views of the body.
    If `existing_element` is given (ketabonline), reuse it instead of
    creating a new element.
    If `hadith_split` is set, the body is split into sanad/matn views with
    distinct view types and text roles, and isnad edges are emitted.
    If `qanda_pair` is set, the body is split into question/answer views.
    """
    # Newlines intact. Torn articles are joined here as clean_text joins them, so
    # raw_body.split() counts the same words as the parent's text_original, which
    # is what the boundary pass measures its word offsets in.
    raw_body = join_torn_articles(strip_source_markers(sanitize_markup(body)))
    body = clean_text(raw_body)
    title = strip_source_markers(sanitize_markup(title))
    if not body and not clean_text(title):
        return

    # An oversized section is divided only where something in the text says it may
    # be divided: a speaker changing, or a boundary a reader chose and recorded.
    # There is deliberately no fallback that cuts at a word count — see the note
    # above `load_semantic_boundaries`. A section we have no boundary for stays
    # whole and carries `needs_semantic_split` so the gap is visible in the corpus
    # and can be sent for a boundary decision, rather than being quietly mangled.
    body_words = len(raw_body.split())
    if body_words > DIALOGUE_TRIGGER_WORDS:
        # Transcribed lessons carry their own boundaries. A speaker change is a
        # real structural signal from the source, so prefer it over any judgement
        # we could ask for: it is the difference between cutting a lesson where
        # the shaykh stops talking and cutting it at a word count. Audited across
        # the corpus this is the largest single cause of oversized parents. This
        # runs at its own low threshold because an exchange is a better unit than
        # a whole lesson even when the lesson is not egregiously long.
        exchanges = split_speaker_turns(raw_body)
        if len(exchanges) > 1:
            for part, (lead, chunk_text) in enumerate(exchanges, start=1):
                base = clean_text(title)[:100]
                part_title = title if part == 1 else f"{base} ({lead or part})"
                emit_section(
                    profile, state, book=book, source_path=source_path,
                    unit_id=f"{unit_id}:turn{part}", title=part_title, body=chunk_text,
                    order=order * 1000 + part, parent_type="dialogue_exchange",
                    element_type=element_type, view_types=view_types,
                    text_roles=text_roles, refs=refs,
                    extra_flags=(extra_flags or []) + ["speaker_turn_split"],
                    pack_words=pack_words, existing_element=None,
                )
            return

    if body_words > PARENT_SPLIT_ABOVE:
        # Two sources of cuts, both found in the text itself, selected as one set.
        #
        # `this_id` is derived from unit_id and order, which are fixed before any
        # split happens — so adding the heading source does not move the key the
        # recorded boundaries are filed under, and a boundary file stays valid
        # across this change.
        this_id = f"parent:{profile['book_id']}:{stable_hash(unit_id + str(order))}"
        recorded = (profile.get("_semantic_boundaries") or {}).get(this_id, [])
        candidates = merge_cuts(recorded, inline_heading_cuts(raw_body),
                                section_marker_cuts(raw_body, profile.get("category")))
        target = profile.get("_parent_target_words") or PARENT_TARGET_WORDS
        pieces = split_at_anchors(raw_body, thin_cuts(candidates, body_words, target))
        if len(pieces) > 1:
            for part, chunk_text in enumerate(pieces, start=1):
                part_title = title if part == 1 else f"{clean_text(title)[:100]} ({part})"
                emit_section(
                    profile, state, book=book, source_path=source_path,
                    unit_id=f"{unit_id}:part{part}", title=part_title, body=chunk_text,
                    order=order * 1000 + part, parent_type=parent_type,
                    element_type=element_type, view_types=view_types,
                    text_roles=text_roles, refs=refs,
                    extra_flags=(extra_flags or []) + [BOUNDARY_FLAG_APPLIED],
                    pack_words=pack_words, existing_element=None,
                )
            return
        extra_flags = (extra_flags or []) + [BOUNDARY_FLAG_PENDING]
    refs = refs or default_refs()
    flags = hold_flags(profile) + (extra_flags or [])
    if existing_element is not None:
        elem_id = existing_element["element_id"]
    else:
        elem_id = f"elem:{profile['book_id']}:{stable_hash(unit_id + str(order))}"
    parent_id = f"parent:{profile['book_id']}:{stable_hash(unit_id + str(order))}"
    section_path = [profile.get("title_ar") or profile["book_id"]]
    section_ids = [f"book:{profile['book_id']}"]
    if clean_text(title):
        section_path.append(clean_text(title)[:120])
        section_ids.append(unit_id)
    if existing_element is None:
        state.elements.append(
            make_element_fixed(
                profile,
                element_id=elem_id,
                source_path=source_path,
                source_record_id=unit_id,
                element_type=element_type,
                level=2,
                parent_element_id=book["element_id"],
                ancestor_ids=[book["element_id"]],
                order_in_parent=order,
                section_path=section_path,
                section_path_ids=section_ids,
                text=body,
                refs=refs,
                quality_flags=flags,
                parse_confidence=0.9,
            )
        )
    view_ids: list[str] = []
    view_key = f"view:{profile['book_id']}:{stable_hash(unit_id + str(order))}"

    def add_view(suffix: str, view_type: str, text_role: str, order_in_parent: int,
                 text: str, view_flags: list[str]) -> None:
        vid = f"{view_key}:{suffix}"
        state.views.append(
            make_view_fixed(
                profile,
                view_id=vid,
                parent_id=parent_id,
                view_type=view_type,
                text_role=text_role,
                order_in_parent=order_in_parent,
                section_path=section_path,
                section_path_ids=section_ids,
                refs=sanitize_refs(refs),
                text=text,
                quality_flags=view_flags,
            )
        )
        view_ids.append(vid)

    def add_packs(text: str, label: str | None, view_type: str, text_role: str,
                  order_base: int, view_flags: list[str]) -> None:
        for j, pack in enumerate(split_paragraph_packs(text, pack_words), start=1):
            add_view(f"{label}:{j:03d}" if label else f"{j:03d}", view_type, text_role,
                     order_base + j, pack, view_flags)

    if clean_text(title):
        add_view("title", view_types[0], text_roles[0], 0, title, flags)
    if body:
        if hadith_split:
            prefix, sanad, matn, hflags = split_hadith_parts(body)
            if prefix:
                # Editorial/takhrij material that precedes the chain. Kept as its own
                # view so it stays retrievable without being mislabelled as isnad.
                add_packs(prefix, "pre", "hadith_commentary", "commentary", 0, flags + hflags)
            if sanad:
                add_packs(sanad, "sanad", "hadith_sanad", "sanad", 100, flags + hflags)
                state.isnad_edges.extend(
                    narrator_edges_from_sanad(
                        sanad, parent_id=parent_id, profile=profile, view_id=view_ids[-1]
                    )
                )
            if matn:
                add_packs(matn, "matn", "hadith_matn", "matn", 200, flags + hflags)
            if not sanad and not matn:
                add_packs(body, "full", "hadith_full", "full", 0, flags + hflags)
        elif qanda_pair:
            question, answer = split_qanda_pair(body)
            if len(question) > MAX_QUESTION_CHARS:
                # A "question" this long is never a question. It happens when a
                # book is assigned a Q&A policy but is really something else
                # (e.g. a lecture-tape transcript), and the marker match swallows
                # the whole record. Fall back to plain paragraph views so the
                # text stays retrievable at a sane granularity.
                question, answer = "", ""
                flags = flags + ["oversized_qanda_fallback"]
            if question:
                add_view("question", "qanda_question", "question", 1, question, flags)
            if answer:
                add_packs(answer, "answer", "qanda_answer", "answer", 2, flags)
            if not question and not answer:
                add_packs(body, None, view_types[1], text_roles[1], 0, flags)
        else:
            add_packs(body, None, view_types[1], text_roles[1], 0, flags)
    state.parents.append(
        make_parent_fixed(
            profile,
            parent_id=parent_id,
            parent_type=parent_type,
            source_element_ids=[elem_id],
            ancestor_ids=[book["element_id"], elem_id],
            section_path=section_path,
            section_path_ids=section_ids,
            refs=sanitize_refs(refs),
            text=body,
            citation=citation(profile, clean_text(title)[:100]),
            child_view_ids=view_ids,
            quality_flags=flags,
        )
    )
    # quran refs: explicit refs from refs + all numeric refs in title/body
    emitted: set[tuple[int, int, int]] = set()
    if refs.get("quran"):
        q = refs["quran"]
        if 1 <= q["surah"] <= 114:
            emitted.add((q["surah"], q["ayah_start"], q["ayah_end"]))
    for qref in extract_all_quran_refs(title) + extract_all_quran_refs(body):
        if qref not in emitted:
            emitted.add(qref)
    for surah, ayah_start, ayah_end in sorted(emitted):
        add_quran_ref(
            state,
            source_kind="parent",
            source_id=parent_id,
            book_id=profile["book_id"],
            surah=surah,
            ayah_start=ayah_start,
            ayah_end=ayah_end,
            match_method="explicit_ref",
            confidence=0.75,
            matched_text=title,
        )
