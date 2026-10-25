"""Corpus records and their building blocks: element / parent / retrieval-view
records, content-derived ids, paragraph packs, hadith sanad/matn splitting and
narrator (isnad) edges. Shared by both corpus builders.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# NOTE: the three regexes below are applied to text that has already been through
# normalize_arabic(), which folds أ/إ/آ/ٱ -> ا and ة -> ه. Their alternatives must
# therefore be written in the *normalized* spelling (اخبرنا, not أخبرنا), otherwise
# the hamzated narrator verbs can never match.
HADITH_START_RE = re.compile(r"^\s*(?:حدثنا|حدثني|اخبرنا|اخبرني|انبانا|ثنا|سمعت|عن)\b")
# Start-of-chain anchor used to locate where an isnad actually begins.
SANAD_ANCHOR_RE = re.compile(r"(?:حدثنا|حدثني|اخبرنا|اخبرني|انبانا|ثنا)\s+|عن\s+[ء-ي]")
# How far back from the matn transition we are willing to look for the isnad start.
SANAD_LOOKBEHIND_CHARS = 600
# A real isnad names a narrator every few words. If two chain markers are further
# apart than this, the earlier one belongs to different material, not to this chain.
SANAD_MAX_LINK_GAP_CHARS = 120


def sanad_start_offset(normalized: str, matn_start: int) -> int | None:
    """Return where the isnad feeding `matn_start` begins, or None if there is none.

    Anchors are collected inside a bounded look-behind window and then walked
    backwards from the matn: the chain is extended only while consecutive markers
    stay close together, so an unrelated earlier 'عن' cannot drag the sanad back
    across intervening prose.
    """
    window_start = max(0, matn_start - SANAD_LOOKBEHIND_CHARS)
    anchors = [m.start() for m in SANAD_ANCHOR_RE.finditer(normalized, window_start, matn_start)]
    if not anchors:
        return None
    start = anchors[-1]
    if matn_start - start > SANAD_MAX_LINK_GAP_CHARS:
        return None
    for earlier in reversed(anchors[:-1]):
        if start - earlier > SANAD_MAX_LINK_GAP_CHARS:
            break
        start = earlier
    return start


MATN_TRANSITION_RE = re.compile(
    r"(?:"
    r"قال\s+رسول\s+الله"
    r"|(?<![ء-ي])ان\s+رسول\s+الله"
    r"|سمعت\s+رسول\s+الله"
    r"|رسول\s+الله.{0,80}?\b(?:قال|يقول|فقال)\b"
    r"|عن\s+النبي"
    r"|سمعت\s+النبي"
    r"|(?<![ء-ي])ان\s+النبي"
    r"|قال\s+النبي"
    r"|النبي.{0,80}?\b(?:قال|يقول|فقال)\b"
    r")"
)
NARRATOR_SPLIT_RE = re.compile(r"\b(?:حدثنا|حدثني|اخبرنا|اخبرني|انبانا|ثنا|سمعت|عن)\s+")


@dataclass
class BuildState:
    elements: list[dict[str, Any]]
    parents: list[dict[str, Any]]
    views: list[dict[str, Any]]
    quran_refs: list[dict[str, Any]]
    isnad_edges: list[dict[str, Any]]
    report_rows: list[dict[str, Any]]
    # Footnote counts of the book being parsed; moved into its coverage row.
    footnote_counts: Counter = field(default_factory=Counter)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def safe_num(text: Any, default: int = 0) -> int:
    if text is None:
        return default
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    try:
        return int(str(text).translate(trans).strip())
    except ValueError:
        return default


# A definite article torn from its word by the source's conversion ("ا لجواب" for
# "الجواب"): 171 views in about 50 books, 2026-10-16 audit. The alif must not follow
# any Arabic character, harakat and Quranic marks included, so a word that ends in
# tanween alif ("مؤخَّرًا لفائدتين") or an Uthmani verse ("قَالُوٓا۟ لَن") is left alone.
# Nor is a stray alif before a word that already starts with a lam of its own
# ("ا للمسلمين", "ا لأنهم"), except لله, which is الله torn.
TORN_ARTICLE_RE = re.compile(r"(?<![؀-ۿ])ا\s+ل(?=[؀-ۿ]{2,})(?!ل(?!ه\b)|أن)")


def join_torn_articles(text: str) -> str:
    """Rejoin torn articles, keeping every other character, newlines included."""
    return TORN_ARTICLE_RE.sub("ال", text)


def clean_text(text: str) -> str:
    text = " ".join((text or "").split())
    return join_torn_articles(text) if "ا ل" in text else text


# Structure-matching form used by the parsers (sanad/matn markers, surah names).
# This is NOT the search normalisation (dalil.arabic.normalize): it also folds
# ى -> ي and ة -> ه, which search deliberately does not.
_NORMALIZE_TABLE = str.maketrans(
    {**dict.fromkeys("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0640"),
     "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}
)


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    return clean_text(text.translate(_NORMALIZE_TABLE))


def normalize_arabic_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize Arabic while retaining a normalized-character -> source offset map."""
    marks = {"\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652", "\u0670", "\u0640"}
    replacements = {"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}
    chars: list[str] = []
    offsets: list[int] = []
    previous_space = False
    for idx, ch in enumerate(text or ""):
        if ch in marks:
            continue
        ch = replacements.get(ch, ch)
        if ch.isspace():
            if chars and not previous_space:
                chars.append(" ")
                offsets.append(idx)
                previous_space = True
            continue
        chars.append(ch)
        offsets.append(idx)
        previous_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def default_refs() -> dict[str, Any]:
    return {"quran": None, "hadith": None, "openiti": None, "source_record": None}


def token_counts(text: str) -> list[dict[str, Any]]:
    # Placeholder estimator until model-specific tokenization is introduced.
    # Arabic whitespace tokens are enough for structural distribution checks.
    count = len(clean_text(text).split()) if text else 0
    return [{"tokenizer_name": "whitespace_ar_v1", "token_count": count}]


def base_common(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "book_id": profile["book_id"],
        "work_id": profile.get("work_id"),
        "author_id": None,
        "source_corpus": profile["source_corpus"],
        "file_format": profile["file_format"],
        "parser_profile": profile["parser_profile"],
        "parent_policy": profile["parent_policy"],
        "view_policy": profile["view_policy"],
        "recommended_detectors": profile.get("recommended_detectors", []),
    }


def make_element(
    profile: dict[str, Any],
    *,
    element_id: str,
    source_path: str,
    source_record_id: str,
    element_type: str,
    level: int,
    parent_element_id: str | None,
    ancestor_ids: list[str],
    order_in_parent: int,
    section_path: list[str],
    section_path_ids: list[str],
    text: str,
    refs: dict[str, Any] | None = None,
    char_span: list[int] | None = None,
    quality_flags: list[str] | None = None,
    parse_confidence: float = 1.0,
) -> dict[str, Any]:
    common = base_common(profile)
    cleaned = clean_text(text)
    return {
        "schema_version": "dalil.element.v1",
        **common,
        "element_id": element_id,
        "source_path": source_path,
        "source_record_id": source_record_id,
        "element_type": element_type,
        "level": level,
        "parent_element_id": parent_element_id,
        "ancestor_ids": ancestor_ids,
        "order_in_parent": order_in_parent,
        "section_path": section_path,
        "section_path_ids": section_path_ids,
        "refs": refs or default_refs(),
        "text_original": text,
        "text_clean": cleaned,
        "normalized_text": normalize_arabic(text),
        "char_span": char_span or [0, len(text or "")],
        "token_counts": token_counts(text),
        "parser": {
            "parser_name": profile["parser_profile"],
            "parser_version": "v1",
            "parse_confidence": parse_confidence,
            "detectors_used": profile.get("recommended_detectors", []),
        },
        "quality_flags": sorted(set(quality_flags or [])),
    }


def make_parent(
    profile: dict[str, Any],
    *,
    parent_id: str,
    parent_type: str,
    source_element_ids: list[str],
    ancestor_ids: list[str],
    section_path: list[str],
    section_path_ids: list[str],
    refs: dict[str, Any],
    text: str,
    citation: str,
    child_view_ids: list[str],
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "dalil.parent.v1",
        "parent_id": parent_id,
        "book_id": profile["book_id"],
        "work_id": profile.get("work_id"),
        "author_id": None,
        "category": profile["category"],
        "subcategory": profile.get("subcategory"),
        "parent_type": parent_type,
        "source_element_ids": source_element_ids,
        "ancestor_ids": ancestor_ids,
        "section_path": section_path,
        "section_path_ids": section_path_ids,
        "refs": refs,
        "text_for_synthesis": clean_text(text),
        "text_original": text,
        "normalized_text": normalize_arabic(text),
        "citation": citation,
        "child_view_ids": child_view_ids,
        "token_counts": token_counts(text),
        "parser_profile": profile["parser_profile"],
        "parent_policy": profile["parent_policy"],
        "view_policy": profile["view_policy"],
        "quality_flags": sorted(set(quality_flags or [])),
    }


def make_view(
    profile: dict[str, Any],
    *,
    view_id: str,
    parent_id: str,
    view_type: str,
    text_role: str | None,
    order_in_parent: int,
    section_path: list[str],
    section_path_ids: list[str],
    refs: dict[str, Any],
    text: str,
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    role = f"\nنوع النص: {text_role}" if text_role else ""
    embedding_text = (
        f"الكتاب: {profile.get('title_ar') or profile['book_id']}\n"
        f"التصنيف: {profile['category']}\n"
        f"الموضع: {' > '.join(section_path)}{role}\n"
        f"النص: {clean_text(text)}"
    )
    return {
        "schema_version": "dalil.retrieval_view.v1",
        "view_id": view_id,
        "parent_id": parent_id,
        "book_id": profile["book_id"],
        "work_id": profile.get("work_id"),
        "author_id": None,
        "category": profile["category"],
        "subcategory": profile.get("subcategory"),
        "source_corpus": profile["source_corpus"],
        "file_format": profile["file_format"],
        "view_type": view_type,
        "text_role": text_role,
        "order_in_parent": order_in_parent,
        "section_path": section_path,
        "section_path_ids": section_path_ids,
        "refs": refs,
        "display_text": clean_text(text),
        "normalized_text": normalize_arabic(text),
        "embedding_text": embedding_text,
        "token_counts": token_counts(embedding_text),
        "embedding_profile": {"context_header_template": "arabic_structural_v1", "late_chunking_span": None},
        "parser_profile": profile["parser_profile"],
        "parent_policy": profile["parent_policy"],
        "view_policy": profile["view_policy"],
        "quality_flags": sorted(set(quality_flags or [])),
    }


def add_quran_ref(
    state: BuildState,
    *,
    source_kind: str,
    source_id: str,
    book_id: str,
    surah: int,
    ayah_start: int,
    ayah_end: int,
    match_method: str,
    confidence: float,
    matched_text: str,
    flags: list[str] | None = None,
) -> None:
    state.quran_refs.append(
        {
            "schema_version": "dalil.quran_ref_index.v1",
            "quran_ref_id": f"qref:{source_id}:{surah}:{ayah_start}-{ayah_end}",
            "surah": surah,
            "ayah_start": ayah_start,
            "ayah_end": ayah_end,
            "source_kind": source_kind,
            "source_id": source_id,
            "book_id": book_id,
            "match_method": match_method,
            "confidence": confidence,
            "matched_text": clean_text(matched_text)[:500],
            "quality_flags": sorted(set(flags or [])),
        }
    )


def citation(profile: dict[str, Any], suffix: str) -> str:
    title = profile.get("title_ar") or profile["book_id"]
    return f"{title}، {suffix}" if suffix else title


def split_paragraph_packs(text: str, max_words: int = 420) -> list[str]:
    max_unit_words = max(max_words * 2, max_words + 150)

    def split_overlong_unit(unit: str) -> list[str]:
        unit = clean_text(unit)
        if len(unit.split()) <= max_unit_words:
            return [unit] if unit else []
        normalized_breaks = re.sub(r"([.؟!؛:])\s*", r"\1\n", unit)
        normalized_breaks = re.sub(
            r"\s+(?=(?:قال|قوله|فقال|حدثنا|أخبرنا|ثم|وأما|وأما قوله|واختلف|فإن قال|والصواب|يعني)\s)",
            "\n",
            normalized_breaks,
        )
        fragments = [frag.strip() for frag in normalized_breaks.splitlines() if frag.strip()]
        split_fragments: list[str] = []
        for frag in fragments:
            words = frag.split()
            if len(words) <= max_unit_words:
                split_fragments.append(frag)
                continue
            for start in range(0, len(words), max_words):
                split_fragments.append(" ".join(words[start : start + max_words]))
        return split_fragments

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(split_overlong_unit(paragraph))
    for paragraph in units:
        words = len(paragraph.split())
        if current and current_words + words > max_words:
            chunks.append("\n".join(current))
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += words
    if current:
        chunks.append("\n".join(current))
    return chunks or ([text] if text.strip() else [])


def split_hadith_parts(text: str) -> tuple[str, str, str, list[str]]:
    """Split a hadith body into (prefix, sanad, matn, flags).

    `prefix` is any material that precedes the isnad — in takhrij and hadith
    criticism works this is editorial commentary, and it must be preserved
    rather than mislabelled as part of the chain. `sanad` is the narrator chain
    itself and `matn` is the prophetic text. Any of the three may be empty; the
    concatenation of the three always reconstructs the cleaned input.
    """
    cleaned = clean_text(text)
    flags: list[str] = []
    if not cleaned:
        return "", "", "", ["empty_hadith_text"]
    normalized, offsets = normalize_arabic_with_offsets(cleaned)
    match = MATN_TRANSITION_RE.search(normalized)
    if match and match.start() > 20 and match.start() < len(offsets):
        # The matn transition tells us where the matn begins, but everything before
        # it is not automatically an isnad. Anchor the sanad to the first plausible
        # chain marker inside a bounded look-behind window instead of swallowing the
        # whole prefix, and keep whatever precedes that anchor as `prefix`.
        anchor = sanad_start_offset(normalized, match.start())
        source_start = offsets[match.start()]
        if anchor is None:
            flags.append("no_sanad_detected")
            return cleaned[:source_start].strip(), "", cleaned[source_start:].strip(), flags
        sanad_start = offsets[anchor]
        if anchor > 0:
            flags.append("sanad_start_anchored")
        return (
            cleaned[:sanad_start].strip(),
            cleaned[sanad_start:source_start].strip(),
            cleaned[source_start:].strip(),
            flags,
        )
    quote_positions = [pos for pos in (cleaned.find('"'), cleaned.find("«"), cleaned.find("”")) if pos > 20]
    if quote_positions and HADITH_START_RE.match(normalized):
        source_start = min(quote_positions)
        flags.append("sanad_matn_split_quote_boundary")
        return "", cleaned[:source_start].strip(), cleaned[source_start:].strip(), flags
    if HADITH_START_RE.match(normalized):
        flags.append("sanad_matn_split_low_confidence")
        return "", "", cleaned, flags
    flags.append("no_sanad_detected")
    return "", "", cleaned, flags


def split_hadith_sanad_matn(text: str) -> tuple[str, str, list[str]]:
    """Legacy three-part contract: any prefix is folded back onto the sanad."""
    prefix, sanad, matn, flags = split_hadith_parts(text)
    if prefix and sanad:
        sanad = f"{prefix} {sanad}"
    elif prefix:
        sanad = prefix
    return sanad, matn, flags


def narrator_edges_from_sanad(sanad: str, *, parent_id: str, profile: dict[str, Any], view_id: str) -> list[dict[str, Any]]:
    if not sanad:
        return []
    normalized_sanad = normalize_arabic(sanad)
    pieces = [clean_text(p) for p in NARRATOR_SPLIT_RE.split(normalized_sanad) if clean_text(p)]
    narrators = [p.split("،")[0].split(" قال")[0].split(" انه ")[0].strip() for p in pieces]
    narrators = [n for n in narrators if 2 <= len(n) <= 80]
    edges = []
    for idx, (a, b) in enumerate(zip(narrators, narrators[1:]), start=1):
        edges.append(
            {
                "schema_version": "dalil.isnad_edge.v1",
                "edge_id": f"edge:{parent_id}:{idx:03d}",
                "hadith_parent_id": parent_id,
                "collection": profile.get("work_id") or profile["book_id"],
                "book_id": profile["book_id"],
                "narrator_a": a,
                "narrator_b": b,
                "order": idx,
                "extraction_confidence": 0.55,
                "source_view_id": view_id,
                "quality_flags": ["heuristic_isnad_edge"],
            }
        )
    return edges
