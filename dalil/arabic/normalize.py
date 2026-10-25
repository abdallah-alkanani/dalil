#!/usr/bin/env python3
"""Canonical Arabic search normalisation for the RAG stores.

Single source of truth. Both the index build (bm25 rebuild / corpus append) and
the query path import from here, because the two MUST agree exactly: any rule
applied on one side and not the other silently makes rows unreachable.

Character classes are written with \\u escapes on purpose. Arabic literals inside
a regex range are unreliable — visual order is not byte order, so two ranges that
look identical on screen can compile to different character sets.

Rules, and why each one is here
-------------------------------
strip tashkeel      REQUIRED. The corpus is fully vocalised (قَالَ رَسُولُ اللَّهِ)
                    while queries never are. Without this, nothing matches.
strip tatweel       Decorative kashida, carries no meaning.
NFKC                Expands ligatures (ﷺ -> صلى الله عليه وسلم) and presentation forms.
أ إ آ ٱ -> ا        Alif seats vary freely in print and in generated text; folding
                    them collides with no distinct word.
ک -> ك, ی -> ي      Persian/Urdu codepoints mapped to their Arabic equivalents.
                    Character-set repair, not orthographic folding.
Arabic-Indic digits Unified to ASCII.
lowercase           For the Latin that appears in transliterations and ids.

Deliberately NOT folded
-----------------------
ى -> ي              Would merge على ("on") with علي (the name Ali), both very common
                    and semantically distinct. Corpus spells ى correctly ~100% of
                    the time, so the fold buys nothing and costs real precision.
ؤ -> و, ئ -> ي      Same reasoning: the corpus is consistent (~0.5% variance), and
                    queries are machine-generated, so the recall these would buy
                    does not materialise.
ة -> ه              Would merge رحمة (mercy) with رحمه ("had mercy on him"), كتابة
                    (writing) with كتابه (his book), وجهة with وجهه. Large precision
                    loss for negligible recall.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic honorific signs, tashkeel, superscript alif, Quranic marks, tatweel.
SEARCH_MARKS_RE = re.compile(
    r"[\u0610-\u061A\u0640\u064B-\u065F\u0670\u06D6-\u06ED]"
)
# Zero-width joiners, bidi controls, BOM.
SEARCH_CONTROL_RE = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFEFF]"
)
SEARCH_TRANSLATION = str.maketrans({
    # alif seats -> bare alif
    "أ": "ا",  # أ
    "إ": "ا",  # إ
    "آ": "ا",  # آ
    "ٱ": "ا",  # ٱ
    # Persian/Urdu codepoints -> Arabic equivalents
    "ک": "ك",  # ک -> ك
    "ی": "ي",  # ی -> ي
    # Extended Arabic-Indic digits
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    # Arabic-Indic digits
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})

# Source-format debris that should never reach the search field. The new corpus
# is cleaned at extraction time; the older 78 books never were, so the bm25
# rebuild strips them here instead of rewriting stored evidence text.
SOURCE_MARKER_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"</?[A-Za-z][^<>]{0,300}>"), " "),          # html tags
    (re.compile(r"#{1,3}\s*\|+\s*(?:CHECK)?"), " "),          # openiti heading bars
    (re.compile(r"(?m)^[ \t]*#{1,3}[ \t]+"), " "),            # openiti line markers
    (re.compile(r"(?<=\s)#{1,3}(?=\s)"), " "),
    (re.compile(r"(?m)(?:^|(?<=\s))(?:AUTO|CHECK|PageBeg|PageEnd)(?=\s|$)"), " "),
    (re.compile(r"@[A-Z]{2}@"), " "),                          # @QB@ / @QE@
    (re.compile(r"\bms\d{1,4}\b"), " "),                       # manuscript milestones
    (re.compile(r"\(\s*¬\s*[0-9٠-٩]{1,4}\s*\)"), " "),  # (¬1) footnotes
    (re.compile(r"_{6,}"), " "),                               # rules
    (re.compile(r"(?:\.\s*){6,}"), " "),                       # dot leaders
    (re.compile(r"[*\-–—=]{6,}"), " "),              # ==== separators
    (re.compile(r"https?://\S+"), " "),
)


# Cheap pre-check before running the substitution list. Latin letters are rare in
# this corpus, so their presence is a good hint that a marker (AUTO, CHECK, msN,
# a URL, an html tag) may be present. Keep this broad: a guard that is too narrow
# silently skips cleaning, which is how 'AUTO' survived an earlier version.
MARKER_HINT_RE = re.compile(r"[#@¬_*=<>A-Za-z]|\.\s*\.\s*\.|[-–—]{6,}")


def strip_source_markers(text: str) -> str:
    """Remove source-format debris. Cheap guard first: most text has none."""
    if not text or not MARKER_HINT_RE.search(text):
        return text
    for pattern, repl in SOURCE_MARKER_SUBS:
        text = pattern.sub(repl, text)
    return text


def normalize_for_search(text: str, clean_markers: bool = True) -> str:
    """Recall-oriented Arabic search form. Never applied to displayed evidence."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    if clean_markers:
        value = strip_source_markers(value)
    value = SEARCH_CONTROL_RE.sub("", value)
    value = SEARCH_MARKS_RE.sub("", value)
    value = value.translate(SEARCH_TRANSLATION).lower()
    value = "".join(c if c.isalnum() or c == "_" else " " for c in value)
    return re.sub(r"\s+", " ", value).strip()
