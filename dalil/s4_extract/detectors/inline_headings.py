"""inline_heading_detector — structural headings that sit inside a line, not on one.

`heading_detector` reads the text line by line, anchors every pattern on `^`, and
skips any line longer than 120 characters. That is correct for sources that put a
heading on its own line, and blind to the ones that do not. زاد المعاد ships an
entire book as a single line:

    [فصل في هديه صلى الله عليه وسلم في الأقضية] [فصل جواز الحبس] وليس الغرض من ذلك
    ذكر التشريع العام ... [فصل في حكمه فيمن قتل عبده] روى الأوزاعي عن عمرو بن شعيب

Every one of those headings is real, and every one of them scrolls past unseen —
which is how a 182,016-word parent happens. Measured over the parents still
oversized after the speaker-turn fix, 229 such headings sit inside one parent and
932 bracketed spans inside another.

Precision is the whole game here, because the same brackets carry things that are
NOT headings — Quran references (`[البقرة: 255]`), editor footnote markers, page
numbers. So a bracketed span counts as a heading only when its contents *open* with
a structural keyword. `[فصل جواز الحبس]` is a heading; `[البقرة: 255]` is not, and
is left alone rather than guessed at.

Emits one Span per heading, covering the bracketed span itself, so a caller can cut
immediately before it and keep the heading with the section it introduces.
"""
from __future__ import annotations

import re

from . import Span

# Whole-word alternation, never a character range: a literal Arabic range inside a
# regex class is reordered by bidi and the source stops describing what it matches.
HEADING_KW = (
    "فصل", "فصول", "الفصل",
    "باب", "الباب", "أبواب",
    "كتاب", "الكتاب",
    "مسألة", "المسألة", "مسائل",
    "مبحث", "المبحث", "مطلب", "المطلب",
    "قاعدة", "القاعدة",
    "تنبيه", "خاتمة", "مقدمة", "تمهيد", "فوائد", "فائدة",
)
_KW = "|".join(HEADING_KW)

# A bracketed span whose contents begin with a structural keyword. The keyword must
# be a whole word — `[بابل]` (Babylon) is not a باب.
RE_BRACKET_HEADING = re.compile(
    rf"\[\s*(?:AUTO|CHECK)?\s*((?:{_KW})(?![^\W\d_])[^\]\n]{{0,118}})\]"
)

# A bare `فصل` standing between two sentences, the other form زاد المعاد uses. It
# must be followed by a heading-ish continuation (في / a colon) or be alone before
# more prose, and must not be the ordinary verb/noun inside a phrase such as
# `من هذا الفصل` or `فصل الخطاب`, so a preceding particle or preposition blocks it.
RE_BARE_FASL = re.compile(
    r"(?:(?<=[.؟!؛])|(?<=\])|(?<=\n))\s*(فصل)(?![^\W\d_])\s*(?=في\s|[:：])"
)


def detect(text: str, record=None) -> list[Span]:
    if not text:
        return []
    spans: list[Span] = []
    for m in RE_BRACKET_HEADING.finditer(text):
        title = " ".join(m.group(1).split())
        spans.append(Span(m.start(), m.end(), "inline_heading", m.group(0),
                          {"heading_text": title, "heading_keyword": title.split()[0],
                           "form": "bracketed"}))
    for m in RE_BARE_FASL.finditer(text):
        spans.append(Span(m.start(1), m.end(1), "inline_heading", m.group(1),
                          {"heading_text": "فصل", "heading_keyword": "فصل",
                           "form": "bare"}))
    spans.sort(key=lambda s: s.start)

    # Two headings back to back (`[فصل ...] [فصل ...]`) are one boundary, not two:
    # cutting between them would leave the first as a parent containing only its own
    # title. Keep the first and let its section run to the next real gap.
    kept: list[Span] = []
    for span in spans:
        if kept and text[kept[-1].end:span.start].strip() == "":
            continue
        kept.append(span)
    return kept
