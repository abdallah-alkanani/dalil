"""speaker_turn_detector — recognizes speaker changes in transcribed lessons.

Recorded lessons and tape transcripts are dialogue: a questioner speaks, the
shaykh answers, a student interjects. The turn label is the only structure such
a text has — there are no chapters, no numbered sections, nothing else a parser
can grip. Miss it and an entire recorded lesson becomes one parent; this is the
single largest cause of oversized parents in the corpus, 523 of 1,154 audited.

The reason it was missed is that the labels are *inline*, not at line start:

    مداخلة: الحقيقة أرغب أن الأخ إبراهيم ... الشيخ: طيب لا مانع. مداخلة: أعرفك

Every other detector here anchors on `^`, so all of this scrolls past unseen.
This one scans the whole buffer.

The trap is attribution. `قال الشيخ:` introduces a quotation and is NOT a speaker
turn — treating it as one would shred ordinary scholarly prose, where quoting a
shaykh is constant. Measured over the audited parents, attribution accounts for
0.1% of raw matches (12 of 9,072), so the guard costs almost nothing and prevents
a failure that would be silent and severe.

Emits one span per turn. Grouping turns into parents is a policy decision, not a
detector one: a turn can be four words long ("الشيخ: طيب لا مانع") and must not
become a parent by itself.
"""
from __future__ import annotations

import re

from . import Span

# Written as an alternation of whole words rather than a character class so the
# Arabic stays in logical order and cannot be reordered by bidi display.
SPEAKERS = (
    "الشيخ", "شيخنا",
    "طالب", "الطالب", "طلبة", "الطلبة", "الطالبة",
    "السائل", "سائل", "سائلة",
    "القارئ", "قارئ", "القارىء",
    "مداخلة", "المداخلة",
    "المقدم", "مقدم", "المذيع",
    "الضيف", "ضيف",
    "المتصل", "متصل", "متصلة",
    "الجمهور", "الحضور",
)
_ALT = "|".join(SPEAKERS)

# A turn is a speaker word followed by a colon. The lookbehind keeps the match
# from starting inside a longer word (وطالب, المشايخ), since Arabic script has no
# case or spacing cue to lean on.
RE_TURN = re.compile(rf"(?<![^\W\d_])(?:{_ALT})\s*[:：]")

# Attribution frames: the same label, but introducing a quotation.
RE_ATTRIBUTION = re.compile(
    rf"(?:قال|قاله|يقول|فقال|وقال|تقول|سمعت|سئل|وسئل|أجاب|فأجاب|أجابه|عن|قول)\s+(?:{_ALT})\s*[:：]"
)


def detect(text: str, record=None) -> list[Span]:
    if not text:
        return []

    # Character positions where an attribution frame ends, i.e. where the colon
    # of a quoted-speech label sits. Turns landing on those are dropped.
    attributed = {m.end() for m in RE_ATTRIBUTION.finditer(text)}

    spans: list[Span] = []
    for m in RE_TURN.finditer(text):
        if m.end() in attributed:
            continue
        label = m.group(0).rstrip(":：").strip()
        spans.append(Span(
            m.start(), m.end(), "speaker_turn", m.group(0),
            {"speaker": label, "is_shaykh": label in ("الشيخ", "شيخنا")},
        ))
    return spans
