"""section_marker_detector — the section openers `inline_heading_detector` does not take.

`inline_heading_detector` only trusts a bracketed span (`[فصل ...]`) or a bare `فصل`
that is followed by `في` or a colon. That is the right rule for زاد المعاد, and it
leaves a second family of books untouched: works whose divisions are announced by a
bare keyword on its own line, or — in tafsir — by the ayah being commented on.

Measured over the parents still oversized at 35,000 words, 18 of them carry one of
these markers densely enough to divide on, holding 2,238,357 words between them:

    bare فصل      8 parents      مسألة      3 parents
    الآية N       4 parents      قوله تعالى  2 parents
    bare باب      1 parent

Precision is the whole game, exactly as in the sibling detector. Every pattern here
is anchored to a *line* start rather than merely a sentence end, because these are
books that put their headings on their own line; a keyword sitting mid-paragraph is
prose, not a division. `باب` in particular is a common noun ("door", "chapter of"),
and `قوله تعالى` appears constantly mid-argument whenever an author quotes a verse —
so those two are additionally gated (below) rather than trusted everywhere.

The two tafsir-only markers are `القرآن` structure, not prose:
  - `الآية N`  a bracketed ayah reference standing alone, which in a tafsir is the
               title of the block that comments on it.
  - `قوله تعالى` opening a line, which is how a tafsir announces the next verse.
Both are refused outside tafsir, where the same strings are ordinary citations
inside an argument and cutting on them would sever the argument from its evidence.

Emits one Span per marker covering the marker itself, so a caller cuts immediately
before it and the marker opens the section it introduces.
"""
from __future__ import annotations

import re

from . import Span

# These books carry NO newlines at all — 3a1cb09eeaa9 is 939,342 characters on one
# line, d838d1f1c2df is 1,387,339. So nothing here may anchor on `^`; the only
# available anchor is the end of the previous sentence, or the `-` these editions use
# as a separator. Every pattern below therefore opens at sentence-end and then earns
# its keep with what FOLLOWS the keyword, because the keyword alone is not evidence:
# `قرع باب سيدك` is a door, `القباب` is a dome, and `[التوبة: ٢٨]` is a citation
# sitting mid-argument, not a title.
_AFTER = r"(?:\A|[\r\n]|(?<=[.؟!؛])|(?<=\])|(?<=﴾)|(?<=-))\s*"
_NOT_WORD = r"(?![^\W\d_])"
_D = r"[0-9٠-٩]"          # these editions number in Arabic-Indic digits

# `فصل` introducing a division. Requires a heading-shaped continuation — `في …`,
# a parenthesised title, a colon, or `قال` — which is what separates the observed
# `- فصل (في ذكر سؤال أورده المعترض)` from the noun inside `فصل الخطاب`.
RE_FASL = re.compile(
    rf"{_AFTER}(فصل){_NOT_WORD}(?!\s*الخطاب{_NOT_WORD})\s*(?=[(\[:：]|في{_NOT_WORD}|قال{_NOT_WORD})",
)

# `باب` only when a heading continuation follows. Bare `باب` is refused outright:
# measured on 7f50850872e0, most occurrences are the ordinary noun.
RE_BAB = re.compile(
    rf"{_AFTER}(الباب|باب){_NOT_WORD}\s*(?=[(\[:：]|في{_NOT_WORD}|ما{_NOT_WORD})",
)

# `مسألة:` — the unit these fiqh works are built from, and reliable *with* its colon
# or number. `مسألة` running on into prose is not counted.
RE_MASALA = re.compile(
    rf"{_AFTER}(المسألة|مسألة){_NOT_WORD}\s*(?=[:：]|{_D})",
)

# tafsir only — `قوله تعالى: ﴿…﴾` is how a tafsir announces the next verse it treats.
# The Quranic quote is required, and a preceding connective (`فمثلاً`, `وكذلك أيضاً`,
# `كقوله`) blocks it, because those forms quote a verse as evidence inside an
# argument rather than opening the commentary on it.
RE_QAWLUHU = re.compile(
    rf"{_AFTER}(قوله){_NOT_WORD}\s+(?:تعالى|عز\s+وجل|سبحانه)\s*[:：]\s*(?=[﴿«\"])",
)
RE_CONNECTIVE = re.compile(
    r"(?:فمثلا|مثلا|وكذلك|كذلك|أيضا|نحو|مثل|كقول|ومنه|ولهذا|ونظير)[\sً-ْ]*$"
)

TAFSIR_ONLY = ("qawluhu_taala",)

_PATTERNS = (
    (RE_FASL, "فصل", "fasl"),
    (RE_BAB, "باب", "bab"),
    (RE_MASALA, "مسألة", "masala"),
    (RE_QAWLUHU, "قوله تعالى", "qawluhu_taala"),
)


def detect(text: str, record=None, *, category: str | None = None) -> list[Span]:
    """Structural section openers, as Spans over `text`.

    `category` gates the two tafsir-only forms. It is optional so the detector can be
    called like every other one; when it is not given, those forms are refused rather
    than guessed at — the failure mode of a wrong cut in a fiqh work is severing an
    argument from the verse it rests on, which is worse than leaving a parent long.
    """
    if not text:
        return []
    is_tafsir = (category or "").strip().lower() == "tafsir"
    spans: list[Span] = []
    for pattern, label, form in _PATTERNS:
        if form in TAFSIR_ONLY and not is_tafsir:
            continue
        for m in pattern.finditer(text):
            # A verse quoted as evidence reads identically to one being introduced;
            # only the words immediately before it tell them apart.
            if form == "qawluhu_taala" and RE_CONNECTIVE.search(text[max(0, m.start(1) - 40):m.start(1)]):
                continue
            spans.append(Span(m.start(1), m.end(1), "section_marker", m.group(1),
                              {"heading_text": label, "heading_keyword": label,
                               "form": form}))
    spans.sort(key=lambda s: s.start)

    # Two markers back to back are one boundary, not two — cutting between them would
    # leave a parent holding nothing but its own opener. Same rule as the sibling
    # detector, and it is what keeps `مسألة` lists from exploding into empty parents.
    kept: list[Span] = []
    for span in spans:
        if kept and text[kept[-1].end:span.start].strip() == "":
            continue
        kept.append(span)
    return kept
