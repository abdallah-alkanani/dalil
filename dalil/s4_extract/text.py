"""Source-text cleanup applied to everything the extractor emits: HTML/XML debris,
symbol-font (Private Use Area) codepoints, honorific ligatures, and the control
markers of the source formats.
"""
from __future__ import annotations

import html as html_mod
import re

HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    text = HTML_TAG_RE.sub("", text or "")
    return html_mod.unescape(text)


# Fallback stripper, used when lxml is unavailable or refuses a fragment.
# Bounded so a stray '<' cannot swallow a page of Arabic.
MARKUP_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z!][^<>]{0,400}>")
# Truncated tags such as a trailing '<a' left behind by upstream cleaning.
MARKUP_DANGLING_RE = re.compile(r"<\s*/?\s*[A-Za-z][A-Za-z0-9]{0,15}(?=\s|$)")
try:
    import lxml.html as _lxml_html
except ImportError:  # pragma: no cover - fallback path
    _lxml_html = None


def _sanitize_markup_regex(text: str) -> str:
    out = MARKUP_TAG_RE.sub(" ", text)
    out = MARKUP_DANGLING_RE.sub(" ", out)
    if "&" in out:
        out = html_mod.unescape(out)
    return out


# Structural markers from the source formats that must not reach the embedded
# text. These are not content: OpenITI uses '#'/'###' to mark prose and heading
# lines, 'msNNN' as manuscript milestones (injected mid-sentence, splitting
# words), '@QB@/@QE@' to delimit Quran quotations, and Shamela carries '(¬N)'
# footnote references whose targets are not in the body.
SOURCE_MARKER_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenITI 'uncertain heading' annotation, e.g. '### || CHECK'
    (re.compile(r"#{1,3}\s*\|+\s*CHECK\s*"), " "),
    # Quran quote delimiters become the ornate brackets the rest of the corpus uses
    (re.compile(r"@QB@\s*"), "﴿"),
    (re.compile(r"\s*@QE@"), "﴾"),
    (re.compile(r"@[A-Z]{2}@"), " "),
    # manuscript milestones, e.g. 'وخلع ms01 عليه' — single-digit forms exist too
    (re.compile(r"\bms\d{1,4}\b"), " "),
    # stray halves of '@QB@'/'@QE@' left when one '@' was consumed elsewhere
    (re.compile(r"\bQ[BE]@|@Q[BE]\b"), " "),
    # URLs and bare site links left by HTML sources
    (re.compile(r"https?://\S+|\b\w+\.php\?\S*"), " "),
    # OpenITI '$' page/section and '%' verse separators
    (re.compile(r"(?m)(?:^|(?<=\s))[$%]+(?=\s|$)"), " "),
    # footnote references '(¬1)' / '(¬١)'
    (re.compile(r"\(\s*¬\s*[0-9٠-٩]{1,4}\s*\)"), " "),
    # heading bars and bare '#' separators, at line start or standing alone
    (re.compile(r"#{1,3}\s*\|+\s*"), " "),
    (re.compile(r"(?m)^[ \t]*#{1,3}[ \t]+"), " "),
    (re.compile(r"(?<=\s)#{1,3}(?=\s)"), " "),
    (re.compile(r"(?m)^[ \t]*#{1,3}[ \t]*$"), " "),
    # OpenITI editorial annotations left as bare Latin words in Arabic prose
    (re.compile(r"(?m)(?:^|(?<=\s))(?:AUTO|CHECK|PageBeg|PageEnd)(?=\s|$)"), " "),
    # CSS that leaked out of an HTML source (braces already transliterated to ﴿﴾)
    (re.compile(
        r"(?:font-family|font-size|font-weight|text-align|line-height|margin|padding"
        r"|background|color|direction)\s*:\s*[^;﴾}\n]{0,90}[;﴾}]?"), " "),
    # rules and leaders used as visual separators in print
    (re.compile(r"_{6,}"), " "),
    (re.compile(r"(?:\.\s*){6,}"), " "),
    (re.compile(r"[*\-–—=]{6,}"), " "),
)
# Honorific ligatures that collapse a whole phrase into one rare codepoint. The
# expansions are the official Unicode names for these characters. Spelling them
# out lets the tokenizer see the actual words, so 'صلى الله عليه وسلم' written as
# a ligature matches the same phrase written normally. NFKC already expands ﷺ and
# ﷻ; the U+FD40..FD4F block and ﷽/﷿ have no decomposition and need this table.
HONORIFIC_EXPANSIONS = {
    "﵀": "رحمه الله", "﵁": "رضي الله عنه", "﵂": "رضي الله عنها",
    "﵃": "رضي الله عنهم", "﵄": "رضي الله عنهما", "﵅": "رضي الله عنهن",
    "﵆": "صلى الله عليه وآله", "﵇": "عليه السلام", "﵈": "عليهم السلام",
    "﵉": "عليهما السلام", "﵊": "عليه الصلاة والسلام", "﵋": "قدس سره",
    "﵌": "صلى الله عليه وآله وسلم", "﵍": "عليها السلام",
    "﵎": "تبارك وتعالى", "﵏": "رحمهم الله",
    "ﷺ": "صلى الله عليه وسلم", "ﷻ": "جل جلاله",
    "﷽": "بسم الله الرحمن الرحيم", "﷿": "عز وجل",
}
HONORIFIC_RE = re.compile("[" + "".join(HONORIFIC_EXPANSIONS) + "]")
# Cheap pre-check. Latin letters are rare in this corpus, so their presence is a
# good signal that a marker (AUTO, CHECK, msN, font-family, a URL) may be there.
MARKER_HINT_RE = re.compile(r"[#@¬_*=$%A-Za-z]|\.\s*\.\s*\.|[-–—]{6,}")
# Documents produced with 2000s-era Arabic symbol fonts (AGA / Traditional Arabic)
# encode honorifics and Quran brackets as Private Use Area codepoints. They carry
# real meaning but are invisible to any tokenizer, so they are expanded rather than
# dropped. Each mapping below was confirmed from surrounding context in the source:
# F065/F072 always follow the Prophet's name, F074 follows a single Companion,
# F079 follows الصحابة, F075 follows an angel or earlier prophet, and F07D/F07B
# occur as a matched pair wrapping Quranic quotation.
PUA_EXPANSIONS = {
    "\uf065": " صلى الله عليه وسلم ",
    "\uf072": " صلى الله عليه وسلم ",
    "\uf074": " رضي الله عنه ",
    "\uf079": " رضي الله عنهم ",
    "\uf075": " عليه السلام ",
    "\uf07d": "\ufd3f",
    "\uf07b": "\ufd3e",
}
PUA_RE = re.compile("[\ue000-\uf8ff]")


def expand_private_use(text: str) -> str:
    """Replace symbol-font codepoints with the Arabic they stand for."""
    if not text or not PUA_RE.search(text):
        return text
    return PUA_RE.sub(lambda m: PUA_EXPANSIONS.get(m.group(0), " "), text)


def strip_source_markers(text: str) -> str:
    """Remove source-format control markers from text destined for embedding."""
    text = expand_private_use(text)
    if not text:
        return text
    if HONORIFIC_RE.search(text):
        text = HONORIFIC_RE.sub(lambda m: f" {HONORIFIC_EXPANSIONS[m.group(0)]} ", text)
    if not MARKER_HINT_RE.search(text):
        return text
    for pattern, repl in SOURCE_MARKER_SUBS:
        text = pattern.sub(repl, text)
    return text


def sanitize_markup(text: str) -> str:
    """Strip HTML/XML debris from source text before it reaches the corpus.

    Shamela bodies were already run through strip_html(), but OpenITI TXT and
    Ketabonline JSON also carry embedded '</span>' and '<a href=...>' fragments,
    and those parsers never stripped anything. Applying this centrally means the
    text that gets embedded is free of markup regardless of which parser produced
    it.

    lxml does the parsing because it copes with the malformed fragments these
    sources actually contain (unclosed tags, unbalanced quotes). It is guarded
    two ways: a cheap substring test so the ~99% of clean text never reaches the
    parser at all, and a length check so a fragment lxml mis-parses into near
    nothing falls back to the regex stripper instead of silently losing content.
    """
    if not text or ("<" not in text and "&" not in text):
        return text
    fallback = _sanitize_markup_regex(text)
    if _lxml_html is None:
        return fallback
    try:
        parsed = _lxml_html.fromstring(text).text_content()
    except Exception:
        return fallback
    # A correct parse never loses much beyond the markup itself; if it did,
    # trust the conservative regex instead.
    if len(parsed) < len(fallback) * 0.75:
        return fallback
    return parsed
