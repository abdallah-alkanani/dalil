"""Shamela 4 footnotes: the `footnotes` field of pages.jsonl.

Shamela keeps a page's footnotes (حواشي) out of its body, and the body parser
never read them. That field is where much of the hadith grading lives: every
grade in ضعيف الجامع الصغير (`<s0> (ضعيف)` on each one-hadith page), al-Albani's
takhrij in هداية الرواة, and, in القول المفيد, the whole sharh (the body holds the
matn). 197 of the 241 Shamela books carry footnotes, on 74,207 of 218,629 pages.

They are added after the book's parents are emitted, and only added: no
element, view or id that exists without them changes. Each parent that receives
footnotes gets them appended to `text_for_synthesis` after a blank line and
`الحواشي:` (so the answering model reads them with the text they annotate), one
per line, and gets `footnote` views, packed like paragraphs, with ids
`{view_key}:fn:{NNN}`. A line reads `(١) [عند: «…»] text`: the marker as printed
(minus Shamela's `¬`), then, for a footnote located by its marker, the six words
before the marker. Numbering restarts on every page and `(¬N)` markers are
removed from body text, so in a parent holding several hadith the number alone
would not say which one a grade belongs to.

The field, as found in the data:
- one footnote per line (lines split on `\\r`); a footnote can run over lines
- markers: `(¬١)` and `[¬*]` (Shamela's own; the body carries the same marker,
  which `strip_source_markers` removes from body text), `(١)`, `[١]`, `(*)`
  as typed in the edition, and bare `١ ...` in some books (body `فأجابهم١`)
- `=` opening a page's footnotes continues the previous page's last footnote,
  and a trailing `=`/`==` announces it
- text with no marker: ضعيف الجامع's grades, editor's notes, whole commentaries

Assignment rule, one page at a time:
1. *marker*: the footnote's marker is found in the page body, and the words
   around it are found in a parent: that parent gets it. The body is searched
   in order, so the n-th footnote's marker is looked for after the (n-1)-th's.
2. *page*: otherwise (no marker, marker not in the body, or its context not
   found) the parent holding most of the page gets it: the parent that holds
   the middle of the page's longest text segment.
3. *neighbour*: if no text of the page is found in any parent (pages whose body
   is only dots, or text a policy does not keep), the parent where the last
   located page's text was found.
A continuation goes with the footnote it continues.

Parents are located by text, not by tracking positions through each parent
policy: the words around a marker are cleaned exactly as `emit_section` cleans
a body, then searched in the book's parents in reading order, forward from
where the previous page was found. That works for every policy, including
sections later divided by speaker turns or semantic boundaries.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..records import BuildState, clean_text, safe_num, split_paragraph_packs
from ..sections import PARAGRAPH_PACK_WORDS, make_view_fixed
from ..text import sanitize_markup, strip_html, strip_source_markers

FOOTNOTE_VIEW_TYPE = "footnote"
FOOTNOTE_TEXT_ROLE = "footnote"
FOOTNOTES_HEADER = "الحواشي:"

_D = r"0-9\u0660-\u0669"
_ID = rf"([{_D}]{{1,4}}|\*)"
# '(¬١)', '[¬*]': Shamela's own footnote numbering.
NEG_MARKER_RE = re.compile(rf"^(?:\(\s*¬\s*{_ID}\s*\)|\[\s*¬\s*{_ID}\s*\])")
# '(١)', '[١]', '(*)': numbering typed in the edition.
BRACKET_MARKER_RE = re.compile(rf"^(?:\(\s*{_ID}\s*\)|\[\s*{_ID}\s*\])")
# '١ ...', '١. ...', '١- ...', '١سورة ...': books that number without brackets.
BARE_MARKER_RE = re.compile(rf"^([{_D}]{{1,3}})(?![{_D}/])\s*[-–.)]?\s*(?=\S)")
MARKER_FAMILIES = (("neg", NEG_MARKER_RE), ("bracket", BRACKET_MARKER_RE), ("bare", BARE_MARKER_RE))
CONTINUED_TAIL_RE = re.compile(r"\s*=+\s*$")
TOKEN_RE = re.compile(r"\S+")

# Words taken on each side of a marker, and around the middle of a page.
PROBE_WORDS = 8
# A shorter probe matches too easily somewhere else.
MIN_PROBE_WORDS = 3
# Words quoted from before the marker, so a reader can tell which passage a
# footnote annotates: numbering restarts on every page, and `(¬N)` markers are
# removed from body text, so the number alone does not say.
ANCHOR_WORDS = 6
# How far back from the cursor a probe may be found (characters). Forward
# search is unbounded: a book's footnote pages can be far apart.
BACK_CHARS = 200_000


@dataclass
class Footnote:
    page_id: Any
    marker: tuple[str, str, str] | None  # (family, bracket, id), None when unmarked
    body: str
    anchor: str = ""
    parent: int | None = None
    method: str = ""

    @property
    def text(self) -> str:
        """`(١) [عند: «…»] body`: the marker as printed (minus `¬`), the anchor when located."""
        head = [marker_label(self.marker)] if self.marker else []
        if self.anchor:
            head.append(f"[عند: «{self.anchor}»]")
        return " ".join(head + [self.body])


def clean_like_body(text: str) -> str:
    """The cleaning `emit_section` applies to a body, so probes match parent text."""
    return clean_text(strip_source_markers(sanitize_markup(text)))


def _match_marker(rx: re.Pattern[str], line: str) -> tuple[tuple[str, str], int] | None:
    m = rx.match(line)
    if not m:
        return None
    groups = [g for g in m.groups() if g is not None]
    bracket = line[m.start()] if line[m.start()] in "([" else ""
    return (bracket, groups[0]), m.end()


def split_footnote_field(raw: str) -> tuple[str, list[tuple[tuple[str, str, str] | None, str]]]:
    """One page's footnote field -> (continuation of the previous page, footnotes).

    Each footnote is (marker, text), the text without its marker and cleaned
    like body text. Lines before the first marker are the continuation when the
    field opens with `=`, otherwise an unmarked footnote.
    """
    lines = [l.strip() for l in re.split(r"[\r\n]+", strip_html(raw or ""))]
    lines = [l for l in lines if l]
    if not lines:
        return "", []
    lines[-1] = CONTINUED_TAIL_RE.sub("", lines[-1])
    continued = lines[0].startswith("=")
    if continued:
        lines[0] = lines[0].lstrip("= ")

    # One marker family per page. `¬` is never typed in prose, so its presence
    # settles it; otherwise the first marker-like line decides, so a reference
    # such as '(١٢١٦)' opening a later line cannot switch a bare-numbered page.
    family = None
    if any(NEG_MARKER_RE.match(l) for l in lines):
        family = "neg"
    else:
        for line in lines[1 if continued else 0:]:
            family = next((name for name, rx in MARKER_FAMILIES[1:] if rx.match(line)), None)
            if family:
                break
    rx = dict(MARKER_FAMILIES).get(family) if family else None

    lead: list[str] = []
    notes: list[tuple[tuple[str, str, str], list[str]]] = []
    last_num = 0
    for idx, line in enumerate(lines):
        hit = None if rx is None or (continued and idx == 0) else _match_marker(rx, line)
        if hit:
            (bracket, ident), end = hit
            # Numbers rise within a page; a lower one is a reference in the text.
            if ident == "*" or safe_num(ident) > last_num:
                last_num = last_num if ident == "*" else safe_num(ident)
                notes.append(((family, bracket, ident), [line[end:]]))
                continue
        (notes[-1][1] if notes else lead).append(line)

    out: list[tuple[tuple[str, str, str] | None, str]] = []
    lead_text = clean_like_body(" ".join(lead))
    if lead_text and not continued:
        out.append((None, lead_text))
    for marker, body in notes:
        text = clean_like_body(" ".join(body))
        if text:
            out.append((marker, text))
    return (lead_text if continued else ""), out


def marker_label(marker: tuple[str, str, str]) -> str:
    _, bracket, ident = marker
    return {"(": f"({ident})", "[": f"[{ident}]"}.get(bracket, ident)


def _digit_class(ident: str) -> str:
    if ident == "*":
        return r"\*"
    out = []
    for ch in ident:
        n = safe_num(ch)
        out.append(f"[{n}{chr(0x0660 + n)}]")
    return "".join(out)


def body_marker_re(marker: tuple[str, str, str]) -> re.Pattern[str]:
    family, bracket, ident = marker
    core = _digit_class(ident)
    neg = r"¬\s*" if family == "neg" else ""
    if bracket == "(":
        return re.compile(rf"\(\s*{neg}{core}\s*\)")
    if bracket == "[":
        return re.compile(rf"\[\s*{neg}{core}\s*\]")
    return re.compile(rf"(?<![{_D}/]){core}(?![{_D}/])")


@lru_cache(maxsize=64)
def _token_spans(part: str) -> tuple[list[int], list[int]]:
    """Start and end offsets of the part's tokens; a page is probed once per marker
    and side, so this is computed once per part instead of each time (46% of the
    extraction time of a footnoted book before)."""
    spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(part)]
    return [t[0] for t in spans], [t[1] for t in spans]


def _window(part: str, start: int, end: int, before: int, after: int) -> str:
    starts, ends = _token_spans(part)
    i = bisect.bisect_right(ends, start)  # tokens wholly before `start`
    j = bisect.bisect_left(starts, end)   # first token wholly after `end`
    a = starts[i - before] if before and i >= before else (0 if before else start)
    b = ends[j + after - 1] if after and j + after - 1 < len(ends) else (len(part) if after else end)
    return clean_like_body(part[a:b])


def _usable(probes: list[str]) -> list[str]:
    """Probes long enough to trust, each also without its edge words.

    A probe is cleaned alone, the parent's text as a whole, and a cleaning rule can
    reach across a page join: a page ending `الآية.` before a page of dots loses its
    period to the leader rule. Dropping the edge words keeps the rest findable.
    """
    out = [p for p in probes if len(p.split()) >= MIN_PROBE_WORDS]
    out += [" ".join(p.split()[1:-1]) for p in out if len(p.split()) >= MIN_PROBE_WORDS + 2]
    return list(dict.fromkeys(out))


def marker_probes(part: str, start: int, end: int) -> list[str]:
    """The words around a marker, as they read in parent text: both sides, then each side."""
    return _usable([_window(part, start, end, PROBE_WORDS, PROBE_WORDS),
                    _window(part, start, end, PROBE_WORDS, 0),
                    _window(part, start, end, 0, PROBE_WORDS)])


def page_probes(parts: list[str]) -> list[str]:
    """Words from the middle, start and end of the page's longest part."""
    best = max(parts, key=lambda p: len(p.split()), default="")
    tokens = [(m.start(), m.end()) for m in TOKEN_RE.finditer(best)]
    if not tokens:
        return []
    n = 2 * PROBE_WORDS
    mid = max(0, len(tokens) // 2 - PROBE_WORDS)
    spans = [(mid, min(len(tokens), mid + n)), (0, min(len(tokens), n)), (max(0, len(tokens) - n), len(tokens))]
    return _usable([clean_like_body(best[tokens[a][0]:tokens[b - 1][1]]) for a, b in spans])


class ParentText:
    """A book's parents as one searchable string, in reading order."""

    def __init__(self, parents: list[dict[str, Any]]):
        chunks: list[str] = []
        self.starts: list[int] = []
        pos = 0
        for p in parents:
            path = p.get("section_path") or []
            chunk = f"{path[-1] if len(path) > 1 else ''}\n{p.get('text_original') or ''}"
            self.starts.append(pos)
            chunks.append(chunk)
            pos += len(chunk) + 1
        self.text = "\x00".join(chunks)

    def parent_at(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos) - 1

    def find(self, probes: list[str], cursor: int) -> int:
        """First match at or after `cursor`, else the nearest a short way before it."""
        for probe in probes:
            pos = self.text.find(probe, cursor)
            if pos >= 0:
                return pos
        for probe in probes:
            pos = self.text.rfind(probe, max(0, cursor - BACK_CHARS), cursor + len(probe))
            if pos >= 0:
                return pos
        return -1


def assign_footnotes(pages: list[dict[str, Any]], parents: list[dict[str, Any]]) -> list[Footnote]:
    """Read every page's footnotes and pick the parent (index into `parents`) for each."""
    from .shamela import page_parts  # here, not at the top: shamela imports this module

    if not parents:
        return []
    stream = ParentText(parents)
    cursor = 0
    located_any = False
    notes: list[Footnote] = []
    previous_page_notes: list[Footnote] = []
    for page in pages:
        continuation, fields = split_footnote_field(page.get("footnotes") or "")
        if continuation and previous_page_notes:
            previous_page_notes[-1].body += " " + continuation
        elif continuation:
            fields.insert(0, (None, continuation))
        if not fields:
            if not continuation:
                previous_page_notes = []
            continue
        parts = page_parts(page.get("body") or "")

        page_pos = stream.find(page_probes(parts), cursor)
        page_parent = stream.parent_at(page_pos) if page_pos >= 0 else None

        page_notes: list[Footnote] = []
        search_from = (0, 0)  # (part, offset) after the previous marker on this page
        for marker, body in fields:
            note = Footnote(page.get("page_id"), marker, body)
            if marker is not None:
                hit = _find_marker(parts, body_marker_re(marker), search_from)
                if hit:
                    k, start, end = hit
                    search_from = (k, end)
                    pos = stream.find(marker_probes(parts[k], start, end), cursor)
                    if pos >= 0:
                        note.parent, note.method = stream.parent_at(pos), "marker"
                        note.anchor = _window(parts[k], start, start, ANCHOR_WORDS, 0)
            if note.parent is None:
                if page_parent is not None:
                    note.parent, note.method = page_parent, "page"
                else:
                    note.parent = stream.parent_at(cursor) if located_any else 0
                    note.method = "neighbour"
            page_notes.append(note)
        if page_pos >= 0:
            cursor, located_any = page_pos, True
        notes.extend(page_notes)
        previous_page_notes = page_notes
    return notes


def _find_marker(parts: list[str], rx: re.Pattern[str], after: tuple[int, int]) -> tuple[int, int, int] | None:
    """(part, start, end) of the marker, searching from `after` first, then from the top."""
    k0, off0 = after
    for k in range(k0, len(parts)):
        m = rx.search(parts[k], off0 if k == k0 else 0)
        if m:
            return k, m.start(), m.end()
    for k in range(0, min(k0 + 1, len(parts))):
        m = rx.search(parts[k])
        if m:
            return k, m.start(), m.end()
    return None


def attach_shamela_footnotes(profile: dict[str, Any], state: BuildState, pages: list[dict[str, Any]],
                             first_parent: int, first_view: int) -> Counter:
    """Add the book's footnotes to the parents emitted since `first_parent`.

    Returns counts for the coverage report.
    """
    parents = state.parents[first_parent:]
    notes = assign_footnotes(pages, parents)
    stats: Counter = Counter()
    if not notes:
        return stats
    by_parent: dict[int, list[str]] = {}
    for note in notes:
        by_parent.setdefault(note.parent, []).append(note.text)
        stats[f"by_{note.method}"] += 1
    stats["footnotes"] = len(notes)

    last_order: dict[str, int] = {}
    for view in state.views[first_view:]:
        pid = view["parent_id"]
        last_order[pid] = max(last_order.get(pid, 0), view["order_in_parent"])

    for idx in sorted(by_parent):
        parent = parents[idx]
        texts = by_parent[idx]
        pid = parent["parent_id"]
        view_key = "view:" + pid[len("parent:"):]
        base = last_order.get(pid, 0)
        new_ids: list[str] = []
        for j, pack in enumerate(split_paragraph_packs("\n\n".join(texts), PARAGRAPH_PACK_WORDS), start=1):
            vid = f"{view_key}:fn:{j:03d}"
            state.views.append(make_view_fixed(
                profile, view_id=vid, parent_id=pid,
                view_type=FOOTNOTE_VIEW_TYPE, text_role=FOOTNOTE_TEXT_ROLE,
                order_in_parent=base + j, section_path=parent["section_path"],
                section_path_ids=parent["section_path_ids"], refs=parent["refs"],
                text=pack, quality_flags=parent["quality_flags"],
            ))
            new_ids.append(vid)
        body = parent["text_for_synthesis"]
        parent["text_for_synthesis"] = (f"{body}\n\n" if body else "") + FOOTNOTES_HEADER + "\n" + "\n".join(texts)
        parent["child_view_ids"] = parent["child_view_ids"] + new_ids
        stats["views"] += len(new_ids)
        stats["parents_updated"] += 1
    return stats
