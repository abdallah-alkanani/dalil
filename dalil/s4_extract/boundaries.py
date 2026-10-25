"""Dividing oversized sections at boundaries found in the text.
"""
from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Any

from .detectors.inline_headings import detect as inline_heading_detect
from .detectors.section_markers import detect as section_marker_detect

# A parent is one semantic unit shown to the answering model whole, so its size is
# the context cost of every hit inside it. Measured against the real tokenizer
# (Qwen3-Embedding-8B) on this corpus, Arabic runs **2.04 tokens per word**, so:
#
#     15,000 words ≈  30,600 tokens      target
#     25,000 words ≈  51,100 tokens      too big
#    555,464 words ≈ 1,134,700 tokens    the parent this work exists to fix
#
# These are *soft*. A parent at 16,000 or 19,000 words is left alone — the point is
# to remove the units that are wildly wrong, not to make every parent the same size.
# Nothing is ever cut to hit a number; a parent is only divided where a boundary
# was actually found, and 18,000 words with no good boundary stays 18,000 words.
PARENT_TARGET_WORDS = 15000
# Only past here is a parent considered plainly wrong and sent for a boundary
# decision. 1.5x target: 20,000 passes, 25,000 does not.
PARENT_SPLIT_ABOVE = 22500
BOUNDARY_FLAG_PENDING = "needs_semantic_split"
BOUNDARY_FLAG_APPLIED = "semantic_boundary_split"


def thin_cuts(cuts: list[dict[str, Any]], total_words: int,
              target_words: int) -> list[dict[str, Any]]:
    """Keep the recorded boundaries that give pieces near `target_words`.

    The boundary pass records every real topic change it finds, which is denser
    than we want to cut — that density is deliberate, so the wanted parent size can
    change without asking the model again.

    Each step takes the boundary *closest* to the target, before or after it. The
    earlier rule took the first boundary at or past the target, which never cuts
    early and therefore overshoots to wherever the next boundary happens to be: on
    the 555,464-word parent that produced a 29,736-word piece against a 15,000-word
    target.

    Overshooting is not merely imprecise, it is unstable. Because each step starts
    where the last one landed, an overshoot moves every later target, so *adding*
    boundaries can make pieces bigger. Measured: feeding 19 newly-found, verified
    cuts into the old rule took the worst piece from 29,736 words to 34,405. The
    same file under this rule gives 21,006 — the largest stretch the model found no
    boundary in at all, which is the floor and stays whole by design.

    Choosing the nearest boundary keeps the same guarantee as before: every cut is
    one that was found in the text. It only changes *which* found cut is used.

    The last piece is merged back if the final boundary would leave a stub: a
    300-word tail is not a parent, it is the end of the one before it.
    """
    if not cuts or target_words <= 0:
        return []
    kept: list[dict[str, Any]] = []
    last = 0
    remaining = [c for c in cuts if c["word_index"] > 0]
    while remaining and total_words - last > target_words:
        goal = last + target_words
        best = min(remaining, key=lambda c: abs(c["word_index"] - goal))
        if best["word_index"] <= last:
            break
        kept.append(best)
        last = best["word_index"]
        remaining = [c for c in remaining if c["word_index"] > last]
    if kept and total_words - kept[-1]["word_index"] < target_words // 3:
        kept.pop()
    return kept


def load_semantic_boundaries(path: Path | None
                             ) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read a boundary decision file, grouped {book_id: {parent_id: [cut, …]}}.

    Grouped by book so each parser worker is handed only its own book's decisions
    rather than a copy of the whole corpus's.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not path:
        return out
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            cuts = sorted((c for c in (row.get("cuts") or []) if c.get("anchor")),
                          key=lambda c: c["word_index"])
            # Thinning is deliberately NOT done here any more. Headings found in the
            # text are a second source of cuts, and the two have to be thinned as
            # one set — thinning the recorded cuts first would pick a boundary at
            # 15,000 words and never see the heading 200 words away that is the
            # better cut. Selection happens at the split site, over everything.
            if cuts:
                out.setdefault(row["book_id"], {})[row["parent_id"]] = cuts
    return out


def inline_heading_cuts(text: str) -> list[dict[str, Any]]:
    """Structural headings sitting inside a line, expressed as cuts.

    `heading_detector` anchors on `^` and skips lines over 120 characters, so a
    source that ships a whole book as one line — زاد المعاد does — hides every
    `[فصل ...]` it contains. Those headings are the author's own division of the
    text, which makes them better evidence than any boundary we could ask a model
    for, and they cost nothing to find.

    The cut goes immediately *before* the heading so the heading opens the section
    it introduces rather than trailing the one before it. The anchor recorded is
    the corpus's own words at that offset, the same contract the recorded
    boundaries follow, so `split_at_anchors` can verify both the same way.
    """
    spans = inline_heading_detect(text)
    if not spans:
        return []
    words, starts = [], []
    for match in re.finditer(r"\S+", text):
        words.append(match.group(0))
        starts.append(match.start())
    out: list[dict[str, Any]] = []
    for span in spans:
        idx = bisect.bisect_left(starts, span.start)
        if not 0 < idx < len(words):
            continue
        out.append({"word_index": idx,
                    "anchor": " ".join(words[idx:idx + 6]),
                    "reason": "inline heading"})
    return out


def section_marker_cuts(text: str, category: str | None) -> list[dict[str, Any]]:
    """Section openers that are not bracketed headings, expressed as cuts.

    `inline_heading_cuts` only sees `[فصل ...]` and a bare `فصل` followed by `في` or a
    colon. A second family of books announces its divisions differently — `- فصل (…)`,
    `مسألة:`, and in tafsir `قوله تعالى: ﴿…﴾` — and those books are equally newline-free,
    so nothing anchored on `^` reaches them either. See `section_marker_detector` for
    why each keyword needs its continuation before it counts as a division.

    `category` is passed through because the tafsir-only forms are refused elsewhere:
    outside tafsir the same strings quote a verse as evidence mid-argument.
    """
    spans = section_marker_detect(text, category=category)
    if not spans:
        return []
    words, starts = [], []
    for match in re.finditer(r"\S+", text):
        words.append(match.group(0))
        starts.append(match.start())
    out: list[dict[str, Any]] = []
    for span in spans:
        idx = bisect.bisect_left(starts, span.start)
        if not 0 < idx < len(words):
            continue
        out.append({"word_index": idx,
                    "anchor": " ".join(words[idx:idx + 6]),
                    "reason": f"section marker ({span.meta.get('form')})"})
    return out


def merge_cuts(*groups: list[dict[str, Any]], min_gap: int = 300) -> list[dict[str, Any]]:
    """One ordered candidate list from several sources, dropping near-duplicates.

    A recorded boundary and a heading often name the same seam a few words apart.
    Keeping both would offer the thinner two cuts where the text has one.
    """
    merged = sorted((c for g in groups for c in g), key=lambda c: c["word_index"])
    out: list[dict[str, Any]] = []
    for cut in merged:
        if out and cut["word_index"] - out[-1]["word_index"] < min_gap:
            continue
        out.append(cut)
    return out


def split_at_anchors(text: str, cuts: list[dict[str, Any]]) -> list[str]:
    """Cut `text` at each recorded anchor, verifying the anchor is really there.

    Word offsets index `text.split()`, which is what the boundary pass measured —
    it reads the parent's `text_original`, which is this same text with whitespace
    collapsed (torn articles are joined in both), so the two agree word for word. The cut itself is made on the
    original string at the matching character offset, which keeps the paragraph
    breaks the view layer packs on.

    Returns [] when no cut verifies, so the caller can emit the section whole
    rather than cut it somewhere the model never chose.
    """
    words, starts = [], []
    for match in re.finditer(r"\S+", text):
        words.append(match.group(0))
        starts.append(match.start())

    verified: list[int] = []
    for cut in cuts:
        idx = cut.get("word_index")
        anchor = (cut.get("anchor") or "").split()
        if not isinstance(idx, int) or not anchor or not 0 < idx < len(words):
            continue
        if words[idx:idx + len(anchor)] != anchor:
            continue
        if verified and idx <= verified[-1]:
            continue
        verified.append(idx)
    if not verified:
        return []

    offsets = [0] + [starts[i] for i in verified] + [len(text)]
    return [p for p in (text[a:b].strip() for a, b in zip(offsets, offsets[1:])) if p]
