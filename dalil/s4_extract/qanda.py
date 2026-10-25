"""Splitting question-and-answer text into questions and answers.
"""
from __future__ import annotations

import re

from .records import clean_text

QANDA_QUESTION_RE = re.compile(r"^\s*(?:س|سؤال|السؤال)(?=[ـ:：\s]|$)\s*[ـ:：]*\s*([0-9٠-٩]{0,4})")
QANDA_ANSWER_RE = re.compile(r"^\s*(?:ج|جواب|الجواب)(?=[ـ:：\s]|$)\s*[ـ:：]*\s*([0-9٠-٩]{0,4})")
# Inline variants. Sources routinely put a whole Q&A on one physical line
# ("س ١: ما هو ...؟ جـ: الغزو الفكري ..."), and emit_section() collapses newlines
# before splitting, so line-anchored markers alone can never find the answer.
# A trailing colon is required, which keeps the single-letter markers precise.
QANDA_QUESTION_INLINE_RE = re.compile(
    r"(?:^|(?<=[\s.؟!،؛:\)\]]))(?:السؤال|سؤال|س)\s*[ـ]*\s*[0-9٠-٩]{0,4}\s*[:：]"
)
QANDA_ANSWER_INLINE_RE = re.compile(
    r"(?:^|(?<=[\s.؟!،؛:\)\]]))(?:الجواب|جواب|ج)\s*[ـ]*\s*[0-9٠-٩]{0,4}\s*[:：]"
)
# Above this, a Q&A 'question' is a mis-parse rather than a question.
MAX_QUESTION_CHARS = 4000


def split_qanda_pair(text: str) -> tuple[str, str]:
    """Split a Q&A body into (question, answer).

    A question marker (س/سؤال/السؤال) starts the question; an answer marker
    (ج/جواب/الجواب) starts the answer. Text before any marker is treated as
    part of the question. Returns ("", "") when no markers are found.

    Inline markers are tried first because the body has usually been whitespace
    collapsed by this point; the line-anchored form is kept as a fallback for
    sources that really do put the question and answer on separate lines.
    """
    text = text or ""
    answer_marks = [m.start() for m in QANDA_ANSWER_INLINE_RE.finditer(text)]
    if answer_marks:
        question_marks = [m.start() for m in QANDA_QUESTION_INLINE_RE.finditer(text)]
        question_start = question_marks[0] if question_marks else 0
        answer_start = next((p for p in answer_marks if p > question_start), None)
        if answer_start is not None:
            question = clean_text(text[question_start:answer_start])
            answer = clean_text(text[answer_start:])
            if question or answer:
                return question, answer
    lines = text.splitlines()
    question_lines: list[str] = []
    answer_lines: list[str] = []
    in_answer = False
    for line in lines:
        s = line.strip()
        if QANDA_ANSWER_RE.match(s):
            in_answer = True
            answer_lines.append(s)
        elif QANDA_QUESTION_RE.match(s):
            in_answer = False
            question_lines.append(s)
        elif in_answer:
            answer_lines.append(s)
        else:
            question_lines.append(s)
    question = clean_text("\n".join(question_lines))
    answer = clean_text("\n".join(answer_lines))
    if not answer and not any(QANDA_QUESTION_RE.match(l.strip()) for l in lines):
        # No answer and no explicit question marker: this is ordinary prose, not a
        # Q&A pair. Report nothing so the caller falls back to paragraph views
        # rather than labelling the whole body as a "question".
        return "", ""
    return question, answer


def split_qanda_units(text: str) -> list[tuple[str, str]]:
    """Split a body into Q&A pairs: each question marker starts a new pair;
    the following answer marker and its text belong to that pair.

    Inline question markers are used when present so that several Q&A pairs
    sharing one physical line are still separated into one unit each.
    """
    text = text or ""
    starts = [m.start() for m in QANDA_QUESTION_INLINE_RE.finditer(text)]
    if starts:
        units_inline: list[tuple[str, str]] = []
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
            chunk = text[start:end]
            if not clean_text(chunk):
                continue
            question, _ = split_qanda_pair(chunk)
            title = clean_text(question or chunk)[:150]
            units_inline.append((title, chunk))
        if units_inline:
            return units_inline

    units: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if QANDA_QUESTION_RE.match(s):
            if current_title is not None and current:
                units.append((current_title, current))
            current_title = s
            current = [s]
        elif current_title is not None:
            current.append(s)
    if current_title is not None and current:
        units.append((current_title, current))
    return [(t, "\n".join(c)) for t, c in units if clean_text("\n".join(c))]
