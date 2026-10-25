"""Splitting transcribed lessons at speaker turns.
"""
from __future__ import annotations

from .detectors.speaker_turns import detect as speaker_turn_detect

# Separate and much lower: the size at which we *try* the speaker-turn split. A
# transcript is divided at speaker changes whatever its length, because that is the
# source's own structure and one exchange is a better retrieval unit than a lesson.
# This is not a cap — the split lands on a speaker change or does not happen.
DIALOGUE_TRIGGER_WORDS = 4000
# A dialogue parent should hold a whole exchange, not a single utterance. Turns
# are accumulated until the parent is worth retrieving, then closed at the next
# turn — so a parent ends where a speaker ends, never mid-sentence.
DIALOGUE_MIN_WORDS = 250
DIALOGUE_TARGET_WORDS = 1200
# Whose turn counts as the answering side. A new exchange begins when someone
# else speaks after these have spoken.
SHAYKH_SPEAKERS = frozenset({"الشيخ", "شيخنا"})
# A transcript talks. Requiring merely "three speaker labels" says nothing about a
# long text: 555,464 words of ابن القيم fiqh contain three, and on that evidence an
# entire classical book was reclassified as a recorded lesson and cut into 228
# `dialogue_exchange` parents that hold no dialogue at all.
#
# Density separates the two cleanly. Measured over every oversized parent in the
# corpus that contains any speaker label:
#
#     real transcripts (تفريغ الأشرطة، لقاءات، فتاوى جدة)   11 – 1,671 words/turn
#     classical prose with an incidental `الشيخ:`        2,083 – 185,156 words/turn
#
# Nothing real sits in the gap, so the gate goes in it. Below the threshold the
# labels are the text's own structure; above it they are three stray colons in a
# book, and the speaker split must not fire.
MAX_WORDS_PER_TURN = 2000


def split_speaker_turns(text: str, *, min_words: int = DIALOGUE_MIN_WORDS,
                        target_words: int = DIALOGUE_TARGET_WORDS,
                        hard_max: int = DIALOGUE_TRIGGER_WORDS) -> list[tuple[str, str]]:
    """Group a transcript into (speaker, text) exchanges at speaker boundaries.

    Returns [] when the text is not dialogue, so callers can fall back to their
    normal policy. Two gates decide that: at least three turns, and a speaker
    label at least every MAX_WORDS_PER_TURN words. The count alone is not enough
    — a stray `الشيخ:` inside prose would otherwise convert an ordinary book into
    fake dialogue, and did.

    Every boundary is a real speaker change found in the source, which is why
    this can cut a 40,000-word lesson without ever landing mid-sentence.
    """
    turns = speaker_turn_detect(text)
    if len(turns) < 3:
        return []
    if len(text.split()) / len(turns) > MAX_WORDS_PER_TURN:
        return []

    # Build (speaker, body) pairs from each label to the next.
    utterances: list[tuple[str, str]] = []
    if turns[0].start > 0:
        opening = text[: turns[0].start].strip()
        if opening:
            utterances.append(("", opening))
    for i, span in enumerate(turns):
        end = turns[i + 1].start if i + 1 < len(turns) else len(text)
        body = text[span.end:end].strip()
        if body:
            utterances.append((span.meta["speaker"], body))

    # A group may only close where one exchange ends and the next begins: at a
    # questioner's turn that follows the shaykh's reply. Closing on the word
    # budget alone splits a question from its answer — reviewed output showed
    # exactly that, exchanges ending on "the sheikh's final question is left
    # unanswered". The budget decides when we start *looking* for a boundary;
    # the dialogue decides where it actually falls.
    def opens_exchange(idx: int) -> bool:
        speaker = utterances[idx][0]
        if not speaker or speaker in SHAYKH_SPEAKERS:
            return False
        return any(utterances[j][0] in SHAYKH_SPEAKERS for j in range(idx))

    groups: list[tuple[str, str]] = []
    buf: list[str] = []
    lead = ""
    for i, (speaker, body) in enumerate(utterances):
        if buf:
            held = sum(len(p.split()) for p in buf)
            # Prefer an exchange boundary, but a long monologue may not offer one
            # for thousands of words. Past the hard cap, close at any speaker
            # change: still never mid-sentence, just a less natural seam.
            if (opens_exchange(i) and held >= target_words) or held >= hard_max:
                groups.append((lead, "\n".join(buf)))
                buf = []
        if not buf:
            lead = speaker
        buf.append(f"{speaker}: {body}" if speaker else body)
    if buf:
        tail = "\n".join(buf)
        # A short trailing remainder belongs with the exchange before it rather
        # than becoming a stub parent of its own.
        if groups and len(tail.split()) < min_words:
            groups[-1] = (groups[-1][0], groups[-1][1] + "\n" + tail)
        else:
            groups.append((lead, tail))
    return groups
