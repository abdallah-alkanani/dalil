"""dalil.s4_extract.sources.shamela: a chapter of numbered hadith becomes one entry per hadith."""
from __future__ import annotations

from dalil.s4_extract.sources.shamela import split_numbered_hadith

# The shape صحيح الجامع stores a chapter in: entries run together, a زيادة number
# follows some entry numbers, and takhrij references carry numbers of their own.
CHAPTER = ("حرف الألف. ١٢ - «آمركم بثلاث» .(صحيح) [حل] عن أبى هريرة. الصحيحة ٦٨٥."
           "١٣ - «آمروا النساء» .(صحيح) الإرواء ١٨٣٦."
           "١٤ - ٦ - «آمروا اليتيمة» .(صحيح) الصحيحة ٦٥٦."
           "١٦ - «آية المنافق ثلاث»")


def test_entries_split_at_consecutive_numbers_only():
    preface, entries = split_numbered_hadith(CHAPTER)
    assert preface == "حرف الألف. "
    assert [n for n, _ in entries] == [12, 13, 14, 16]  # 15 is missing from the source
    assert entries[0][1] == "١٢ - «آمركم بثلاث» .(صحيح) [حل] عن أبى هريرة. الصحيحة ٦٨٥."
    assert entries[2][1].startswith("١٤ - ٦ - «")  # the زيادة number stays in its entry
    assert "".join(text for _, text in entries) == CHAPTER[len(preface):]


def test_a_single_numbered_item_is_not_a_run():
    text = "نص بلا أرقام ١٢ - واحد فقط"
    assert split_numbered_hadith(text) == (text, [])
