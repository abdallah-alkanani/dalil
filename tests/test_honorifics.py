"""Honorific ligatures fonts cannot draw are spelled out in text shown to readers."""
from dalil.s4_extract.text import expand_undrawn_honorifics


def test_undrawn_ligature_becomes_its_phrase():
    assert expand_undrawn_honorifics("فليسأل الله ﷾ على الصفات") == "فليسأل الله سبحانه وتعالى على الصفات"
    assert expand_undrawn_honorifics("قال عمر ﵁:") == "قال عمر رضي الله عنه:"


def test_space_added_only_where_a_word_would_run_on():
    assert expand_undrawn_honorifics("اللهِ﷿وقال") == "اللهِ عز وجل وقال"
    assert expand_undrawn_honorifics("(﵀)") == "(رحمه الله)"


def test_ligatures_fonts_draw_are_kept():
    text = "قال رسول الله ﷺ، والله ﷻ، ﷽"
    assert expand_undrawn_honorifics(text) == text


def test_empty_and_none_pass_through():
    assert expand_undrawn_honorifics("") == ""
    assert expand_undrawn_honorifics(None) is None
