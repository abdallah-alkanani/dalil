"""Behavior tests for source-preserving Arabic text cleaning (dalil.arabic.clean)."""
from dalil.arabic.clean import (
    make_embedding_text,
    normalize_arabic_light,
    parse_openiti,
    remove_urls_and_html,
    score_arabic_quality,
    strip_openiti_markup_for_embedding,
)


def test_persian_kaf_yeh_and_tatweel_normalized_for_embedding():
    text = "كتاب كیفـية و مدرسۀ"
    normalized = normalize_arabic_light(text, dediac=False)
    assert normalized == "كتاب كيفية و مدرسة"


def test_bidi_controls_removed_from_embedding_text():
    text = "قال‫ ابن تيمية‬ رحمه الله"
    normalized = normalize_arabic_light(text, dediac=False)
    assert "‫" not in normalized and "‬" not in normalized, "bidi controls should be removed"
    assert "ابن تيمية" in normalized, "Arabic content should remain"


def test_hamza_ta_marbuta_and_sanad_formulae_are_preserved():
    text = "حدَّثنا أبو بكر قال: أخبرنا حمزة عن فاطمة"
    normalized = normalize_arabic_light(text, dediac=True)
    assert "أبو" in normalized, "hamza forms must not be deleted"
    assert "فاطمة" in normalized, "ta marbuta must not become ha"
    for token in ["حدثنا", "قال", "أخبرنا", "عن"]:
        assert token in normalized, f"sanad formula {token} must remain"


def test_openiti_metadata_and_page_markers_become_metadata_not_embedding_text():
    raw = """######OpenITI#
#META#Header#End#
#META#URI: 0123Author.Book
#META#Author: Ibn Taymiyya

### | كتاب الإيمان
PageV01P023
هذا نص مهم.
~~ تتمة السطر
"""
    parsed = parse_openiti(raw)
    cleaned, meta = strip_openiti_markup_for_embedding(raw)
    assert parsed["meta"].get("URI") == "0123Author.Book"
    assert meta["page_markers"] == ["PageV01P023"]
    assert "PageV01P023" not in cleaned, "page marker should not be embedded"
    assert "#META#" not in cleaned and "######OpenITI" not in cleaned, "OpenITI control metadata should not be embedded"
    assert "كتاب الإيمان" in cleaned, "heading text should remain as semantic text"


def test_urls_and_html_removed_and_counted():
    cleaned, counts = remove_urls_and_html("<p>نص</p> https://example.com?a=1&amp;b=2")
    assert cleaned.strip() == "نص"
    assert counts["html_tags_removed"] == 2
    assert counts["urls_removed"] == 1
    assert counts["html_entities_removed"] >= 1, "HTML entity should be counted"


def test_make_embedding_text_returns_quality_metrics_without_mutating_original():
    original = "قال أبو هريرة: <b>نصٌّ</b> PageV01P001"
    embedding, metrics = make_embedding_text(original, source_type="openiti")
    assert "<b>" not in embedding and "PageV01P001" not in embedding, "embedding removes markup/page marker"
    assert "أبو هريرة" in embedding, "content preserved"
    assert metrics["arabic_letter_ratio"] > 0.5, "quality metric should be computed"
    assert "text_original" not in metrics, "metrics should not duplicate original text"


def test_quality_counts_replacement_and_control_junk():
    metrics = score_arabic_quality("نص عربي �‫")
    assert metrics["replacement_chars"] == 1
    assert metrics["bidi_controls"] >= 1, "bidi controls counted"


def test_torn_definite_article_is_rejoined_but_word_endings_are_not():
    from dalil.s4_extract.records import clean_text

    assert clean_text("فما هو ا لجواب عن (( ا لإرواء ))") == "فما هو الجواب عن (( الإرواء ))"
    assert clean_text("سبحان ا لله") == "سبحان الله"
    for kept in ("وقدَّرناه مؤخَّرًا لفائدتين", "قَالُوٓا۟ لَن", "حروف ا ل م", "تعبيره ، ا لأنّهم", "ا للمسلمين"):
        assert clean_text(kept) == kept


def test_a_cut_measured_on_the_clean_body_lands_on_the_raw_body():
    # The boundary pass records word offsets in the parent's text_original (the
    # clean body); dalil.s4_extract cuts the raw body. A torn article before the cut
    # must not shift one against the other.
    from dalil.s4_extract.boundaries import split_at_anchors
    from dalil.s4_extract.records import clean_text, join_torn_articles

    raw = "باب\nفما هو ا لجواب   عن هذا\n\nفصل في الطهارة وأحكامها"
    clean = clean_text(raw).split()
    at = clean.index("فصل")
    pieces = split_at_anchors(join_torn_articles(raw), [{"word_index": at, "anchor": "فصل في الطهارة"}])
    assert pieces == ["باب\nفما هو الجواب   عن هذا", "فصل في الطهارة وأحكامها"]
