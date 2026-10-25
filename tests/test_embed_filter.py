"""Views with almost no Arabic are not embedded, except Quran verses: طه is a verse."""
from __future__ import annotations

from dalil.s6_embed.embed_views import too_little_arabic


def test_short_quran_verses_are_kept_and_markers_are_not():
    assert not too_little_arabic({"view_type": "ayah_exact", "display_text": "طه"}, 3)
    assert not too_little_arabic({"view_type": "ayah_exact", "display_text": "حمٓ"}, 3)
    assert too_little_arabic({"view_type": "heading_title", "display_text": "١٢ -"}, 3)
    assert not too_little_arabic({"view_type": "paragraph_pack", "display_text": "باب الصلاة"}, 3)
