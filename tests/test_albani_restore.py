"""dalil.s2_prepare.albani_restore: g-table rows outside a paragraph become paragraphs."""
from __future__ import annotations

from collections import Counter

from dalil.s2_prepare.albani_restore import fix_gtables

# The shape Ketabonline renders a Shamela "A ... B" line in (al-Silsila al-Da'ifa).
ROW = ('<div class="g-table">   <div class="g-cell">"كل بناء وبال على صاحبه </div>'
       '   <div class="g-border"> ... </div>   <div class="g-cell">" .</div>   </div>')


def test_gtable_row_outside_paragraph_becomes_paragraph():
    stats = Counter()
    out = fix_gtables("<p>بلفظ:</p>" + ROW + "<p>فرفعته</p>", stats)
    assert '<div class="g-table">' not in out
    assert out.startswith('<p>بلفظ:</p><p class="g-paragraph g-rtl" data-derived="g-table">"كل بناء وبال على صاحبه')
    assert '"كل بناء وبال على صاحبه ... " .</p>' in out  # the last cell, after " ... ", is kept
    assert out.endswith("</p><p>فرفعته</p>")
    assert stats["g_table_rows_converted"] == 1


def test_gtable_row_inside_paragraph_is_left_alone():
    stats = Counter()
    content = "<p>نص " + ROW + "</p>"
    assert fix_gtables(content, stats) == content
    assert stats["g_table_rows_inside_p_left_as_is"] == 1
    assert stats["g_table_rows_converted"] == 0
