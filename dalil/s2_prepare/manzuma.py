"""Pair the hemistich columns of المنظومة السنية (.doc exported to HTML by LibreOffice).

Each page has two positioned text boxes: first hemistichs (left ~1.8in) and second
hemistichs (left ~5.3in), one per line. Prints one verse per line, "first ... second".

    soffice --headless --convert-to html 054.doc && python -m dalil.s2_prepare.manzuma 054.html > 054.txt
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html", type=Path, help="the .doc exported to HTML by LibreOffice")
    args = ap.parse_args(argv)
    t = args.html.read_text(encoding="utf-8", errors="replace")
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'<span id="Frame\d+"[^>]*style="([^"]*)"', t)]
    boxes = []
    for i, (pos, style) in enumerate(starts):
        chunk = t[pos:starts[i + 1][0] if i + 1 < len(starts) else len(t)]
        lines = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p))).strip() for p in re.split(r"<br\s*/?>", chunk)]
        lines = [l for l in lines if re.search("[ء-ي]", l)]
        if lines:
            boxes.append((float(re.search(r"left:\s*([-\d.]+)in", style).group(1)), lines))
    verses = []
    for a, b in zip(boxes[0::2], boxes[1::2]):
        first, second = sorted([a, b])  # smaller left offset = first hemistich
        assert len(first[1]) == len(second[1]), (len(first[1]), len(second[1]))
        verses += [f"{x} ... {y}" for x, y in zip(first[1], second[1])]
    print("\n".join(verses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
