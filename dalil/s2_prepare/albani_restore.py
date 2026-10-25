"""Step 4: derived prepare inputs (the copies in raw/ stay byte-identical to the source).

Two documented changes, both only so that dalil.s2_prepare.ketabonline (unchanged)
keeps text and structure it would otherwise lose:

1. g-table rows -> <p>.  Ketabonline renders Shamela lines that contain " ... "
   as <div class="g-table"><div class="g-cell">A</div><div class="g-border"> ...
   </div><div class="g-cell">B</div></div>, outside any <p>.  prepare's
   extract_blocks() only emits <p> and title blocks, so this text is dropped
   (e.g. 9.8% of the live صحيح أبي داود - الأم, 97156).  Each such row that is
   not already inside a <p> becomes <p class="g-paragraph g-rtl">A ... B</p>.
2. 1423 only (structure_1423): the native index lists entries 1-4500 (vols 1-9,
   with a few gaps) but only volume and "القسم" nodes for vols 10-14, and
   prepare/extract file paragraphs by page. Entry openers found in sequence get
   a bare-number index node like the native ones, and each page is cut before
   an entry opener so the entry's node points to where its text starts.

    python -m dalil.s2_prepare.albani_restore 1423 6006

Reads <id>.data.zip from its book folder under books/original/ and writes the derived
copy to the same <author>/<book>/ under build/books/, with albani_restore_log.json
beside it.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from pathlib import Path

from dalil.paths import BUILD_BOOKS_DIR, DATA_DIR, ORIGINAL_DIR

AR_DIG = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
TAG = re.compile(r'<[^>]+>')
# The row's cells, then the table's own </div>. A lazy (.*?)</div>\s*</div> stopped at the last cell's
# closing tag, so that cell (the text after " ... ") was dropped: 44,812 characters of 1423.
GT = re.compile(r'<div class="g-table">((?:\s*<div class="g-(?:cell|border)">.*?</div>)*)\s*</div>', re.S)
CELL = re.compile(r'<div class="g-(cell|border)">(.*?)</div>', re.S)
P_TOKENS = re.compile(r'<p\b|</p>')
# an entry opener: a main-text paragraph starting with the entry number and a dash
# (optionally a leading quote or a footnote mark "(١)" before the dash)
OPENER = re.compile(r'<p class="g-paragraph g-rtl"[^>]*>\s*(?:<span[^>]*>\s*)?"?\s*([٠-٩0-9]{1,5})\s*(?:</span>)?\s*(?:\(\s*[٠-٩0-9]\s*\)\s*)?[-–—]')
MAX_STEP = 12   # the numbering has gaps (publisher notes, entries moved to al-Sahiha), never jumps backwards
SYNTH_ID_BASE = 3_000_000_000
SPLIT_ID_BASE = 900_000_000


def todig(s):
    return s.translate(AR_DIG)


def strip(h):
    return re.sub(r'[ \t ]+', ' ', html.unescape(TAG.sub(' ', h or ''))).strip()


def ranges(nums):
    """compress sorted ints into 'a-b' runs"""
    s = sorted(set(nums))
    out = []
    for n in s:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def p_depth_at(content, pos):
    d = 0
    for m in P_TOKENS.finditer(content, 0, pos):
        d = d + 1 if m.group(0) == '<p' else max(0, d - 1)
    return d


def fix_gtables(content, stats):
    out, last = [], 0
    for m in GT.finditer(content):
        if p_depth_at(content, m.start()) > 0:
            stats['g_table_rows_inside_p_left_as_is'] += 1
            continue
        cells = [c.group(2) for c in CELL.finditer(m.group(1))]
        inner = ' '.join(c.strip() for c in cells)
        out.append(content[last:m.start()])
        out.append(f'<p class="g-paragraph g-rtl" data-derived="g-table">{inner}</p>')
        last = m.end()
        stats['g_table_rows_converted'] += 1
        stats['g_table_chars_recovered'] += len(strip(inner))
    out.append(content[last:])
    return ''.join(out)


def structure_1423(d, stats):
    """Give every Silsila entry its own index node and its own page piece.

    * Entry openers are read in reading order and accepted only in sequence
      (last < N <= last + MAX_STEP), so numbers quoted inside a discussion are
      ignored. Native nodes (1-4500) are kept; an accepted number without a node
      gets a bare-number node like the native ones (title "N", title_level 2,
      under its volume, part_name of its page). The native index stops at 4500
      and also lacks a few earlier numbers.
    * prepare/extract give every paragraph of a page to the last index entry that
      starts on that page, so the tail of entry N-1 on the page where N starts was
      filed under N (402k words, 19.5%, in a first build). Each page is therefore
      cut before an accepted opener that has text before it, and the entry's node
      points to that piece. Pieces keep the print page number and part; the
      footnotes stay with the last piece of their page.
    """
    pages = d['pages']
    vol_by_part = {n['part_name']: n for n in d['index']}
    nodes = {}
    for v in d['index']:
        for n in v['children']:
            if n['title'].isdigit():
                nodes.setdefault(int(n['title']), n)
    native = set(nodes)
    accepted, last = [], 0       # (N, page index, char offset)
    for i, p in enumerate(pages):
        c = p['content']
        foot = c.find('g-page-footer')
        for m in OPENER.finditer(c):
            if foot != -1 and m.start() > foot:
                break
            n = int(todig(m.group(1)))
            if last < n <= last + MAX_STEP:
                accepted.append((n, i, m.start()))
                last = n
    stats['entry_openers_accepted'] = len(accepted)
    added = []
    for n, i, _ in accepted:
        if n in nodes:
            continue
        p = pages[i]
        part = p['part']['name']
        vol = vol_by_part[part]
        node = dict(parent=vol['id'], page_id=p['id'], children=[], title_level=2, id=SYNTH_ID_BASE + n,
                    page=p['page'], title=str(n), part_name=part, reciters=[], derived='albani_restore_20261016')
        vol['children'].append(node)
        nodes[n] = node
        added.append(n)
    stats['index_nodes_added'] = len(added)
    stats['index_nodes_added_numbers'] = ranges(added)
    # cut pages before accepted openers
    by_page = {}
    for n, i, at in accepted:
        by_page.setdefault(i, []).append((at, nodes[n]))
    pos = {p['id']: i for i, p in enumerate(pages)}
    new_pages, cut, repointed = [], 0, 0
    for i, p in enumerate(pages):
        c = p['content']
        cuts = [(at, node) for at, node in by_page.get(i, []) if 0 <= i - pos.get(node['page_id'], -99) <= 2]
        pieces, start, where = [], 0, []
        for at, node in cuts:
            if strip(c[start:at]):
                pieces.append((start, at))
                start = at
            where.append((node, len(pieces)))
        pieces.append((start, len(c)))
        ids = [p['id']] + [SPLIT_ID_BASE + p['id'] * 10 + j for j in range(1, len(pieces))]
        for j, (a, b) in enumerate(pieces):
            q = dict(p, id=ids[j], content=c[a:b])
            if j:
                q['derived'] = 'albani_restore_20261016:page_split'
            new_pages.append(q)
        cut += len(pieces) - 1
        for node, j in where:
            if node['page_id'] != ids[j]:
                node['page_id'] = ids[j]
                repointed += 1
    d['pages'] = new_pages
    # children in reading order (an existing heading before an entry on the same piece)
    npos = {p['id']: k for k, p in enumerate(new_pages)}
    for vol in d['index']:
        vol['children'].sort(key=lambda k: (npos.get(k['page_id'], 0), 1 if k['title'].isdigit() else 0, int(k['title']) if k['title'].isdigit() else 0))
    stats['pages_split_pieces_added'] = cut
    stats['index_nodes_pointed_to_entry_start'] = repointed
    stats['native_numbers'] = len(native)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", help="Ketabonline package ids")
    ap.add_argument("--original", type=Path, default=ORIGINAL_DIR, help="the downloads (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=BUILD_BOOKS_DIR, help="the repaired copies (default: %(default)s)")
    args = ap.parse_args(argv)
    original, out = args.original.absolute(), args.out.absolute()
    for pid in args.ids:
        src = next(original.rglob(f'{pid}.data.zip'))
        dst = out / src.relative_to(original)
        dst.parent.mkdir(parents=True, exist_ok=True)
        log_path = dst.parent / "albani_restore_log.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        with zipfile.ZipFile(src) as z:
            name = [n for n in z.namelist() if n.endswith('.json')][0]
            d = json.loads(z.read(name))
        stats = dict(g_table_rows_converted=0, g_table_chars_recovered=0, g_table_rows_inside_p_left_as_is=0)
        for pg in d['pages']:
            pg['content'] = fix_gtables(str(pg.get('content') or ''), stats)
        if pid == '1423':
            d = structure_1423(d, stats)
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(name, json.dumps(d, ensure_ascii=False))
        stats['source'] = str(src.relative_to(DATA_DIR))
        stats['derived'] = str(dst.relative_to(DATA_DIR))
        log[pid] = stats
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(pid, stats, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
