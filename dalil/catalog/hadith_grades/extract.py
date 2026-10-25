"""Step 1: read the four graded copies and the four plain hadith_api Sunan from the DB (read-only).

Graded parent text: "<marker> <isnad+matn> تحقيق الألباني: <grade>, <refs>", where the marker is
"[N]" (Abu Dawud) or "(سنن X) N" (the other three) and N is the standard printed number.
52 Abu Dawud parents hold two entries. Writes WORK/extract.pkl.

    python -m dalil.catalog.hadith_grades.extract [--db DB] [--work WORK]
"""
import pickle, collections
from .common import *

GR = re.compile(r'تحقيق الألباني\s*:?\s*')

# Printed numbers in the graded copies that break the sequence, reviewed by hand (2026-10-16).
# key: (graded book, order_in_book, entry_in_parent) -> (standard number, evidence)
# Evidence in every case: the number fills the only hole between its physical neighbours, the
# plain hadith_api record at the same position carries the same text, and (Tirmidhi) the
# Ketabonline citation index agrees.
AD, TI, IM = COLL['sunan_abi_dawud'][0], COLL['jami_al_tirmidhi'][0], COLL['sunan_ibn_majah'][0]
NUMBER_FIXES = {
    (AD, 1586, 1): (1602, 'printed 1601 twice; 1602 missing; sits between 1601 and 1603'),
    (AD, 2111, 1): (2132, 'printed 2131 twice; 2132 missing; sits between 2131 and 2133'),
    (AD, 3185, 1): (3217, 'printed 3216 twice; 3217 missing; sits between 3216 and 3218'),
    (AD, 4274, 1): (4317, 'printed 4316 twice and 4317 for 4318; 4318 missing; two entries between 4316 and 4319'),
    (AD, 4275, 1): (4318, 'printed 4317; see 4274'),
    (TI, 343, 1): (343, 'printed 243 (typo); 343 missing; citation index 343'),
    (TI, 458, 1): (458, 'empty entry without number or text; citation index 458; 458 missing'),
    (TI, 1333, 1): (1333, 'printed 1332 twice; 1333 missing; citation index 1333'),
    (TI, 3729, 1): (3729, 'empty entry without number or text; citation index 3729; 3729 missing'),
    (IM, 392, 1): (320, 'printed 321 twice; 320 missing; sits between 319 and 321'),
    (IM, 906, 1): (834, 'printed 835 twice; 834 missing; sits between 833 and 835'),
    (IM, 920, 1): (848, 'printed 849 twice; 848 missing; sits between 847 and 849'),
}
# AD 418 is printed twice with the same text: once in place (order 414) and once misplaced
# after 316 (order 313). Both are kept; the misplaced one is flagged as a duplicate entry.
DUPLICATE_ENTRIES = {(AD, 313, 1): 'misplaced second copy of Abu Dawud 418 (same text as order 414)'}
# Abu Dawud 239 has no own entry: its marker (bare "٢٣٩") and text are inside the grade text of 238,
# and al-Albani's verdict for it is not in this copy.
EMBEDDED = re.compile(r'\s([٠-٩]{1,4})\s+((?:حدثنا|أخبرنا|حدثني|أخبرني)\s.*)$', re.S)

def marker_re(label):
    if label == 'سنن أبي داود':
        return re.compile(r'(?:^|(?<=\s))\[([٠-٩0-9]+)\]')
    return re.compile(r'\(' + label + r'\)\s*([٠-٩0-9]*)')

def graded_entries(coll, p, v):
    book, label = COLL[coll]
    mre = marker_re(label)
    views = collections.defaultdict(list)
    for fr, vid, pid, vt, dt in v.execute("select faiss_row, view_id, parent_id, view_type, display_text from child_metadata where book_id=? order by faiss_row", (book,)):
        views[pid].append(dict(faiss_row=fr, view_id=vid, view_type=vt, text=dt))
    out = []
    for pid, cit, t, o, refs in p.execute("select parent_id, citation, text_for_synthesis, order_in_book, refs_json from parents where book_id=? order by order_in_book", (book,)):
        cit_num = todig(cit.rsplit('،', 1)[-1]).strip(' []')
        ms = [m for m in mre.finditer(t)]
        # a marker counts only if a grade marker follows before the next marker (drops stray "[N]" inside text)
        segs = []
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
            segs.append((m, t[m.end():end]))
        if not segs:
            segs = [(None, t)]
        for k, (m, seg) in enumerate(segs):
            gm = GR.search(seg)
            body = seg[:gm.start()] if gm else seg
            graw = re.sub(r'\s+', ' ', seg[gm.end():]).strip() if gm else None
            num = int(todig(m.group(1))) if m and m.group(1) else None
            # the view that carries this entry's grade text
            gv = None
            if graw is not None:
                tail = norm(body)[-40:]
                cands = [x for x in views[pid] if 'تحقيق الألباني' in (x['text'] or '')]
                if len(cands) > 1:
                    cands = [x for x in cands if tail and tail[-25:] in norm(x['text'])] or cands
                gv = cands[0] if cands else None
            embedded = EMBEDDED.search(graw) if graw and label == 'سنن أبي داود' else None
            if embedded:
                graw = graw[:embedded.start()].strip()
            fix = NUMBER_FIXES.get((book, o, k + 1))
            row = dict(graded_book_id=book, graded_parent_id=pid, citation=cit, citation_number=cit_num, order_in_book=o,
                       entry_in_parent=k + 1, entries_in_parent=len(segs), printed_number=num,
                       standard_number=fix[0] if fix else num, number_note=fix[1] if fix else DUPLICATE_ENTRIES.get((book, o, k + 1)),
                       duplicate_entry=(book, o, k + 1) in DUPLICATE_ENTRIES,
                       body=body.strip(), grade_raw=graw,
                       graded_view_faiss_row=gv['faiss_row'] if gv else None, graded_view_id=gv['view_id'] if gv else None,
                       all_view_faiss_rows=[x['faiss_row'] for x in views[pid]])
            out.append(row)
            if embedded:
                n2 = int(todig(embedded.group(1)))
                out.append(dict(row, entry_in_parent=k + 1.5, printed_number=n2, standard_number=n2, duplicate_entry=False,
                                number_note='entry embedded in the grade text of %s (no marker, no verdict)' % num,
                                body=embedded.group(2).strip(), grade_raw=None))
    return out

def plain_entries(coll, p):
    out = []
    for pid, refs, t, sec, o in p.execute("select parent_id, refs_json, text_for_synthesis, section_path_json, order_in_book from parents where book_id=? order by order_in_book", ('hadith_api::' + coll,)):
        h = json.loads(refs)['hadith']; sp = json.loads(sec)
        out.append(dict(book_id='hadith_api::' + coll, parent_id=pid, hadith_api_number=int(h['hadith_number']), id_in_book=h.get('id_in_book'),
                        chapter_id=h.get('chapter_id'), chapter=sp[1] if len(sp) > 1 else None, order_in_book=o, text=t))
    return out

def main(argv=None):
    args = path_args(__doc__, db=True).parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    p, v = pconn(args.db), vconn(args.db)
    R = {}
    for coll in COLL:
        g = graded_entries(coll, p, v); pl = plain_entries(coll, p)
        R[coll] = dict(graded=g, plain=pl)
        nums = [e['standard_number'] for e in g if e['standard_number'] is not None]
        nn = collections.Counter(e["standard_number"] for e in g if not e["duplicate_entry"]); mx = max(n for n in nn if n)
        print("   still duplicated", [n for n, c in nn.items() if c > 1], "missing", sorted(set(range(1, mx + 1)) - set(nn)), "fixed", sum(1 for e in g if e["number_note"]), "embedded", [e["standard_number"] for e in g if e["entry_in_parent"] == 1.5])
        print(coll, "graded entries", len(g), 'parents', len({e['graded_parent_id'] for e in g}), 'with number', len(nums), 'distinct', len(set(nums)),
              'max', max(nums), 'no grade', sum(1 for e in g if e['grade_raw'] is None), 'no view', sum(1 for e in g if e['graded_view_faiss_row'] is None),
              '| plain', len(pl), 'max running id', max(x['hadith_api_number'] for x in pl))
    pickle.dump(R, open(args.work / 'extract.pkl', 'wb'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
