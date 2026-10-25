"""Step 3: turn the alignment into sunan_grades.jsonl and graded_entry_numbers.jsonl.

Per plain record (hadith_api parent), from its DP segment:
  link       graded entry of the segment sharing text with it; substantial when shared shingles >= 3
             and >= SUBST of the plain record (cov_plain) or of the graded entry (cov_graded) is shared.
  primary    the substantial link holding the record's opening words (ties: larger cov_plain + cov_graded).
  method     1:1                           one plain record <-> one graded entry
             part_of_graded_entry           one standard number spans several hadith_api records
             covers_several_graded_entries  the record holds more than one standard number
             out_of_sequence                text match off the monotone path (hadith_api order differs,
                                            or a 4th fragment of one entry), at most MAX_DIST positions
                                            from the expected slot
             sequence_slot_only             no text match; the record sits alone in the slot of a graded
                                            entry that is empty in the graded copy (number inferred, no grade)
             none                           nothing matched
Thresholds (see summary.json -> thresholds for the borderline cases behind them):
  1:1 confident      dice >= T_DICE; or dice >= T_DICE_SEQ with both neighbours anchored; or one text
                     contains the other (cov >= T_CONTAIN, same opening, neighbours anchored: truncated entry)
  part_of confident  cov_plain >= T_PART; or cov_plain >= T_PART_SEQ with both neighbours anchored
  covers_several     primary passes the 1:1 rule and all other entries share its grade class (or have no verdict)
  out_of_sequence    distance <= MAX_DIST, (dice >= T_OOS or cov_plain >= T_PART) and runner-up < 0.5
  any method         downgraded when another graded entry is about as similar (>= 0.5 and >= score - 0.1)
                     and the neighbours are not anchored, or when the grade text cannot be parsed
Writes WORK/build.pkl and WORK/graded_entries.pkl; finalize writes the two jsonl files.

    python -m dalil.catalog.hadith_grades.build [--work WORK]
"""
import pickle, collections
from .common import *

SUBST, T_DICE, T_DICE_SEQ, T_CONTAIN, T_PART, T_PART_SEQ, T_OOS, MAX_DIST, T_ANCHOR, COMP_WINDOW = 0.5, 0.7, 0.5, 0.85, 0.8, 0.6, 0.8, 3, 0.8, 10

def opening_hits(text, y_sh):
    return len(shingles(words(text)[:8]) & y_sh)

def link(P, G, x, y):
    S = len(P[x]['sh'] & G[y]['sh'])
    cp = S / max(1, len(P[x]['sh'])); cg = S / max(1, len(G[y]['sh']))
    return dict(y=y, S=S, cp=cp, cg=cg, dice=2 * S / max(1, len(P[x]['sh']) + len(G[y]['sh'])), subst=S >= 3 and max(cp, cg) >= SUBST)

def assign(coll, a):
    """per plain index: method, primary link, other substantial links, extra evidence"""
    G, P, segs, oos = a['G'], a['P'], a['segs'], a['oos']
    asg = {}; far = {}
    holders = collections.defaultdict(set)
    for (i, k, j, m) in segs:
        for x in range(i, i + k):
            subs = [l for l in (link(P, G, x, y) for y in range(j, j + m)) if l['subst']]
            if not subs:
                continue
            prim = max(subs, key=lambda l: (opening_hits(P[x]['text'], G[l['y']]['sh']), l['cp'] + l['cg']))
            asg[x] = dict(prim=prim, others=[l for l in subs if l is not prim], oos=None)
            for l in subs: holders[l['y']].add(x)
    for (x, y, st, dist, runner, used) in oos:
        if dist > MAX_DIST:
            far[x] = (G[y]['standard_number'], round(st[2], 3)); continue
        l = link(P, G, x, y)
        asg[x] = dict(prim=l, others=[], oos=dict(dist=dist, runner=runner, graded_used=used))
        holders[y].add(x)
    for x, s in asg.items():
        p = s['prim']
        if s['oos']: s['method'], s['score'] = 'out_of_sequence', max(p['dice'], p['cp'])
        elif s['others']: s['method'], s['score'] = 'covers_several_graded_entries', p['dice']
        elif len(holders[p['y']]) > 1: s['method'], s['score'] = 'part_of_graded_entry', p['cp']
        else: s['method'], s['score'] = '1:1', p['dice']
    a['far'] = far
    return asg, holders

def finish(coll, a, asg, holders):
    G, P, cand = a['G'], a['P'], a['cand']
    cand_by_plain = collections.defaultdict(list)
    for (i, j), st in cand.items(): cand_by_plain[i].append((j, st))
    strong = {x for x, s in asg.items() if not s['oos'] and ((s['method'] == '1:1' and s['score'] >= T_ANCHOR) or (s['method'] == 'part_of_graded_entry' and s['score'] >= 0.9))}
    graded_used = set(holders)
    rows = []
    for x, pl in enumerate(P):
        row = dict(collection=coll, book_id=pl['book_id'], parent_id=pl['parent_id'], hadith_api_number=pl['hadith_api_number'])
        s = asg.get(x)
        anch = (x - 1 in strong) and (x + 1 in strong)
        if s is None:
            row.update(_unmatched(x, P, G, asg, graded_used, anch))
            if x in a.get('far', {}):
                n, d = a['far'][x]
                row['notes'].append('its text is also standard %s (dice %.2f), too far from this position to be its counterpart' % (n, d))
            rows.append(row); continue
        p, others, method, score = s['prim'], s['others'], s['method'], s['score']
        g = G[p['y']]
        lab, kind, cls = grade_label(g['grade_raw'])
        mine = {p['y']} | {l['y'] for l in others}
        # competitor: another graded entry about as similar AND close enough to be a rival on the monotone path
        # (repeats far away in the book, e.g. AD 4406 = AD 2957, cannot be the counterpart)
        comp = max([st[2] for j, st in cand_by_plain.get(x, []) if j not in mine and abs(j - p['y']) <= COMP_WINDOW] + [0.0])
        comp_far = max([st[2] for j, st in cand_by_plain.get(x, []) if j not in mine and abs(j - p['y']) > COMP_WINDOW] + [0.0])
        same_open = opening_hits(pl['text'], g['sh']) >= 2
        reasons, notes = [], []
        if method == '1:1':
            ok = score >= T_DICE or (score >= T_DICE_SEQ and anch) or (max(p['cp'], p['cg']) >= T_CONTAIN and same_open and anch)
            if not ok: reasons.append('text similarity %.2f below threshold' % score)
            elif score < T_DICE_SEQ: notes.append('one text contains the other (cov_plain %.2f, cov_graded %.2f): truncated entry' % (p['cp'], p['cg']))
        elif method == 'part_of_graded_entry':
            ok = score >= T_PART or (score >= T_PART_SEQ and anch)
            if not ok: reasons.append('only %.0f%% of the record is inside the graded entry' % (100 * score))
            notes.append('standard number %s spans %d hadith_api records' % (g['standard_number'], len(holders[p['y']])))
        elif method == 'covers_several_graded_entries':
            ok = p['dice'] >= T_DICE or (p['dice'] >= T_DICE_SEQ and anch) or (max(p['cp'], p['cg']) >= T_CONTAIN and same_open)
            if not ok: reasons.append('primary entry overlap weak (dice %.2f)' % p['dice'])
            classes = {cls} | {grade_label(G[l['y']]['grade_raw'])[2] for l in others}
            classes.discard(None)
            if len(classes) > 1:
                ok = False; reasons.append('record spans standard numbers with different grades')
        else:
            o = s['oos']
            ok = o['dist'] <= MAX_DIST and (p['dice'] >= T_OOS or p['cp'] >= T_PART) and o['runner'] < 0.5
            if o['dist'] > MAX_DIST: reasons.append('text match %d positions from the expected slot' % o['dist'])
            elif not ok: reasons.append('out-of-sequence text match too weak or not unique (dice %.2f, cov_plain %.2f, runner-up %.2f)' % (p['dice'], p['cp'], o['runner']))
            notes.append('hadith_api order differs here; matched by text %d position(s) from the expected slot' % round(o['dist']))
            if o['graded_used']: notes.append('the graded entry is shared with other hadith_api records')
        if comp >= 0.5 and comp >= score - 0.1 and not anch and method != 'out_of_sequence':
            ok = False; reasons.append('another graded entry is about as similar (%.2f) and the neighbours are not anchored' % comp)
        if kind == 'unparsed':
            ok = False; reasons.append('grade text is not a verdict (editor note or reference only); read albani_grade_raw')
        if comp_far >= 0.8:
            notes.append('the same text recurs elsewhere in the book (dice %.2f), outside the alignment window' % comp_far)
        if g['number_note']:
            notes.append(('graded copy: ' if not g['duplicate_entry'] else '') + g['number_note'])
        if kind in ('no_verdict', 'not_found'):
            notes.append('al-Albani gives no verdict for this entry in the graded copy')
        nums = sorted({g['standard_number']} | {G[l['y']]['standard_number'] for l in others})
        row.update(standard_number=g['standard_number'], standard_numbers=nums if len(nums) > 1 else None,
                   albani_grade=lab if kind in ('single', 'compound', 'per_narration') else None,
                   albani_grade_kind=kind, albani_grade_class=cls, albani_grade_raw=g['grade_raw'],
                   graded_book_id=g['graded_book_id'], graded_parent_id=g['graded_parent_id'],
                   graded_entry_in_parent=g['entry_in_parent'] if g['entries_in_parent'] > 1 or g['entry_in_parent'] != 1 else None,
                   graded_view_faiss_row=g['graded_view_faiss_row'], graded_view_id=g['graded_view_id'],
                   match_method=method, match_score=round(score, 3), cov_plain=round(p['cp'], 3), cov_graded=round(p['cg'], 3),
                   competitor_score=round(comp, 3) if comp else None, neighbours_anchored=anch,
                   status='confident' if ok else 'uncertain', uncertain_reasons=reasons, notes=notes)
        if others:
            row['other_graded_entries'] = [dict(standard_number=G[l['y']]['standard_number'], graded_parent_id=G[l['y']]['graded_parent_id'],
                                                albani_grade=grade_label(G[l['y']]['grade_raw'])[0], albani_grade_raw=G[l['y']]['grade_raw'],
                                                cov_plain=round(l['cp'], 3), cov_graded=round(l['cg'], 3)) for l in others]
        rows.append(row)
    return rows

def _unmatched(x, P, G, asg, graded_used, anch):
    """no text match. If the record is alone between two matched neighbours whose standard numbers leave
    exactly one free number, and that graded entry is empty, the number is inferred from the slot."""
    out = dict(standard_number=None, standard_numbers=None, albani_grade=None, albani_grade_kind=None, albani_grade_class=None,
               albani_grade_raw=None, graded_book_id=None, graded_parent_id=None, graded_entry_in_parent=None,
               graded_view_faiss_row=None, graded_view_id=None, match_method='none', match_score=None, cov_plain=None, cov_graded=None,
               competitor_score=None, neighbours_anchored=anch, status='unmatched', uncertain_reasons=[], notes=[])
    a, b = asg.get(x - 1), asg.get(x + 1)
    if a and b:
        na, nb = G[a['prim']['y']]['standard_number'], G[b['prim']['y']]['standard_number']
        free = [j for j, e in enumerate(G) if na < e['standard_number'] < nb and j not in graded_used]
        if nb - na == 2 and len(free) == 1 and not G[free[0]]['body'].strip():
            e = G[free[0]]
            out.update(standard_number_inferred=e['standard_number'], match_method='sequence_slot_only', graded_book_id=e['graded_book_id'],
                       graded_parent_id=e['graded_parent_id'], graded_view_faiss_row=e['graded_view_faiss_row'], graded_view_id=e['graded_view_id'])
            out['notes'].append('sits alone in the slot of %s, whose entry in the graded copy is empty (no text, no verdict); number inferred from position only' % e['standard_number'])
            return out
        out['notes'].append('no counterpart in the graded copy; neighbours are %s and %s, so no standard number is free for it' % (na, nb))
    return out

def graded_entry_rows(coll, G, P, rows, holders):
    by_parent = {r['parent_id']: r for r in rows}
    out = []
    for y, e in sorted(enumerate(G), key=lambda t: (t[1]['order_in_book'], t[1]['entry_in_parent'])):
        lab, kind, cls = grade_label(e['grade_raw'])
        plain = [P[x] for x in holders.get(y, [])]
        src = 'printed'
        if e['entry_in_parent'] != int(e['entry_in_parent']): src = 'embedded_in_previous_entry'
        elif e['number_note'] and not e['duplicate_entry']: src = 'corrected'
        out.append(dict(collection=coll, graded_book_id=e['graded_book_id'], graded_parent_id=e['graded_parent_id'],
                        entry_in_parent=e['entry_in_parent'], entries_in_parent=e['entries_in_parent'], order_in_book=e['order_in_book'],
                        citation=e['citation'], citation_number=int(e['citation_number']) if e['citation_number'].isdigit() else e['citation_number'],
                        printed_number=e['printed_number'], standard_number=e['standard_number'], number_source=src, number_note=e['number_note'],
                        duplicate_entry=e['duplicate_entry'],
                        citation_number_wrong=str(e['standard_number']) != e['citation_number'],
                        albani_grade=lab if kind in ('single', 'compound', 'per_narration') else None, albani_grade_kind=kind,
                        albani_grade_class=cls, albani_grade_raw=e['grade_raw'],
                        graded_view_faiss_row=e['graded_view_faiss_row'], graded_view_id=e['graded_view_id'], all_view_faiss_rows=e['all_view_faiss_rows'],
                        plain_parent_ids=[x['parent_id'] for x in plain], plain_hadith_api_numbers=[x['hadith_api_number'] for x in plain],
                        plain_statuses=[by_parent[x['parent_id']]['status'] for x in plain],
                        text_start=e['body'][:80]))
    return out

def main(argv=None):
    args = path_args(__doc__).parse_args(argv)
    A = pickle.load(open(args.work / 'align.pkl', 'rb'))
    B = {}; allg = []
    for coll, a in A.items():
        asg, holders = assign(coll, a)
        rows = finish(coll, a, asg, holders)
        hold = {y: sorted(v) for y, v in holders.items()}
        B[coll] = dict(rows=rows, G=a['G'], P=a['P'], holders=hold)
        st = collections.Counter(r['status'] for r in rows); me = collections.Counter((r['match_method'], r['status']) for r in rows)
        print(coll, dict(st), dict(sorted(me.items())))
        allg += graded_entry_rows(coll, a['G'], a['P'], rows, hold)
    pickle.dump(allg, open(args.work / 'graded_entries.pkl', 'wb'))
    pickle.dump(B, open(args.work / 'build.pkl', 'wb'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
