"""Step 5: merge the internal match (build.py) with the external check (external_check.py) and write
sunan_grades.jsonl and graded_entry_numbers.jsonl.

Downgrades a confident row to uncertain when
  number conflict  the external numbering puts this text clearly at another nearby number
                   (external text at N non-empty, share of the record there < 0.3, best number within +-3
                   holds >= 0.5 of it and >= 0.3 more): the graded copy and the sunnah.com-lineage numbering
                   order these hadith differently, a human has to pick
  grade conflict   a single verdict whose acceptability (sahih/hasan vs weak classes) differs from the
                   external al-Albani grade for the same number
Differences inside one side (صحيح vs حسن, ضعيف vs منكر) and compound verdicts only get a note.
Pipeline: extract -> align -> build -> external_check <dir> -> finalize (python -m dalil.catalog.hadith_grades <dir> runs all five).

    python -m dalil.catalog.hadith_grades.finalize [--work WORK] [--out OUT]
"""
import pickle, collections
from .common import *

ACCEPT = {'sahih': 'acceptable', 'hasan': 'acceptable', 'daif': 'weak', 'daif_jiddan': 'weak', 'munkar': 'weak', 'shadh': 'weak', 'mawdu': 'weak'}
OWNER_VERDICT = re.compile(r"^\s*(صحيح بما قبله|صحيح الإسناد|حسن صحيح|حسن الإسناد|صحيح|حسن|مرسل)\b")
OWNER_NOTE = ("accepted 2026-10-16 (owner review): the graded copy gives the verdict first and only references after it "
              "(e.g. «صحيح، الإرواء (٢٢٠٩)، الضعيفة (٤٦٠)»); the external dataset read «ضعيف» from a reference title. "
              "Where al-Albani adds a caveat (منكر/شاذ for one word or addition) the full grade text is shown with the hadith.")

def main(argv=None):
    args = path_args(__doc__).parse_args(argv)
    B = pickle.load(open(args.work / 'build.pkl', 'rb'))
    X = pickle.load(open(args.work / 'external_check.pkl', 'rb'))
    allg = pickle.load(open(args.work / 'graded_entries.pkl', 'rb'))
    rows_out = []; ext_by_std = {}
    for coll, b in B.items():
        ext = {o['parent_id']: o for o in X[coll]}
        for r in sorted(b['rows'], key=lambda r: r['hadith_api_number']):
            r = dict(r)
            o = ext.get(r['parent_id'])
            chk = dict(number=None, external_number=None, external_albani_grade=None, grade_class=None)
            if o:
                n = o['standard_number']
                empty = o['ext_albani'] is None and o['cp_here'] == 0.0 and o['cp_best'] == 0.0
                if o['cp_here'] >= 0.5 or o['best_nearby'] == n:
                    chk['number'] = 'agrees'
                elif o['cp_best'] >= 0.5 and o['cp_here'] < 0.3 and o['cp_best'] - o['cp_here'] >= 0.3:
                    chk['number'] = 'conflict'; chk['external_number'] = o['best_nearby']
                else:
                    chk['number'] = 'not_checkable'      # external text empty at N (merged into a neighbour) or segmented differently
                chk['external_albani_grade'] = o['ext_albani']
                if o['ext_class'] and o['our_class'] and r['albani_grade_kind'] == 'single':
                    if o['ext_class'] == o['our_class']: chk['grade_class'] = 'agrees'
                    elif ACCEPT.get(o['ext_class']) and ACCEPT.get(o['ext_class']) == ACCEPT.get(o['our_class']): chk['grade_class'] = 'same_side'
                    else: chk['grade_class'] = 'conflict'
                elif o['ext_class'] and r['albani_grade_kind'] in ('compound', 'per_narration'):
                    chk['grade_class'] = 'compound_not_compared'
                ext_by_std[(coll, n)] = dict(o, number_check=chk['number'])
            r['external_check'] = chk
            if r['status'] == 'confident':
                if chk['number'] == 'conflict':
                    r['status'] = 'uncertain'
                    r['uncertain_reasons'] = r['uncertain_reasons'] + ['numbering conflict: the graded copy prints %s for this text, the sunnah.com-lineage numbering (fawazahmed0 hadith-api, and the hadith_api order) has it at %s' % (r['standard_number'], chk['external_number'])]
                if chk['grade_class'] == 'conflict':
                    r['status'] = 'uncertain'
                    why = 'grade conflict: this copy says "%s", the external al-Albani grade for %s is "%s"' % (r['albani_grade'], r['standard_number'], chk['external_albani_grade'])
                    # most conflicts: a sahih/hasan verdict followed by a reference into a weak-hadith work
                    # ("صحيح الضعيفة تحت الحديث (N)", "صحيح، ضعيف أبي داود (N)"); the external grade looks derived from that reference
                    if chk.get('external_albani_grade', '').lower().startswith('daif') and re.search('الضعيفة|ضعيف أبي داود|ضعيف أبى داود|ضعيف الجامع|ضعيف سنن|ضعيف ابن ماجة', r['albani_grade_raw'] or ''):
                        chk['grade_conflict_pattern'] = 'acceptable verdict followed by a reference to a weak-hadith work'
                        why += ' (the raw text cites a weak-hadith work after the verdict; the external grade may come from that citation)'
                    r['uncertain_reasons'] = r['uncertain_reasons'] + [why]
            if chk['grade_class'] == 'same_side':
                r['notes'] = r['notes'] + ['external al-Albani grade "%s" differs in degree only' % chk['external_albani_grade']]
            # Owner decision 2026-10-16: when the only doubt is a grade conflict and the raw text opens
            # with the verdict itself, the graded copy is right; the external grade came from a reference.
            reasons = r['uncertain_reasons']
            if r['status'] == 'uncertain' and reasons and all(x.startswith('grade conflict') for x in reasons):
                m = OWNER_VERDICT.match(r['albani_grade_raw'] or '')
                if m and m.group(1) == r['albani_grade']:
                    r['status'] = 'confident'
                    r['notes'] = r['notes'] + [OWNER_NOTE]
            rows_out.append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    jl_write(args.out / 'sunan_grades.jsonl', rows_out)
    # graded entries: add the external number check of the plain record(s) they matched
    for g in allg:
        o = ext_by_std.get((g['collection'], g['standard_number']))
        g['external_number_check'] = o['number_check'] if o else None
    jl_write(args.out / 'graded_entry_numbers.jsonl', allg)
    for coll in B:
        rs = [r for r in rows_out if r['collection'] == coll]
        print(coll, dict(collections.Counter(r['status'] for r in rs)), 'number', dict(collections.Counter(r['external_check']['number'] for r in rs)),
              'grade', dict(collections.Counter(r['external_check']['grade_class'] for r in rs)))
    print('wrote', len(rows_out), 'plain rows,', len(allg), 'graded entries')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
