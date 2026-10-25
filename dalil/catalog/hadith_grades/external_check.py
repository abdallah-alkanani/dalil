"""Step 4: independent check of every mapped row against fawazahmed0/hadith-api (Arabic editions).

That dataset (https://github.com/fawazahmed0/hadith-api, CDN
https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-<book>.min.json) carries the
standard printed number (hadithnumber) and al-Albani's grade (English label) for each hadith.
It is NOT used to build the mapping; it only measures it:
  number check  the plain record's text vs the external text at the same standard number
                (share of the plain record found there, and the best number within +-3)
  grade check   our albani_grade_class vs the external al-Albani grade class
Writes WORK/external_check.pkl.

    python -m dalil.catalog.hadith_grades.external_check <folder with ara-*.min.json> [--work WORK]
"""
import os, json, pickle, collections
from .common import *

FILES = {'sunan_abi_dawud': 'ara-abudawud', 'jami_al_tirmidhi': 'ara-tirmidhi', 'sunan_al_nasai': 'ara-nasai', 'sunan_ibn_majah': 'ara-ibnmajah'}
EN_CLASS = [('very daif', 'daif_jiddan'), ('mawdu', 'mawdu'), ('munkar', 'munkar'), ('shadh', 'shadh'), ('daif', 'daif'),
            ('hasan sahih', 'sahih'), ('sahih', 'sahih'), ('hasan', 'hasan')]
def en_class(g):
    g = (g or '').lower().replace("'", '').replace('ʿ', '')
    if g.startswith('sanad daif'): return 'daif'
    return next((c for k, c in EN_CLASS if g.startswith(k)), None)   # labels such as "Maqtu" are not compared

def load(folder, coll):
    d = json.load(open(os.path.join(folder, FILES[coll] + '.min.json'), encoding='utf-8'))
    by = collections.defaultdict(list)
    for h in d['hadiths']:
        n = h['hadithnumber']
        if not float(n).is_integer(): n = int(float(n))      # sub-numbers (e.g. 925.1) fold into their number
        alb = [g['grade'] for g in h.get('grades', []) if 'albani' in g['name'].lower()]
        by[int(n)].append(dict(text=h['text'], albani=alb[0] if alb else None))
    out = {}
    for n, hs in by.items():
        out[n] = dict(sh=shingles(words(' '.join(h['text'] for h in hs))), albani=next((h['albani'] for h in hs if h['albani']), None))
    return out

def main(argv=None):
    ap = path_args(__doc__)
    ap.add_argument('folder', help='fawazahmed0/hadith-api ara-{abudawud,tirmidhi,nasai,ibnmajah}.min.json')
    args = ap.parse_args(argv)
    folder = args.folder
    B = pickle.load(open(args.work / 'build.pkl', 'rb'))
    res = {}
    for coll, b in B.items():
        ext = load(folder, coll)
        Pmap = {x['parent_id']: x for x in b['P']}
        out = []
        for r in b['rows']:
            n = r['standard_number'] if r['standard_number'] is not None else r.get('standard_number_inferred')
            if n is None: continue
            sh = Pmap[r['parent_id']]['sh']
            def cp(k):
                e = ext.get(k)
                return len(sh & e['sh']) / max(1, len(sh)) if e else 0.0
            here = cp(n)
            best = max(range(n - 3, n + 4), key=cp)
            e = ext.get(n)
            out.append(dict(parent_id=r['parent_id'], hadith_api_number=r['hadith_api_number'], standard_number=n, status=r['status'],
                            method=r['match_method'], cp_here=round(here, 3), best_nearby=best, cp_best=round(cp(best), 3),
                            ext_albani=e['albani'] if e else None, ext_class=en_class(e['albani']) if e else None,
                            our_class=r['albani_grade_class'], our_kind=r['albani_grade_kind'], our_grade=r['albani_grade']))
        res[coll] = out
        conf = [o for o in out if o['status'] == 'confident']
        agree = [o for o in conf if o['cp_here'] >= 0.5 or o['best_nearby'] == o['standard_number']]
        shifted = [o for o in conf if o['best_nearby'] != o['standard_number'] and o['cp_best'] >= 0.5 and o['cp_best'] > o['cp_here'] + 0.2]
        gc = [o for o in conf if o['our_class'] and o['ext_class'] and o['our_class'] != 'mixed']
        print(coll, 'confident', len(conf), 'number text-agrees', len(agree), 'shifted', len(shifted),
              'grade comparable', len(gc), 'grade class agrees', sum(1 for o in gc if o['our_class'] == o['ext_class']))
    pickle.dump(res, open(args.work / 'external_check.pkl', 'wb'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
