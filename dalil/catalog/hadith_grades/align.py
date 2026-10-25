"""Step 2: align each plain hadith_api Sunan with al-Albani's graded copy.

Method
1. Both books are in printed order once the graded entries are sorted by their standard number
   (printed number, with the hand-reviewed fixes from extract.py) and Ibn Majah's Muqaddima
   (hadith_api running ids 4080-4345) is moved to the front.
2. Text: word 3-shingles of a normalised text (diacritics, honorifics, archaic spellings folded).
3. Backbone: max-weight chain strictly increasing in both books over candidate pairs found by
   rare-shingle votes (1:1 pairs whose texts substantially agree).
4. Segment DP inside a band of +-BAND graded positions around the backbone: monotone segments
   k plain : m graded (k, m <= 3), scored by shared shingles minus ALPHA x non-shared shingles
   minus BETA per extra record, or skip. This captures one standard number split into several
   hadith_api records (Tirmidhi's extra chains) and records merged on either side.
5. Leftovers: an unmatched plain record is compared with the unmatched graded entries within
   +-15 positions (out-of-sequence, text only).
Writes WORK/align.pkl.

    python -m dalil.catalog.hadith_grades.align [--work WORK]
"""
import pickle, collections, bisect
from .common import *

BAND, ALPHA, BETA = 8, 0.1, 1.0
SEGS = [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (2, 2), (3, 2), (2, 3)]

def sort_graded(g):
    return sorted(g, key=lambda e: (e['standard_number'], e['duplicate_entry'], e['order_in_book'], e['entry_in_parent']))

def sort_plain(coll, pl):
    pl = sorted(pl, key=lambda x: x['hadith_api_number'])
    if coll == 'sunan_ibn_majah':   # Muqaddima placed at the end by hadith_api
        pl = [x for x in pl if x['chapter_id'] == 0] + [x for x in pl if x['chapter_id'] != 0]
    return pl

def pair_stats(A, B):
    inter = len(A & B)
    return (inter / len(A) if A else 0.0), (inter / len(B) if B else 0.0), (2 * inter / (len(A) + len(B)) if A and B else 0.0)

def chain(pairs, nG):
    """max-weight chain strictly increasing in plain index i and graded index j (Fenwick prefix max)."""
    tree = [(0.0, -1)] * (nG + 1)
    def upd(j, val):
        j += 1
        while j <= nG:
            if val[0] > tree[j][0]: tree[j] = val
            j += j & -j
    def query(j):
        best = (0.0, -1)
        while j > 0:
            if tree[j][0] > best[0]: best = tree[j]
            j -= j & -j
        return best
    byi = collections.defaultdict(list)
    for k, (i, j, w) in enumerate(pairs): byi[i].append(k)
    score = [0.0] * len(pairs); back = [-1] * len(pairs)
    for i in sorted(byi):
        for k in byi[i]:
            b = query(pairs[k][1]); score[k] = b[0] + pairs[k][2]; back[k] = b[1]
        for k in byi[i]: upd(pairs[k][1], (score[k], k))
    k = max(range(len(pairs)), key=lambda k: score[k]); out = []
    while k != -1: out.append(pairs[k]); k = back[k]
    return out[::-1]

def band_centres(bb, nP, nG):
    """expected graded index for every plain index, interpolated between backbone pairs"""
    xs = [i for i, j, w in bb]; ys = [j for i, j, w in bb]
    c = []
    for i in range(nP + 1):
        k = bisect.bisect_left(xs, i)
        if k < len(xs) and xs[k] == i: c.append(ys[k]); continue
        lo = (xs[k - 1], ys[k - 1]) if k > 0 else (-1, -1)
        hi = (xs[k], ys[k]) if k < len(xs) else (nP, nG)
        c.append(lo[1] + (i - lo[0]) * (hi[1] - lo[1]) / max(1, hi[0] - lo[0]))
    return c

def segment_dp(P, G, bb):
    nP, nG = len(P), len(G)
    cen = band_centres(bb, nP, nG)
    best = [dict() for _ in range(nP + 1)]
    best[0][0] = (0.0, None, None)
    for j in range(1, min(nG, BAND) + 1): best[0][j] = (0.0, (0, j - 1), ('skipG', 0, 1))
    def relax(i2, j2, val, frm, op):
        if i2 > nP or j2 > nG or abs(j2 - cen[i2]) > BAND + 1: return
        cur = best[i2].get(j2)
        if cur is None or val > cur[0] + 1e-9: best[i2][j2] = (val, frm, op)
    for i in range(nP + 1):
        for j in range(max(0, int(cen[i]) - BAND - 2), min(nG, int(cen[i]) + BAND + 2) + 1):
            if j not in best[i]: continue
            v = best[i][j][0]
            relax(i, j + 1, v, (i, j), ('skipG', 0, 1))
            relax(i + 1, j, v, (i, j), ('skipP', 1, 0))
            for k, m in SEGS:
                if i + k > nP or j + m > nG: continue
                A = set().union(*(P[x]['sh'] for x in range(i, i + k)))
                B = set().union(*(G[y]['sh'] for y in range(j, j + m)))
                inter = len(A & B)
                if inter == 0: continue
                # every record in the segment must carry some of the shared text
                if k > 1 and any(len(P[x]['sh'] & B) < 2 for x in range(i, i + k)): continue
                if m > 1 and any(len(G[y]['sh'] & A) < 2 for y in range(j, j + m)): continue
                sc = inter - ALPHA * (len(A) + len(B) - 2 * inter) - BETA * (k + m - 2)
                relax(i + k, j + m, v + sc, (i, j), ('seg', k, m))
    # trace back from the end (allow trailing skips)
    i, j = nP, nG
    if j not in best[i]: j = max(best[i], key=lambda jj: best[i][jj][0])
    segs = []
    while best[i][j][1] is not None:
        _, frm, op = best[i][j]
        if op[0] == 'seg': segs.append((frm[0], op[1], frm[1], op[2]))
        i, j = frm
    return segs[::-1]

def align(coll, d):
    G = sort_graded(d['graded']); P = sort_plain(coll, d['plain'])
    for e in G: e['sh'] = shingles(words(e['body']))
    for x in P: x['sh'] = shingles(words(x['text']))
    df = collections.Counter(s for e in G for s in e['sh'])
    inv = collections.defaultdict(list)
    for j, e in enumerate(G):
        for s in e['sh']:
            if df[s] <= 8: inv[s].append(j)
    cand = {}
    for i, x in enumerate(P):
        votes = collections.Counter(j for s in x['sh'] if s in inv for j in inv[s])
        for j, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:  # ties by index, not set order
            cand[(i, j)] = pair_stats(x['sh'], G[j]['sh'])
    pairs = sorted((i, j, st[2]) for (i, j), st in cand.items() if max(st[0], st[1]) >= 0.5 and st[2] >= 0.3)
    bb = chain(pairs, len(G))
    segs = segment_dp(P, G, bb)
    # leftovers: out-of-sequence text matches
    usedP = {i for (i, k, j, m) in segs for i in range(i, i + k)}
    usedG = {j for (i, k, j, m) in segs for j in range(j, j + m)}
    cen = band_centres(bb, len(P), len(G))
    oos = []
    for i, x in enumerate(P):
        if i in usedP: continue
        c = cen[i]; sc = []
        for j in range(max(0, int(c) - 15), min(len(G), int(c) + 16)):
            st = pair_stats(x['sh'], G[j]['sh'])
            sc.append((max(st[2], st[0]), j, st))
        sc.sort(reverse=True)
        if sc and (sc[0][2][2] >= 0.3 or sc[0][2][0] >= 0.6):
            # (plain idx, graded idx, (cp, cg, dice), distance from the expected slot, runner-up score)
            oos.append((i, sc[0][1], sc[0][2], abs(sc[0][1] - c), sc[1][0] if len(sc) > 1 else 0.0, sc[0][1] in usedG))
    return G, P, cand, bb, segs, oos

def main(argv=None):
    args = path_args(__doc__).parse_args(argv)
    R = pickle.load(open(args.work / 'extract.pkl', 'rb'))
    A = {}
    for coll, d in R.items():
        G, P, cand, bb, segs, oos = align(coll, d)
        A[coll] = dict(G=G, P=P, cand=cand, bb=bb, segs=segs, oos=oos)
        shapes = collections.Counter((k, m) for i, k, j, m in segs)
        usedP = sum(k for i, k, j, m in segs); usedG = sum(m for i, k, j, m in segs)
        print(coll, 'G', len(G), 'P', len(P), 'backbone', len(bb), 'segments', len(segs), dict(shapes),
              'plain unsegmented', len(P) - usedP, 'graded unsegmented', len(G) - usedG, 'out-of-sequence', len(oos))
    pickle.dump(A, open(args.work / 'align.pkl', 'wb'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
