"""Shared helpers for the al-Albani / plain Sunan grade mapping (read-only on the DBs)."""
import argparse, json, re, sqlite3, unicodedata
from pathlib import Path

from dalil.arabic.normalize import normalize_for_search as NFS
from dalil.paths import DATA_DIR, DB_DIR

OUT = DATA_DIR / 'catalog/hadith'    # sunan_grades.jsonl, graded_entry_numbers.jsonl; review, then gzip into recipe/decisions
WORK = OUT / 'work'                  # the steps' intermediate pickles

COLL = {  # plain collection -> (graded book, entry marker label in the graded text)
    'sunan_abi_dawud': ('book:hadith:al_albani_1332_1420:15534ebcf87c', 'سنن أبي داود'),
    'jami_al_tirmidhi': ('book:hadith:al_albani_1332_1420:b7cd5faad66a', 'سنن الترمذي'),
    'sunan_al_nasai': ('book:hadith:al_albani_1332_1420:367ad8db1bcf', 'سنن النسائي'),
    'sunan_ibn_majah': ('book:hadith:al_albani_1332_1420:5c7e61615880', 'سنن ابن ماجة'),
}

def vconn(db=DB_DIR): return sqlite3.connect(f'file:{db / "faiss_metadata.sqlite"}?mode=ro', uri=True)
def pconn(db=DB_DIR): return sqlite3.connect(f'file:{db / "parents.sqlite"}?mode=ro', uri=True)

def path_args(doc, *, db=False):
    """--work and --out (and --db) shared by the steps; their defaults are where the files live."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if db:
        ap.add_argument('--db', type=Path, default=DB_DIR, help='search stores to read (read-only)')
    ap.add_argument('--work', type=Path, default=WORK, help='intermediate pickles')
    ap.add_argument('--out', type=Path, default=OUT, help='sunan_grades.jsonl and graded_entry_numbers.jsonl')
    return ap

AR_DIG = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
def todig(s): return s.translate(AR_DIG)

# ---- text normalisation for matching (never used for display) ----
SAW = re.compile(r'صلى الله عليه (?:وآله )?وسلم|عليه السلام|رضي الله عنهما|رضي الله عنها|رضي الله عنهم|رضي الله عنه|عز وجل|تبارك وتعالى|سبحانه وتعالى')
# archaic Ketabonline spellings (graded copies) vs the vocalised hadith_api text
SPELL = [('اسمعيل', 'اسماعيل'), ('اسحق', 'اسحاق'), ('سليمن', 'سليمان'), ('عثمن', 'عثمان'), ('معوية', 'معاوية'),
         ('هرون', 'هارون'), ('ابرهيم', 'ابراهيم'), ('الحرث', 'الحارث'), ('مروان', 'مروان'), ('الرحمن', 'الرحمن'),
         ('سفين', 'سفيان'), ('ثلثة', 'ثلاثة'), ('ثلث ', 'ثلاث '), ('ملك بن', 'مالك بن'), ('القسم', 'القاسم'), ('صلوة', 'صلاة'), ('زكوة', 'زكاة')]
def norm(s):
    s = NFS(s).replace('ى', 'ي')
    s = SAW.sub(' ', s)
    s = s.replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي')
    for a, b in SPELL:
        s = s.replace(a, b)
    s = re.sub(r'(?<!\S)و\s+(?=\S)', 'و', s)      # detached waw
    return re.sub(r'\s+', ' ', s).strip()

# "قال" is inserted between isnad links in parts of hadith_api ("حدثنا قتيبة قال حدثنا ...") and not in the
# graded copies; dropping it on both sides keeps the 3-shingles comparable.
DROP = {'قال', 'قالا', 'قالوا', 'قالت'}
def words(s):
    return [w for w in norm(s).split() if not w.isdigit() and w not in DROP]

def shingles(ws, k=3):
    if len(ws) >= k:
        return {' '.join(ws[i:i + k]) for i in range(len(ws) - k + 1)}
    return {' '.join(ws)} if ws else set()

# ---- al-Albani grade label ----
# The grade text is "<verdict> <references>" or "<verdict>، <references>". The verdict is read word by word
# from a closed vocabulary (VERDICT_TOK). What follows decides the kind:
#   a reference-title start (REF_START: الأحكام / الإرواء / ابن ماجة / انظر ...), a number or nothing -> single
#   anything else ("دون قوله ...", "قصة الدجال ...", "بهذا اللفظ", "وهو مكرر ...")              -> compound, kept as written
#   after the first "،" / " - ": a qualifier (دون / إلا / لكن / والمحفوظ ...), a second verdict
#   ("ضعيف السند، صحيح المتن") or a "(...)" clause continues the verdict                        -> compound
# Per-narration verdicts "(حديث X) صحيح، (حديث Y) ضعيف" -> per_narration. Missing verdicts "**" -> no_verdict.
VERDICT_TOK = {'صحيح', 'حسن', 'ضعيف', 'جدا', 'جدًا', 'الاسناد', 'الإسناد', 'السند', 'المتن', 'لغيره', 'موقوف', 'موقوفا', 'مقطوع', 'مقطوعا',
               'مرسل', 'مرسلا', 'موضوع', 'منكر', 'شاذ', 'باطل', 'مرفوعا', 'مرفوع', 'معضل', 'مدرج', 'مضطرب', 'متواتر', 'مرفوض', 'متصل'}
BIMA = {'قبله', 'بعده', 'تقدم', 'سبق', 'مضى', 'قبلها', 'بعدها'}
REF_START = {'ابن', 'الترمذي', 'الأحكام', 'مختصر', 'أحاديث', 'احاديث', 'التعليق', 'غاية', 'تخريج', 'النسائي', 'انظر', 'أنظر', 'الظلال',
             'ظلال', 'صفة', 'صفةالصلاة', 'آداب', 'الآداب', 'الصحيحة', 'الصحيح', 'الإرواء', 'حجة', 'الضعيفة', 'الصعيفة', 'أحكام', 'الروض',
             'تيسير', 'تيسر', 'التيسير', 'ومضى', 'مضى', 'الكلم', 'الجامع', 'جزء', 'الجزء', 'حجاب', 'الحجاب', 'نقد', 'المصدر', 'أبي', 'أبى',
             'أبو', 'فقه', 'الطحاوية', 'الثمر', 'صلاة', 'السيرة', 'البيوع', 'وتقدم', 'وانظر', 'الأحاديث', 'الأجوبة', 'الترغيب', 'الرد',
             'الغاية', 'وسيأتي', 'ويأتي', 'ويأتى', 'التعليقات', 'فضائل', 'الفضائل', 'المختصر', 'التخريج', 'تعليقي', 'تعليقى', 'المشكاة',
             'مقدمة', 'أصل', 'مشكلة', 'تحذير', 'خطبة', 'سلسلة', 'سنن', 'نصب', 'تصحيح', 'الحج', 'الآيات', 'رسالتي', 'قيام', 'إصلاح',
             'التوسل', 'دفاع', 'التعلق', 'تحقيق', 'أيضا', 'أيصا', 'إيمان', 'مثله', 'وقد', 'تمام', 'الصفة', 'مسلم', 'البخاري', 'أحمد', 'الإسراء'}
QUALIFIER = {'دون', 'الا', 'إلا', 'لكن', 'والمحفوظ', 'المحفوظ', 'بلفظ', 'بزيادة', 'بدون', 'والشطر', 'الشطر', 'والجملة', 'وجملة', 'وقوله',
             'عدا', 'سوى', 'باستثناء', 'بذكر', 'بهذا', 'بهذه', 'والصحيح', 'والصواب', 'والأصح', 'المرفوع', 'وموقوف', 'والموقوف', 'ان', 'إن',
             'ما', 'فيه', 'غير', 'وفيه', 'وفي', 'إلى', 'الى', 'دونه', 'بما', 'بلفظه', 'ولكنه', 'لكنه', 'ورواية', 'رواية', 'وروي', 'وروى',
             'وزيادة', 'والزيادة', 'وهو'}
BASE = {'صحيح', 'ضعيف', 'حسن'}

def _norm_tok(t): return NFS(t.strip(':،,()-')).replace('ى', 'ي').replace('ة', 'ه')
_VT = {_norm_tok(t) for t in VERDICT_TOK}; _BM = {_norm_tok(t) for t in BIMA}
_REF = {_norm_tok(t) for t in REF_START}; _QL = {_norm_tok(t) for t in QUALIFIER}; _BASE = {_norm_tok(t) for t in BASE}
# source abbreviations that may sit between a verdict and a qualifier (م = Muslim, خ = Bukhari, ق = both, ...)
_ABBR = {_norm_tok(t) for t in ('م', 'خ', 'ق', 'ن', 'د', 'ت', 'جه', 'حه', 'ص', 'تغ', 'وق', 'وم', 'وخ', 'ك', 'حب', 'هق')}

def _walk(toks):
    """verdict words at the start of toks -> (words, index of the first word after them)"""
    out = []; k = 0
    while k < len(toks):
        n = _norm_tok(toks[k])
        # a second صحيح/ضعيف/حسن starts a reference title ("ضعيف ضعيف أبي داود (٢١٠)", "منكر ضعيف التعليق ..."),
        # except "حسن صحيح" and "... صحيح بما قبله"
        if out and n in _BASE and not ([_norm_tok(t) for t in out] == ['حسن'] and n == 'صحيح') \
                and not (k + 1 < len(toks) and _norm_tok(toks[k + 1]) == 'بما'):
            break
        if n in _VT: out.append(toks[k]); k += 1; continue
        if n == 'بما' and k + 1 < len(toks) and _norm_tok(toks[k + 1]) in _BM:
            out += toks[k:k + 2]; k += 2; continue
        break
    return out, k

def _reference_follows(toks, k):
    if k >= len(toks): return True
    t = toks[k]
    if t[0] in '(/[0123456789٠١٢٣٤٥٦٧٨٩': return True
    n = _norm_tok(t)
    if n in _ABBR:              # م / ق / خ / جه / حه / ص: a qualifier may follow ("صحيح خ دون قوله ...")
        return not (k + 1 < len(toks) and _norm_tok(toks[k + 1]) in _QL)
    return n in _REF or n in _BASE   # a second صحيح/ضعيف starts a title: "ضعيف أبي داود", "صحيح السيرة النبوية"

def _verdict(text):
    toks = text.split()
    if toks and _norm_tok(toks[0]) == 'حديث': toks = toks[1:]          # "حديث صحيح"
    out, k = _walk(toks)
    return (' '.join(out) if out else None), toks, k

def grade_label(raw):
    """(label, kind, coarse). kind: single | compound | per_narration | no_verdict | not_found | unparsed."""
    if raw is None:
        return None, 'no_verdict', None
    g = re.sub(r'\s+', ' ', raw).strip()
    if g.strip('* ') == '' or not re.search(r'[\u0621-\u064a]', g) or g.startswith('**'):
        return None, 'no_verdict', None   # "**" = no verdict; anything after it is the editor's note (// ... //)
    gn = NFS(g)
    if gn.startswith(NFS('لم أجده')) or gn.startswith(NFS('لم أقف')):
        return None, 'not_found', None
    if g.startswith('('):
        parts = []
        for m in re.finditer(r'\(([^()]{1,200})\)\s*([^،,(/]*)', g):
            v = _verdict(m.group(2).strip())[0]
            if v or m.group(2).strip().startswith('**'):
                parts.append('(%s) %s' % (m.group(1).strip(), v or '**'))
        if parts:
            return '، '.join(parts), 'per_narration', coarse_of([p.split(') ', 1)[1] for p in parts])
        return g[:120], 'unparsed', None
    if re.match(r'(?:سنده|إسناده|اسناده) ضعيف', g):
        return 'ضعيف الإسناد', 'single', 'daif'
    first = re.split(r'\s*(?:،|,|//|\s-\s|\s-$)', g, 1)[0].strip()
    v, toks, k = _verdict(first)
    if v is None:
        return g[:120], 'unparsed', None
    v = re.sub(r'\bالاسناد\b', 'الإسناد', v)
    if not _reference_follows(toks, k):
        return first[:160], 'compound', coarse_of([v])
    # after the separator: a qualifier, a second verdict or a "(...)" clause continues the verdict
    rest = g[len(first):].lstrip(' ،,-')
    rt = [t.strip(':،,') for t in rest.split()[:3]]
    if rt and _norm_tok(rt[0]) in _ABBR and len(rt) > 1:   # "م لكن قوله" / "وق دون قوله"
        rt = rt[1:]
    a = _norm_tok(rt[0]) if rt else ''
    b = rt[1] if len(rt) > 1 else ''
    second_verdict = a in _BASE and (not b or _norm_tok(b) in _VT or b.startswith('-') or _norm_tok(b) == 'بما')
    if a and (a in _QL or second_verdict or re.match(r'\(\s*[\u0621-\u064a]', rest)):
        clause = re.split(r'\s-\s*،|\s-\s|\s-$|-،|//', rest, 1)[0]
        while True:   # drop trailing references such as "، الصحيحة (٢٧٥٤)"
            c2 = re.sub(r'[،,]\s*[^،,"]{0,40}\([0-9\u0660-\u0669][^)]*\)\s*$', '', clause)
            if c2 == clause: break
            clause = c2
        return (v + '، ' + clause.strip(' ،,-'))[:160], 'compound', coarse_of([v])
    return v, 'single', coarse_of([v])

_COARSE = [('موضوع', 'mawdu'), ('باطل', 'mawdu'), ('منكر', 'munkar'), ('شاذ', 'shadh'), ('ضعيف جدا', 'daif_jiddan'), ('ضعيف الإسناد جدا', 'daif_jiddan'),
           ('ضعيف', 'daif'), ('حسن صحيح', 'sahih'), ('صحيح', 'sahih'), ('حسن', 'hasan')]
def coarse_of(verdicts):
    cs = set()
    for v in verdicts:
        vn = NFS(v)
        if vn.startswith('**'): continue
        cs.add(next((c for k, c in _COARSE if vn.startswith(NFS(k))), 'other'))
    if not cs: return None
    return cs.pop() if len(cs) == 1 else 'mixed'

def jl_write(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
