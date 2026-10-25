"""Quran citations: surah names and the citation forms that occur in this corpus.
"""
from __future__ import annotations

import json
import re

from dalil.paths import ORIGINAL_DIR

from .records import normalize_arabic, safe_num

# Surah names as spelled in the core corpus's Quran source, loaded on first use.
QURAN_SOURCE_DIR = ORIGINAL_DIR / "القرآن الكريم"
_SOURCE_SURAH_IDS: dict[str, int] | None = None


def surah_name_to_id() -> dict[str, int]:
    global _SOURCE_SURAH_IDS
    if _SOURCE_SURAH_IDS is not None:
        return _SOURCE_SURAH_IDS
    candidates = list(QURAN_SOURCE_DIR.rglob("quran-ar.json"))
    mapping: dict[str, int] = {}
    if candidates:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        for row in data:
            sid = int(row["id"])
            name = normalize_arabic(row["name"])
            mapping[name] = sid
    _SOURCE_SURAH_IDS = mapping
    return mapping


def extract_numeric_quran_ref(text: str) -> tuple[int, int, int] | None:
    if not text:
        return None
    m = re.search(r"سورة\s*\((\d{1,3})\)\s*:\s*آية\s*(\d{1,3})", text)
    if m:
        surah = int(m.group(1))
        ayah = int(m.group(2))
        return surah, ayah, ayah
    m = re.search(r"\[(\d{1,3})\s+([^\]]{2,30})\]", text)
    if m:
        ayah = int(m.group(1))
        # corpus style [21 البقرة]
        # Surah name resolution is intentionally deferred unless source provides a number.
        return None
    normalized = normalize_arabic(text)
    surah = None
    for name, sid in sorted(surah_name_to_id().items(), key=lambda item: len(item[0]), reverse=True):
        if name and re.search(rf"\bسوره\s+{re.escape(name)}\b", normalized):
            surah = sid
            break
    if surah:
        paren_numbers = [safe_num(m.group(1), 0) for m in re.finditer(r"\(([0-9٠-٩]{1,3})\)", text)]
        paren_numbers = [n for n in paren_numbers if n > 0]
        if paren_numbers:
            return surah, min(paren_numbers), max(paren_numbers)
    return None


AYAH_NUM_RE = re.compile(r"الآي[ةات]\s*\(([0-9٠-٩]{1,3})(?:\s*[-–—]\s*([0-9٠-٩]{1,3}))?\)")
QURAN_REF_RE = re.compile(r"سورة\s*\((\d{1,3})\)\s*:\s*آي[ةات]?\s*\(?(\d{1,3})\)?")


def extract_ayah_ref(text: str) -> tuple[int, int, int] | None:
    """Extract (surah, ayah_start, ayah_end) from 'الآية (N)' style headings.

    Surah is unknown from the heading alone; returns surah=0 and the caller
    may leave the quran ref unset (surah 0 is invalid per schema).
    """
    m = AYAH_NUM_RE.search(text or "")
    if m:
        start = safe_num(m.group(1))
        end = safe_num(m.group(2), start) if m.group(2) else start
        return 0, start, end
    return None


SURAH_NAME_TO_ID: dict[str, int] = {
    "الفاتحة": 1, "البقرة": 2, "آل عمران": 3, "النساء": 4, "المائدة": 5,
    "الأنعام": 6, "الأعراف": 7, "الأنفال": 8, "التوبة": 9, "يونس": 10,
    "هود": 11, "يوسف": 12, "الرعد": 13, "إبراهيم": 14, "الحجر": 15,
    "النحل": 16, "الإسراء": 17, "الكهف": 18, "مريم": 19, "طه": 20,
    "الأنبياء": 21, "الحج": 22, "المؤمنون": 23, "النور": 24, "الفرقان": 25,
    "الشعراء": 26, "النمل": 27, "القصص": 28, "العنكبوت": 29, "الروم": 30,
    "لقمان": 31, "السجدة": 32, "الأحزاب": 33, "سبأ": 34, "فاطر": 35,
    "يس": 36, "الصافات": 37, "ص": 38, "الزمر": 39, "غافر": 40,
    "فصلت": 41, "الشورى": 42, "الزخرف": 43, "الدخان": 44, "الجاثية": 45,
    "الأحقاف": 46, "محمد": 47, "الفتح": 48, "الحجرات": 49, "ق": 50,
    "الذاريات": 51, "الطور": 52, "النجم": 53, "القمر": 54, "الرحمن": 55,
    "الواقعة": 56, "الحديد": 57, "المجادلة": 58, "الحشر": 59, "الممتحنة": 60,
    "الصف": 61, "الجمعة": 62, "المنافقون": 63, "التغابن": 64, "الطلاق": 65,
    "التحريم": 66, "الملك": 67, "القلم": 68, "الحاقة": 69, "المعارج": 70,
    "نوح": 71, "الجن": 72, "المزمل": 73, "المدثر": 74, "القيامة": 75,
    "الإنسان": 76, "المرسلات": 77, "النبأ": 78, "النازعات": 79, "عبس": 80,
    "التكوير": 81, "الانفطار": 82, "المطففين": 83, "الانشقاق": 84, "البروج": 85,
    "الطارق": 86, "الأعلى": 87, "الغاشية": 88, "الفجر": 89, "البلد": 90,
    "الشمس": 91, "الليل": 92, "الضحى": 93, "الشرح": 94, "التين": 95,
    "العلق": 96, "القدر": 97, "البينة": 98, "الزلزلة": 99, "العاديات": 100,
    "القارعة": 101, "التكاثر": 102, "العصر": 103, "الهمزة": 104, "الفيل": 105,
    "قريش": 106, "الماعون": 107, "الكوثر": 108, "الكافرون": 109, "النصر": 110,
    "المسد": 111, "الإخلاص": 112, "الفلق": 113, "الناس": 114,
}


def surah_from_title(title: str) -> int | None:
    """Find a surah id mentioned in a book title (e.g. 'تفسير سورة الزمر' or 'تفسير العثيمين: النساء')."""
    norm = normalize_arabic(title or "")
    for name, sid in sorted(SURAH_NAME_TO_ID.items(), key=lambda kv: -len(kv[0])):
        nname = normalize_arabic(name)
        if re.search(rf"سوره\s+{re.escape(nname)}", norm):
            return sid
        if re.search(rf"[:：]\s*{re.escape(nname)}\s*$", norm):
            return sid
    return None


# Normalized surah lookup, plus the alternate names that appear in print.
SURAH_LOOKUP: dict[str, int] = {normalize_arabic(n): i for n, i in SURAH_NAME_TO_ID.items()}
SURAH_LOOKUP.update(
    {
        normalize_arabic(alias): sid
        for alias, sid in {
            "بني إسرائيل": 17, "الإسراء": 17, "المؤمن": 40, "الطول": 40,
            "الدهر": 76, "براءة": 9, "القتال": 47, "فاطر": 35,
            "الملائكة": 35, "حم السجدة": 41, "المنافقين": 63,
            "الشرح": 94, "الانشراح": 94, "الإخلاص": 112, "التوحيد": 112,
        }.items()
    }
)
# Longest names first so 'آل عمران' is not shadowed by a shorter partial match.
SURAH_NAMES_SORTED = sorted(SURAH_LOOKUP, key=len, reverse=True)
# [البقرة: 255] / (البقرة : 255-257) / [سورة النساء/59]
BRACKET_QURAN_RE = re.compile(
    r"[\[\(]\s*(?:سورة\s+)?([ء-ي][ء-ي\s]{1,18}?)\s*[:：/،]\s*"
    r"([0-9٠-٩]{1,3})\s*(?:[-–—]\s*([0-9٠-٩]{1,3}))?\s*[\]\)]"
)
# سورة البقرة الآية 185 / [البقرة الآية 6]
NAMED_AYAH_RE = re.compile(
    r"(?:سورة\s+)?([ء-ي][ء-ي\s]{1,18}?)\s*[,،]?\s*الآي[ةات]\s*[:：]?\s*"
    r"\(?([0-9٠-٩]{1,3})\)?\s*(?:[-–—]\s*\(?([0-9٠-٩]{1,3})\)?)?"
)
NAMED_AYAH_LITERAL = "الآي"
assert NAMED_AYAH_LITERAL in NAMED_AYAH_RE.pattern
MAX_QURAN_REFS_PER_UNIT = 64


def _ayah_range(start_raw: str, end_raw: str | None) -> tuple[int, int] | None:
    start = safe_num(start_raw)
    if not 1 <= start <= 286:
        return None
    end = safe_num(end_raw, start) if end_raw else start
    if end < start or end > 286:
        end = start
    return start, end


def extract_all_quran_refs(text: str) -> list[tuple[int, int, int]]:
    """Extract every Quran citation in `text` as (surah, ayah_start, ayah_end).

    Recognises the three forms that actually occur in this corpus:
      - 'سورة (2) : آية (255)'  — the numeric OpenITI style
      - '[البقرة: 255]' / '(النساء : 59-60)'  — the dominant printed style
      - 'سورة البقرة الآية 185' / '[البقرة الآية 6]'
    Surah names are resolved against the canonical list, so bracketed text that
    is not a surah name is ignored. Results are deduplicated and order-stable.
    """
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(surah: int, span: tuple[int, int] | None) -> None:
        if not span or not 1 <= surah <= 114 or len(out) >= MAX_QURAN_REFS_PER_UNIT:
            return
        key = (surah, span[0], span[1])
        if key not in seen:
            seen.add(key)
            out.append(key)

    if not text:
        return out
    for m in QURAN_REF_RE.finditer(text):
        add(safe_num(m.group(1)), _ayah_range(m.group(2), None))
    for pattern in (BRACKET_QURAN_RE, NAMED_AYAH_RE):
        if pattern is NAMED_AYAH_RE and NAMED_AYAH_LITERAL not in text:
            # Every match contains this literal. Without it the scan can only fail,
            # and it is slow: the pattern can start at any Arabic letter.
            continue
        for m in pattern.finditer(text):
            surah = SURAH_LOOKUP.get(normalize_arabic(m.group(1)))
            if surah is None:
                continue
            add(surah, _ayah_range(m.group(2), m.group(3)))
    return out
