"""Source categories: credibility tiers, the category prior, and display labels."""
from __future__ import annotations

from typing import Any

CATEGORY_TIERS = {
    "quran": (1, "القرآن الكريم", "QURAN"),
    "hadith": (2, "الحديث الشريف", "HADITH"),
    "tafsir": (3, "التفسير", "TAFSIR"),
    "fiqh": (4, "الفقه", "FIQH"),
    "fiqh_usul": (5, "أصول الفقه", "USUL AL-FIQH"),
    "aqeedah": (5, "العقيدة", "AQEEDAH"),
    "seerah": (6, "السيرة النبوية", "SEERAH"),
}
DEFAULT_TIER = (7, "مصدر إسلامي", "OTHER")

CATEGORY_ORDER = ["quran", "hadith", "tafsir", "fiqh", "fiqh_usul", "aqeedah", "seerah"]

CATEGORY_BOOST_KEYWORDS = {
    "hadith": ["حديث", "أحاديث", "hadith", "رواه", "رسول الله", "النبي"],
    "quran": ["آية", "آيات", "سورة", "قرآن", "quran", "verse", "ayah", "surah"],
    "tafsir": ["تفسير", "tafsir", "معنى الآية", "شرح الآية"],
    "fiqh": ["فقه", "حكم", "أحكام", "مذهب", "مذاهب", "fiqh"],
    "fiqh_usul": ["أصول الفقه", "اصول الفقه", "قياس", "إجماع", "اجماع"],
    "aqeedah": ["عقيدة", "العقيدة", "توحيد", "إيمان", "ايمان", "صفات الله", "aqeedah"],
    "seerah": ["سيرة", "السيرة", "غزوة", "غزوات", "هجرة", "seerah", "sirah"],
}

CATEGORY_MATCH_FN = {
    "hadith": lambda cat: cat == "hadith",
    "quran": lambda cat: cat == "quran",
    "tafsir": lambda cat: cat == "tafsir",
    "fiqh": lambda cat: cat in ("fiqh", "fiqh_usul"),
    "fiqh_usul": lambda cat: cat == "fiqh_usul",
    "aqeedah": lambda cat: cat == "aqeedah",
    "seerah": lambda cat: cat == "seerah",
}

BOOST_FACTOR = 1.0
NON_BOOST_FACTOR = 0.9

BOOK_DISPLAY = {
    "hadith_api::sahih_al_bukhari": "صحيح البخاري",
    "hadith_api::sahih_muslim": "صحيح مسلم",
    "hadith_api::sunan_abi_dawud": "سنن أبي داود",
    "hadith_api::jami_al_tirmidhi": "جامع الترمذي",
    "hadith_api::sunan_al_nasai": "سنن النسائي",
    "hadith_api::sunan_ibn_majah": "سنن ابن ماجه",
    "hadith_api::muwatta_malik": "موطأ مالك",
    "hadith_api::musnad_ahmad": "مسند أحمد",
    "hadith_api::al_arbaun_al_nawawiyyah": "الأربعون النووية",
    "hadith_api::al_ahadith_al_qudsiyyah": "الأحاديث القدسية",
    "quran_uthmani_ar": "القرآن الكريم (عثماني)",
    "tafsir_arabic::tafsir_al_tabari": "تفسير الطبري",
    "tafsir_arabic::tafsir_ibn_kathir": "تفسير ابن كثير",
    "tafsir_arabic::tafsir_al_qurtubi": "تفسير القرطبي",
    "tafsir_arabic::tanwir_al_miqbas": "تنوير المقباس",
    "shamela:23587::irab_al_quran_al_nahhas": "إعراب القرآن (النحاس)",
    "shamela:22928::al_tibyan_fi_irab_al_quran": "التبيان في إعراب القرآن",
    "shamela:5538::mushkil_irab_al_quran": "مشكل إعراب القرآن",
}


def book_label(book_id: str | None) -> str:
    if not book_id:
        return ""
    if book_id in BOOK_DISPLAY:
        return BOOK_DISPLAY[book_id]
    if "0728IbnTaymiyya" in book_id:
        work = book_id.rsplit("::", 1)[-1].replace("_", " ")
        return f"ابن تيمية - {work}"
    if "walid_alsaeedan" in book_id:
        work = book_id.rsplit("::", 1)[-1].replace("_", " ")
        return f"وليد السعيدان - {work}"
    if book_id.startswith("shamela:"):
        return book_id.rsplit("::", 1)[-1].replace("_", " ")
    return book_id


def render_citation(parent: dict[str, Any]) -> str:
    citation = parent.get("citation") or ""
    if citation:
        return citation
    parts = []
    book = book_label(parent.get("book_id"))
    if book:
        parts.append(book)
    section = parent.get("section_path") or []
    parts.extend(str(x) for x in section if x)
    return "، ".join(parts) if parts else parent.get("parent_id", "")


def parse_categories(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    categories = []
    valid = set(CATEGORY_TIERS)
    for item in raw:
        for part in item.split(","):
            cat = part.strip()
            if not cat:
                continue
            if cat not in valid:
                raise SystemExit(f"Unknown category {cat!r}. Valid: {', '.join(sorted(valid))}")
            categories.append(cat)
    return sorted(set(categories))


def detect_source_boost_keywords(question: str) -> list[str] | None:
    q = question.lower()
    hits = []
    for cat, words in CATEGORY_BOOST_KEYWORDS.items():
        if any(word.lower() in q for word in words):
            hits.append(cat)
    return hits or None


def apply_source_prior(candidates: list[dict[str, Any]], boost_cats: list[str] | None) -> None:
    def base_score(candidate: dict[str, Any]) -> float:
        fused = candidate.get("query_fusion_score")
        if fused is not None:
            return float(fused)
        hybrid = candidate.get("hybrid_score")
        return float(hybrid if hybrid is not None else candidate["semantic_sim"])

    if not boost_cats:
        for c in candidates:
            c["weighted_score"] = round(base_score(c), 6)
        return

    boost_fns = [CATEGORY_MATCH_FN.get(cat, lambda _cat: False) for cat in boost_cats]
    for c in candidates:
        category = c.get("category") or ""
        factor = BOOST_FACTOR if any(fn(category) for fn in boost_fns) else NON_BOOST_FACTOR
        c["weighted_score"] = round(base_score(c) * factor, 6)
