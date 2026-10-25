#!/usr/bin/env python3
"""Source-preserving Arabic text cleaning utilities for corpus ingestion.

Principle: never collapse the source/display text into the embedding text.  These
helpers create a light-normalized representation for retrieval while returning
metadata and quality counters so page anchors, OpenITI metadata, and source
artifacts can be preserved outside the embedded string.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u066E-\u06D3\u06FA-\u06FF]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
BIDI_CONTROL_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
C0_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PDF_INVISIBLE_RE = re.compile(r"[\u2000-\u200d\u2060]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s+[^<>]*)?>")
HTML_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);")
OPENITI_PAGE_RE = re.compile(r"\bPageV\d+P\d+\b")
OPENITI_HEADER_RE = re.compile(r"^######OpenITI#\s*$", re.M)
OPENITI_META_LINE_RE = re.compile(r"^#META#\s*(.*)$", re.M)
OPENITI_HEADING_RE = re.compile(r"^#{1,3}\s+(?:\|+\s*)?(.*)$", re.M)
OPENITI_NESTED_MARK_RE = re.compile(r"#{1,3}[ \t]*\|+[ \t]*")
OPENITI_STRUCTURAL_PREFIX_RE = re.compile(r"^\s*#\s?", re.M)
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANKLINES_RE = re.compile(r"\n{3,}")

PERSIAN_URDU_TRANSLATION = str.maketrans(
    {
        "ک": "ك",
        "ڪ": "ك",
        "ی": "ي",
        "ى": "ى",  # preserve alif maqsurah; not Persian yeh
        "ے": "ي",
        "ۀ": "ة",
        "ۂ": "ة",
    }
)


def _collapse_spacing(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_unicode_for_embedding(text: str) -> str:
    """Unicode-normalize and remove controls unsafe for embeddings/PDF text."""
    text = unicodedata.normalize("NFKC", text)
    text = BIDI_CONTROL_RE.sub("", text)
    text = C0_CONTROL_RE.sub("", text)
    text = PDF_INVISIBLE_RE.sub(" ", text)
    return _collapse_spacing(text)


def normalize_arabic_light(text: str, *, dediac: bool = True) -> str:
    """Light Arabic normalization for embedding/search.

    Non-goals are intentional: no ta-marbuta-to-ha, no hamza deletion, no
    stemming, and no deletion of sanad formulae.
    """
    text = normalize_unicode_for_embedding(text)
    text = text.translate(PERSIAN_URDU_TRANSLATION)
    text = text.replace(TATWEEL, "")
    if dediac:
        text = DIACRITICS_RE.sub("", text)
    return _collapse_spacing(text)


def parse_openiti(text: str) -> dict[str, Any]:
    """Extract OpenITI metadata, page markers, and headings without losing text."""
    meta: dict[str, str] = {}
    raw_meta: list[str] = []
    for match in OPENITI_META_LINE_RE.finditer(text):
        body = match.group(1).strip()
        raw_meta.append(body)
        if "::" in body:
            key, value = body.split("::", 1)
            meta[key.strip()] = value.strip()
        elif ":" in body:
            key, value = body.split(":", 1)
            meta[key.strip()] = value.strip()
        elif body:
            meta.setdefault("_unparsed", "")
            meta["_unparsed"] = (meta["_unparsed"] + "\n" + body).strip()

    headings = [m.group(1).strip() for m in OPENITI_HEADING_RE.finditer(text) if m.group(1).strip()]
    pages = OPENITI_PAGE_RE.findall(text)
    return {
        "meta": meta,
        "raw_meta": raw_meta,
        "page_markers": pages,
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "headings": headings,
        "book_uri": meta.get("URI") or meta.get("book_uri"),
        "source_type": "openiti",
    }


def strip_openiti_markup_for_embedding(text: str) -> tuple[str, dict[str, Any]]:
    """Remove OpenITI control syntax from embedding text and return metadata."""
    parsed = parse_openiti(text)
    cleaned = OPENITI_HEADER_RE.sub("", text)
    cleaned = OPENITI_META_LINE_RE.sub("", cleaned)
    cleaned = OPENITI_PAGE_RE.sub(" ", cleaned)
    # Preserve heading words, remove only mARkdown heading pipes/hashes.
    cleaned = OPENITI_HEADING_RE.sub(lambda m: m.group(1).strip(), cleaned)
    # Some Shamela/OpenITI conversions nest editorial markers, e.g.
    # ``### | AUTO ### | CHECK [heading]``.  Keep the heading text while
    # removing those conversion controls from the embedding representation.
    cleaned = OPENITI_NESTED_MARK_RE.sub("", cleaned)
    cleaned = re.sub(r"^[ \t]*(?:(?:AUTO|CHECK)[ \t]*)+", "", cleaned, flags=re.M)
    cleaned = cleaned.replace("~~", "")
    cleaned = re.sub(r"!\s*\d+\s*<", "", cleaned)
    cleaned = re.sub(r"\$+", "", cleaned)
    cleaned = OPENITI_STRUCTURAL_PREFIX_RE.sub("", cleaned)
    cleaned = normalize_arabic_light(cleaned, dediac=True)
    meta = {
        **parsed,
        "openiti_tags_removed": text.count("#META#") + text.count("######OpenITI") + text.count("~~"),
    }
    return cleaned, meta


def strip_pdf_artifacts(text: str) -> tuple[str, dict[str, int]]:
    """Remove invisible PDF/OCR artifacts but not repeated religious formulae."""
    counts = {
        "bidi_controls_removed": len(BIDI_CONTROL_RE.findall(text)),
        "c0_controls_removed": len(C0_CONTROL_RE.findall(text)),
        "pdf_invisible_removed": len(PDF_INVISIBLE_RE.findall(text)),
    }
    cleaned = normalize_unicode_for_embedding(text)
    return cleaned, counts


def remove_urls_and_html(text: str) -> tuple[str, dict[str, int]]:
    """Remove URLs, HTML tags, and entities from embedding text with counts."""
    counts = {
        "urls_removed": len(URL_RE.findall(text)),
        "html_tags_removed": len(HTML_TAG_RE.findall(text)),
        "html_entities_removed": len(HTML_ENTITY_RE.findall(text)),
    }
    cleaned = URL_RE.sub(" ", text)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = HTML_ENTITY_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = _collapse_spacing(cleaned)
    return cleaned, counts


def score_arabic_quality(text: str) -> dict[str, Any]:
    """Compute simple quality metrics for Arabic source/package gates."""
    chars = len(text)
    arabic_letters = len(ARABIC_LETTER_RE.findall(text))
    latin_letters = len(re.findall(r"[A-Za-z]", text))
    nonspace_printable = sum(1 for c in text if c.isprintable() and not c.isspace())
    return {
        "chars": chars,
        "arabic_letters": arabic_letters,
        "arabic_letter_ratio": round(arabic_letters / max(nonspace_printable, 1), 4),
        "latin_letters": latin_letters,
        "latin_ratio": round(latin_letters / max(nonspace_printable, 1), 4),
        "diacritics": len(DIACRITICS_RE.findall(text)),
        "bidi_controls": len(BIDI_CONTROL_RE.findall(text)),
        "c0_controls": len(C0_CONTROL_RE.findall(text)),
        "replacement_chars": text.count("�"),
        "urls": len(URL_RE.findall(text)),
        "html_tags": len(HTML_TAG_RE.findall(text)),
        "html_entities": len(HTML_ENTITY_RE.findall(text)),
        "openiti_marks": text.count("#META#") + text.count("######OpenITI") + text.count("~~") + text.count("### |"),
    }


def make_embedding_text(display_text: str, source_type: str) -> tuple[str, dict[str, Any]]:
    """Create `text_embedding` and cleaning/quality metadata from display text."""
    source_type = (source_type or "plain").lower()
    aggregate: dict[str, Any] = {"source_type": source_type, "cleaning_profile": "arabic_light_v1"}

    text = display_text
    if source_type == "openiti":
        text, openiti_meta = strip_openiti_markup_for_embedding(text)
        aggregate.update(openiti_meta)
    else:
        text, pdf_counts = strip_pdf_artifacts(text)
        aggregate.update(pdf_counts)

    text, html_counts = remove_urls_and_html(text)
    aggregate.update(html_counts)
    text = normalize_arabic_light(text, dediac=True)
    aggregate.update(score_arabic_quality(text))
    return text, aggregate


__all__ = [
    "make_embedding_text",
    "normalize_unicode_for_embedding",
    "normalize_arabic_light",
    "parse_openiti",
    "remove_urls_and_html",
    "score_arabic_quality",
    "strip_openiti_markup_for_embedding",
    "strip_pdf_artifacts",
]
