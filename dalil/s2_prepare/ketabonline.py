"""Prepare Ketabonline text packages for structure review.

This is a preparation-only stage.  It:

* reads the downloaded ``files.data`` JSON packages without touching them;
* preserves page, part, native-index, paragraph, and HTML-title structure;
* applies the existing ``dalil.arabic.clean.make_embedding_text`` helper to
  each emitted element;
* writes book labels, element JSONL, quality metrics, and conservative
  duplicate/review decisions.

It deliberately does not build parents, retrieval views, chunks, embeddings,
FAISS rows, or SQLite rows.  Raw packages remain the source of truth.

    python -m dalil.s2_prepare.ketabonline [--output DIR] [--limit N]

Single packages outside the collection (an addition's own download), with an
author slug the built-in table lacks, and no overlap labels:

    python -m dalil.s2_prepare.ketabonline --raw DIR --output DIR --author walid_alsaeedan=walid_alsaeedan \
        --no-db-overlap
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import sqlite3
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dalil.arabic.clean import make_embedding_text, normalize_arabic_light
from dalil.paths import BUILD_DIR, DATA_DIR, DB_DIR, ORIGINAL_DIR, RECIPE_DIR

# Packages sit in their book folders (books/original/<author>/<book>/<id>.data.zip);
# the download records, with each package's author slug and sha256, are in recipe/.
KETAB_ROOT = ORIGINAL_DIR
CATALOG_PATH = RECIPE_DIR / "sources" / "ketabonline_catalog.json"
DOWNLOAD_MANIFEST_PATH = RECIPE_DIR / "sources" / "ketabonline_downloads.json"
CURRENT_DB_PATH = DB_DIR / "faiss_metadata.sqlite"
CURRENT_PARENTS_DB_PATH = DB_DIR / "parents.sqlite"
# Not the handoff the extractor reads (dalil.s2_prepare.ketabonline_handoff, from
# the deduplicated manifest in recipe/): this is the pre-dedup layer before it.
DEFAULT_OUTPUT = BUILD_DIR / "prepared" / "ketabonline_all"


AUTHOR_IDS = {
    "ibn_baz": "ibn_baz_1330_1420",
    "al_albani": "al_albani_1332_1420",
    "ibn_al_qayyim": "ibn_al_qayyim_691_751",
}

CATEGORY_MAP = {
    "العقيدة": "aqeedah",
    "عقائد": "aqeedah",
    "الفرق والردود": "aqeedah",
    "الفتاوى": "fiqh",
    "فقه عام": "fiqh",
    "فقه حنبلي": "fiqh",
    "أصول الفقه والقواعد الفقهية": "fiqh_usul",
    "متون الحديث": "hadith",
    "علوم الحديث": "hadith",
    "شروح الحديث": "hadith",
    "كتب التخريج والزوائد": "hadith",
    "التفاسير": "tafsir",
    "التفاسير والترجمات": "tafsir",
    "علوم القرآن": "tafsir",
    "التراجم والطبقات": "seerah",
}

TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:-]*)\b([^>]*)>", re.S)
ATTR_RE = re.compile(r"([A-Za-z_:][\w:.-]*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)")
TITLE_CLASS_RE = re.compile(r"(?:^|\s)g-title(?:\s|$)", re.I)
TITLE_DATA_RE = re.compile(r"(?:^|\s)data-type\s*=\s*(['\"])title\1", re.I)
HTML_BLOCK_TAGS = {"p", "div", "span"}
EDITION_SUFFIX_RE = re.compile(
    r"\s*(?:[-–—:]\s*)?(?:ط\.?\s*[-.]?\s*(?:أخرى|اولى|الأولى|الثانية|المجمع|عالم الفوائد)?|"
    r"الطبعة\s+[^،؛:]+|دار\s+[^،؛:]+|المكتبة\s+[^،؛:]+|"
    r"عالم الفوائد|الكتاب العربي|المجمع)\s*$",
    re.I,
)
VOLUME_RE = re.compile(r"\s+(?:ج|الجزء)\s*[0-9٠-٩]+\b", re.I)
AUTHOR_SUFFIX_RE = re.compile(r"\s+(?:ل|لابن|لإبن)\s+القيم\b", re.I)
# The index writes "س 2" where the page has "س ٢".
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
# A block is an index entry's heading if its text is the entry's title, or starts
# with it when the title is long enough not to open ordinary prose ("فصل").
MIN_PREFIX_KEY = 12


@dataclass
class Block:
    kind: str
    start: int
    end: int
    raw: str
    tag: str
    attrs: dict[str, str]
    classes: list[str]
    parent_p_id: str | None
    order: int


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_key_text(text: str) -> str:
    text = normalize_arabic_light(text or "", dediac=True)
    text = re.sub(r"[\u2000-\u206f\u2e00-\u2e7f\u3000\"'“”‘’]", " ", text)
    return " ".join(text.split())


def title_key_exact(title: str) -> str:
    return re.sub(r"[^\w\u0600-\u06ff]+", "", clean_key_text(title), flags=re.UNICODE)


def title_key_work(title: str) -> str:
    text = clean_key_text(title)
    text = VOLUME_RE.sub("", text)
    text = AUTHOR_SUFFIX_RE.sub("", text)
    # Edition/source suffixes are review signals, not proof of identity.
    previous = None
    while previous != text:
        previous = text
        text = EDITION_SUFFIX_RE.sub("", text).strip(" -–—:،؛")
    return re.sub(r"[^\w\u0600-\u06ff]+", "", text, flags=re.UNICODE)


def slug_from_title(title: str) -> str:
    key = title_key_work(title)
    return key[:96] or "untitled"


def parse_attrs(tag_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in ATTR_RE.finditer(tag_text):
        value = match.group(2)
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        attrs[match.group(1).lower()] = html_module.unescape(value)
    return attrs


def is_title_tag(tag: str, attrs: dict[str, str]) -> bool:
    classes = attrs.get("class", "").split()
    return bool(TITLE_CLASS_RE.search(attrs.get("class", ""))) or (
        tag == "span" and TITLE_DATA_RE.search(" ".join(f'{k}="{v}"' for k, v in attrs.items())) is not None
    ) or "g-title" in classes


def extract_blocks(content: str) -> tuple[list[Block], Counter[str]]:
    """Extract p and title blocks while retaining offsets into page content.

    A `g-table` row outside any p is a paragraph too: Ketabonline renders Shamela's
    two-part lines (a verse's hemistichs, "A ... B") as
    `<div class="g-table"><div class="g-cell">A</div><div class="g-border"> ... </div>...</div>`.
    """
    blocks: list[Block] = []
    stack: list[dict[str, Any]] = []
    tags = Counter()
    block_order = 0

    for match in TAG_RE.finditer(content or ""):
        closing, tag, attr_text = match.groups()
        tag = tag.lower()
        tags[tag] += 1
        # The cells of a g-table row open and close their own divs.
        in_row = stack[-1] if stack and "open_divs" in stack[-1] else None
        if in_row is not None and tag == "div" and (in_row["open_divs"] or not closing):
            in_row["open_divs"] += -1 if closing else 1
            continue
        if closing:
            target_index = None
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx]["tag"] == tag:
                    target_index = idx
                    break
            if target_index is None:
                continue
            target = stack[target_index]
            del stack[target_index:]
            blocks.append(
                Block(
                    kind=target["kind"],
                    start=target["start"],
                    end=match.end(),
                    raw=content[target["start"] : match.end()],
                    tag=tag,
                    attrs=target["attrs"],
                    classes=target["attrs"].get("class", "").split(),
                    parent_p_id=target.get("parent_p_id"),
                    order=target["order"],
                )
            )
            continue

        attrs = parse_attrs(match.group(0))
        if tag not in HTML_BLOCK_TAGS:
            continue
        p_id = attrs.get("id") if tag == "p" else None
        if tag == "p":
            block_order += 1
            stack.append({"kind": "paragraph", "tag": tag, "start": match.start(), "attrs": attrs, "parent_p_id": p_id, "order": block_order})
        elif tag == "div" and "g-table" in attrs.get("class", "").split() and not any(s["tag"] == "p" for s in stack):
            block_order += 1
            stack.append({"kind": "paragraph", "tag": tag, "start": match.start(), "attrs": attrs, "parent_p_id": None,
                          "order": block_order, "open_divs": 0})
        elif is_title_tag(tag, attrs):
            parent_p_id = None
            for parent in reversed(stack):
                if parent["tag"] == "p":
                    parent_p_id = parent["attrs"].get("id")
                    break
            block_order += 1
            stack.append({"kind": "html_heading", "tag": tag, "start": match.start(), "attrs": attrs, "parent_p_id": parent_p_id, "order": block_order})

    # Be conservative with malformed HTML: close an unclosed block at page end.
    for target in reversed(stack):
        blocks.append(
            Block(
                kind=target["kind"],
                start=target["start"],
                end=len(content),
                raw=content[target["start"] :],
                tag=target["tag"],
                attrs=target["attrs"],
                classes=target["attrs"].get("class", "").split(),
                parent_p_id=target.get("parent_p_id"),
                order=target["order"],
            )
        )

    blocks.sort(key=lambda b: (b.start, 0 if b.kind == "html_heading" else 1, b.end))
    return blocks, tags


def remove_nested_titles(paragraph: Block, titles: list[Block], content: str) -> str:
    ranges = [(t.start, t.end) for t in titles if paragraph.start <= t.start and t.end <= paragraph.end]
    if not ranges:
        return paragraph.raw
    pieces: list[str] = []
    cursor = paragraph.start
    for start, end in sorted(ranges):
        if cursor < start:
            pieces.append(content[cursor:start])
        cursor = max(cursor, end)
    if cursor < paragraph.end:
        pieces.append(content[cursor:paragraph.end])
    return " ".join(pieces)


def flatten_index(nodes: Any, parent_id: int | str | None = None, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if out is None:
        out = []
    if not isinstance(nodes, list):
        return out
    for order, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        entry = {
            "id": node.get("id"),
            "parent": node.get("parent", parent_id),
            "page_id": node.get("page_id"),
            "page": node.get("page"),
            "title_level": node.get("title_level", 1),
            "title": str(node.get("title") or "").strip(),
            "part_name": str(node.get("part_name") or "").strip(),
            "order": len(out),
        }
        if entry["title"] and entry["id"] is not None:
            out.append(entry)
            child_parent = entry["id"]
        else:
            child_parent = parent_id
        flatten_index(node.get("children"), child_parent, out)
    return out


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def category_info(data: dict[str, Any], fallback: dict[str, Any] | None = None) -> tuple[str, str | None, str | None]:
    categories = data.get("categories") or []
    if not any(isinstance(row, dict) and row.get("name") for row in categories) and fallback:
        categories = fallback.get("categories") or []
    raw = None
    category_id = None
    if categories and isinstance(categories[0], dict):
        raw = categories[0].get("name")
        category_id = str(categories[0].get("id")) if categories[0].get("id") is not None else None
    return CATEGORY_MAP.get(raw or "", "general"), raw, category_id


def source_group_from_path(path: Path) -> str:
    for group in AUTHOR_IDS:
        if f"/{group}/" in str(path):
            return group
    return "unknown"


def read_package(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".json")]
        if not names:
            raise ValueError(f"no JSON member in {zip_path}")
        return json.loads(archive.read(names[0]))


def package_paths(root: Path = KETAB_ROOT) -> list[Path]:
    return sorted(root.rglob("*.zip"))


def load_catalog(catalog_path: Path = CATALOG_PATH, manifest_path: Path = DOWNLOAD_MANIFEST_PATH,
                 ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog_by_id = {str(row["id"]): row for row in catalog.get("records", [])}
    return catalog_by_id, {str(k): v for k, v in manifest.items()}


def load_current_title_catalog() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_hashes: set[str] = set()
    if not CURRENT_DB_PATH.exists():
        return matches, block_hashes
    con = sqlite3.connect(CURRENT_DB_PATH)
    try:
        rows = con.execute("SELECT book_id, section_path_json FROM child_metadata GROUP BY book_id").fetchall()
        for book_id, section_json in rows:
            try:
                section = json.loads(section_json or "[]")
            except json.JSONDecodeError:
                section = []
            title = str(section[0]) if section else str(book_id)
            for key in {title_key_exact(title), title_key_work(title)}:
                if key:
                    matches[key].append({"book_id": book_id, "title_ar": title, "source": "faiss_metadata"})
    finally:
        con.close()
    if CURRENT_PARENTS_DB_PATH.exists():
        con = sqlite3.connect(CURRENT_PARENTS_DB_PATH)
        try:
            for (text,) in con.execute("SELECT text_original FROM parents WHERE length(text_original) >= 80"):
                normalized = normalize_arabic_light(text or "", dediac=True)
                if len(normalized) >= 80:
                    block_hashes.add(sha1_text(normalized))
        finally:
            con.close()
    return matches, block_hashes


def load_today_title_catalog() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_hashes: set[str] = set()

    def add(title: Any, source: str, path: str = "") -> None:
        if not title:
            return
        title = str(title).strip()
        row = {"title_ar": title, "source": source, "path": path}
        for key in {title_key_exact(title), title_key_work(title)}:
            if key:
                matches[key].append(row)

    for path in ORIGINAL_DIR.rglob("book_metadata.json"):
        try:
            add(json.loads(path.read_text(encoding="utf-8")).get("title_ar"), "shamela4", str(path))
        except (OSError, json.JSONDecodeError):
            pass

    for path in ORIGINAL_DIR.glob("*/manifest.json"):  # OpenITI's, one per author
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = data if isinstance(data, list) else data.get("books", data.get("records", [])) if isinstance(data, dict) else []
        if isinstance(rows, dict):
            rows = list(rows.values())
        for row in rows:
            if isinstance(row, dict):
                add(row.get("title_ar") or row.get("title") or row.get("book_title"), "openiti_manifest", str(path))

    # Existing cleaning-only pages are the strongest same-day block-level check.
    for path in ORIGINAL_DIR.rglob("*_cleaned.jsonl"):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text_clean") or row.get("answer_clean") or row.get("cleaned_text") or ""
                    normalized = normalize_arabic_light(str(text), dediac=True)
                    if len(normalized) >= 80:
                        block_hashes.add(sha1_text(normalized))
        except OSError:
            continue
    return matches, block_hashes


def make_book_summary(
    zip_path: Path,
    data: dict[str, Any],
    catalog_row: dict[str, Any],
    download_row: dict[str, Any],
) -> dict[str, Any]:
    group = download_row.get("target_author_slug") or source_group_from_path(zip_path)
    author_id = AUTHOR_IDS.get(group)
    package_id = data.get("id") or catalog_row.get("id") or download_row.get("book_id")
    title = str(data.get("title") or catalog_row.get("title") or download_row.get("title") or f"untitled-{package_id}").strip()
    category, category_raw, category_id = category_info(data, catalog_row)
    package_kind = str(download_row.get("package_kind") or "text_book")
    source_label = str(download_row.get("source") or "")
    language = str(data.get("lang") or catalog_row.get("lang") or "unknown")
    text_role = "transcription" if package_kind == "text_transcript" or "تفريغ" in title or "مفرغ" in title else "book_text"
    exact_key = title_key_exact(title)
    work_key = title_key_work(title)
    work_hash = sha1_text(f"{author_id or group}|{work_key}")[:12]
    book_id = f"ketabonline:{group}::{package_id}"
    work_id = f"ketabonline_work:{group}:{work_hash}"
    return {
        "book_id": book_id,
        "work_id": work_id,
        "author_id": author_id,
        "author_slug": group,
        "title_ar": title,
        "title_key_exact": exact_key,
        "title_key_work": work_key,
        "language": language,
        "category": category,
        "category_raw": category_raw,
        "category_id": category_id,
        "text_role": text_role,
        "package_kind": package_kind,
        "source_label": source_label,
        "package_id": package_id,
        "source_path": str(zip_path.relative_to(DATA_DIR)),
        "source_url": download_row.get("url") or catalog_row.get("source_url"),
        "archive_sha256": download_row.get("archive_sha256"),
        "pages_declared": data.get("pages_count", catalog_row.get("pages_count")),
        "parts_declared": data.get("parts_count", catalog_row.get("parts_count")),
        "index_declared": data.get("index_count", catalog_row.get("index_count")),
        "raw_bytes": zip_path.stat().st_size,
        "status": "unprocessed",
        "review_flags": [],
    }


def quality_flags(cleaned: str, metrics: dict[str, Any], language: str, text_role: str) -> list[str]:
    flags: list[str] = []
    if not cleaned.strip():
        flags.append("empty_after_cleaning")
    if len(cleaned) >= 30 and language.lower() in {"ar", "ara", "arabic"} and metrics.get("arabic_letter_ratio", 0) < 0.6:
        flags.append("low_arabic_ratio")
    if metrics.get("replacement_chars", 0):
        flags.append("replacement_chars")
    if metrics.get("latin_ratio", 0) > 0.5:
        flags.append("latin_dominant")
    if metrics.get("html_tags_removed", 0):
        flags.append("html_markup_removed")
    if text_role == "transcription":
        flags.append("text_transcription")
    if language.lower() not in {"ar", "ara", "arabic"}:
        flags.append("non_arabic_language")
    return sorted(set(flags))


def token_counts(text: str) -> list[dict[str, Any]]:
    return [{"tokenizer_name": "whitespace_ar_v1", "token_count": len((text or "").split())}]


def refs_for(
    *,
    record_id: str,
    record_index: int | None,
    package_id: Any,
    page: dict[str, Any] | None = None,
    html_id: str | None = None,
    html_classes: list[str] | None = None,
    index_entry_ids: list[Any] | None = None,
    title_level: int | None = None,
) -> dict[str, Any]:
    page = page or {}
    part = page.get("part") if isinstance(page.get("part"), dict) else {}
    source_record = {
        "record_id": record_id,
        "record_index": record_index,
        "page_id": page.get("id"),
        "page_number": page.get("page"),
        "page_index": page.get("index"),
        "part_id": part.get("id"),
        "part_name": str(part.get("name") or "") or None,
        "html_id": html_id,
        "html_classes": html_classes or [],
        "index_entry_ids": index_entry_ids or [],
        "title_level": title_level,
        "page_field_keys": sorted(k for k in ("seal", "quran", "hadeeth", "shrooh", "rowa", "reciters") if page.get(k)),
        "source_package_id": package_id,
    }
    return {"quran": None, "hadith": None, "openiti": None, "source_record": source_record}


def element(
    *,
    summary: dict[str, Any],
    element_id: str,
    source_record_id: str,
    element_type: str,
    level: int,
    parent_element_id: str | None,
    ancestor_ids: list[str],
    order_in_parent: int,
    section_path: list[str],
    section_path_ids: list[str],
    text_original: str,
    char_span: list[int],
    refs: dict[str, Any],
    flags: list[str],
    parse_confidence: float = 1.0,
) -> dict[str, Any]:
    cleaned, metrics = make_embedding_text(text_original, "ketabonline")
    flags = sorted(set(flags + quality_flags(cleaned, metrics, summary["language"], summary["text_role"])))
    return {
        "schema_version": "dalil.element.v1",
        "element_id": element_id,
        "book_id": summary["book_id"],
        "work_id": summary["work_id"],
        "author_id": summary["author_id"],
        "source_corpus": "ketabonline",
        "file_format": "ketabonline_json",
        "source_path": summary["source_path"],
        "source_record_id": source_record_id,
        "element_type": element_type,
        "level": level,
        "parent_element_id": parent_element_id,
        "ancestor_ids": ancestor_ids,
        "order_in_parent": order_in_parent,
        "section_path": section_path,
        "section_path_ids": section_path_ids,
        "refs": refs,
        "text_original": text_original,
        "text_clean": cleaned,
        "normalized_text": normalize_arabic_light(cleaned, dediac=True),
        "char_span": char_span,
        "token_counts": token_counts(cleaned),
        "parser_profile": "ketabonline_json_parser",
        "parent_policy": "ketabonline_page_parent_v1",
        "view_policy": "ketabonline_structure_view_v1",
        "recommended_detectors": ["ketabonline_html_structure_detector", "ketabonline_index_detector"],
        "parser": {
            "parser_name": "ketabonline_json_parser",
            "parser_version": "v1",
            "parse_confidence": parse_confidence,
            "detectors_used": ["ketabonline_html_structure_detector", "ketabonline_index_detector"],
        },
        "quality_flags": flags,
    }


def build_index_context(data: dict[str, Any], pages: list[dict[str, Any]]) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, list[dict[str, Any]]], list[dict[str, Any]]]:
    page_positions = {p.get("id"): idx for idx, p in enumerate(pages)}
    entries = flatten_index(data.get("index"))
    by_page: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        entry["page_position"] = page_positions.get(entry.get("page_id"), safe_int(entry.get("page")) or 0)
        by_page[entry.get("page_id")].append(entry)
    by_part: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_part[entry.get("part_name") or ""].append(entry)
    return by_page, by_part, entries


def place_entries(entries: list[dict[str, Any]], blocks: list[Block]) -> dict[Any, int]:
    """Offset in the page of each index entry's heading, found by its title.

    The index only gives an entry's page. When several entries share a page, the
    text between two headings belongs to the first, so each entry's heading is
    looked for among the page's blocks, in index order, each after the previous
    one. An entry whose title is not found is placed at the top of the page (-1).
    """
    keys = [title_key_exact(make_embedding_text(b.raw, "ketabonline")[0]).translate(DIGITS) for b in blocks]
    placed: dict[Any, int] = {}
    nxt = 0
    for entry in entries:
        want = title_key_exact(str(entry.get("title") or "")).translate(DIGITS)
        placed[entry.get("id")] = -1
        for j in range(nxt, len(blocks)):
            if want and (keys[j] == want or (len(want) >= MIN_PREFIX_KEY and keys[j].startswith(want))):
                placed[entry.get("id")], nxt = blocks[j].start, j + 1
                break
    return placed


def active_section(
    page: dict[str, Any],
    page_position: int,
    by_part: dict[Any, list[dict[str, Any]]],
    book_title: str,
    placed: dict[Any, int] | None = None,
    offset: int = -1,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """The section in force at `offset` of the page: entries on earlier pages, and
    this page's entries placed at or before `offset` (all of them without `placed`)."""
    part = page.get("part") if isinstance(page.get("part"), dict) else {}
    part_name = str(part.get("name") or "")
    candidates = [e for e in by_part.get(part_name, []) if e.get("page_position", 0) < page_position
                  or (e.get("page_position", 0) == page_position
                      and (placed is None or placed.get(e.get("id"), -1) <= offset))]
    latest: dict[int, dict[str, Any]] = {}
    for entry in candidates:
        level = safe_int(entry.get("title_level")) or 1
        latest[level] = entry
    selected = [latest[k] for k in sorted(latest)]
    section_path = [book_title]
    section_ids = [f"book:{book_title}"]
    if part_name:
        section_path.append(f"الجزء {part_name}")
        section_ids.append(f"part:{part.get('id') or part_name}")
    for entry in selected:
        title = str(entry.get("title") or "").strip()
        if title and title not in section_path:
            section_path.append(title)
            section_ids.append(f"index:{entry.get('id')}")
    return section_path, section_ids, selected


def process_book_first_pass(summary: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    pages = data.get("pages") or []
    by_page, by_part, entries = build_index_context(data, pages)
    raw_chars = 0
    clean_chars = 0
    arabic_letters = 0
    nonspace_chars = 0
    page_flags = Counter()
    tag_counts = Counter()
    block_count = 0
    html_title_count = 0
    paragraph_count = 0
    normalized_parts: list[str] = []
    for page in pages:
        content = str(page.get("content") or "")
        raw_chars += len(content)
        cleaned, metrics = make_embedding_text(content, "ketabonline")
        clean_chars += len(cleaned)
        arabic_letters += int(metrics.get("arabic_letters", 0))
        nonspace_chars += sum(1 for c in cleaned if c.isprintable() and not c.isspace())
        for flag in quality_flags(cleaned, metrics, summary["language"], summary["text_role"]):
            page_flags[flag] += 1
        blocks, tags = extract_blocks(content)
        tag_counts.update(tags)
        block_count += len(blocks)
        html_title_count += sum(b.kind == "html_heading" for b in blocks)
        paragraph_count += sum(b.kind == "paragraph" for b in blocks)
        if cleaned:
            normalized_parts.append(cleaned)
    normalized_text = "\n".join(normalized_parts)
    summary.update(
        {
            "pages_actual": len(pages),
            "parts_actual": len(data.get("parts") or []),
            "index_entries_actual": len(entries),
            "raw_chars": raw_chars,
            "clean_chars": clean_chars,
            "clean_char_ratio": round(clean_chars / max(raw_chars, 1), 4),
            "arabic_letter_ratio_aggregate": round(arabic_letters / max(nonspace_chars, 1), 4),
            "block_count": block_count,
            "paragraph_count": paragraph_count,
            "html_title_count": html_title_count,
            "tag_counts": dict(tag_counts),
            "page_quality_counts": dict(page_flags),
            # SHA-1, despite the key name; existing manifests already use the key.
            "content_sha256": sha1_text(normalized_text),
            "normalized_content_chars": len(normalized_text),
            "index_orphan_count": sum(1 for e in entries if e.get("page_id") not in {p.get("id") for p in pages}),
        }
    )
    return summary


def finalize_book_status(
    summary: dict[str, Any],
    all_summaries: list[dict[str, Any]],
    current_titles: dict[str, list[dict[str, Any]]],
    today_titles: dict[str, list[dict[str, Any]]],
) -> None:
    flags: list[str] = []
    exact_group = [s for s in all_summaries if s.get("content_sha256") == summary.get("content_sha256") and s.get("normalized_content_chars", 0) >= 1000]
    exact_group.sort(key=lambda s: (str(s.get("author_slug")), int(s.get("package_id") or 0)))
    summary["same_content_package_ids"] = [s.get("package_id") for s in exact_group]
    if len(exact_group) > 1 and summary.get("package_id") != exact_group[0].get("package_id"):
        flags.append("exact_duplicate_ketabonline_package")
        summary["status"] = "exclude_exact_duplicate"
        summary["canonical_package_id"] = exact_group[0].get("package_id")
    else:
        summary["status"] = "candidate_new"
        summary["canonical_package_id"] = summary.get("package_id")

    current = []
    for key in {summary.get("title_key_exact"), summary.get("title_key_work")}:
        current.extend(current_titles.get(key, []))
    today = []
    for key in {summary.get("title_key_exact"), summary.get("title_key_work")}:
        today.extend(today_titles.get(key, []))
    summary["current_title_matches"] = unique_rows(current, ("book_id", "title_ar", "source"))
    summary["today_title_matches"] = unique_rows(today, ("title_ar", "source", "path"))
    if summary["current_title_matches"]:
        flags.append("title_or_work_match_current_corpus")
    if summary["today_title_matches"]:
        flags.append("title_or_work_match_today_sources")
    if summary.get("language", "").lower() not in {"ar", "ara", "arabic"}:
        flags.append("exclude_non_arabic_language")
        summary["status"] = "exclude_non_arabic"
    if summary.get("text_role") == "transcription":
        flags.append("transcription_review")
    if summary.get("index_orphan_count", 0):
        flags.append("orphan_index_entries")
    if summary.get("pages_actual") != summary.get("pages_declared") and summary.get("pages_declared") is not None:
        flags.append("declared_page_count_mismatch")
    summary["review_flags"] = sorted(set(flags))
    if summary["status"] == "candidate_new" and flags:
        summary["status"] = "review_required"


def unique_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def emit_elements(
    summary: dict[str, Any],
    data: dict[str, Any],
    output,
    current_block_hashes: set[str],
    today_block_hashes: set[str],
    global_stats: Counter[str],
) -> None:
    pages = data.get("pages") or []
    by_page, by_part, entries = build_index_context(data, pages)
    book_element_id = f"elem:{summary['book_id']}:book"
    title = summary["title_ar"]
    output.write(
        json.dumps(
            element(
                summary=summary,
                element_id=book_element_id,
                source_record_id="book",
                element_type="book",
                level=1,
                parent_element_id=None,
                ancestor_ids=[],
                order_in_parent=0,
                section_path=[title],
                section_path_ids=[f"book:{summary['book_id']}"],
                text_original=title,
                char_span=[0, 0],
                refs=refs_for(record_id="book", record_index=None, package_id=summary["package_id"]),
                flags=list(summary.get("review_flags", [])),
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    global_stats["elements"] += 1
    global_stats["books"] += 1

    part_elements: dict[tuple[str, str], str] = {}
    for part_order, part in enumerate(data.get("parts") or [], start=1):
        part_id = part.get("id") or part.get("name") or part_order
        part_name = str(part.get("name") or part_order)
        part_key = (str(part_id), part_name)
        part_element_id = f"elem:{summary['book_id']}:part:{sha1_text(str(part_key))[:12]}"
        part_elements[part_key] = part_element_id
        part_title = str(part.get("title") or f"الجزء {part_name}").strip()
        row = element(
            summary=summary,
            element_id=part_element_id,
            source_record_id=f"part:{part_id}",
            element_type="chapter",
            level=2,
            parent_element_id=book_element_id,
            ancestor_ids=[book_element_id],
            order_in_parent=part_order,
            section_path=[title, part_title],
            section_path_ids=[f"book:{summary['book_id']}", f"part:{part_id}"],
            text_original=part_title,
            char_span=[0, 0],
            refs=refs_for(record_id=f"part:{part_id}", record_index=part_order, package_id=summary["package_id"], page={"part": part}),
            flags=list(summary.get("review_flags", [])),
        )
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        global_stats["elements"] += 1
        global_stats["chapters"] += 1

    emitted_index: set[Any] = set()
    part_orders: Counter[str] = Counter()
    for page_position, page in enumerate(pages):
        page_id = page.get("id")
        part = page.get("part") if isinstance(page.get("part"), dict) else {}
        part_id = str(part.get("id") or part.get("name") or "")
        part_name = str(part.get("name") or "")
        parent_id = part_elements.get((part_id, part_name), book_element_id)
        ancestors = [book_element_id] if parent_id == book_element_id else [book_element_id, parent_id]
        page_entries = [e for e in by_page.get(page_id, []) if e.get("id") not in emitted_index]
        blocks, tags = extract_blocks(str(page.get("content") or ""))
        placed = place_entries(page_entries, blocks) if page_entries else {}
        # Index headings and blocks in page order; a heading goes before the block it heads.
        events = sorted([(placed[e.get("id")], 0, n, e) for n, e in enumerate(page_entries)]
                        + [(b.start, 1 if b.kind == "html_heading" else 2, b.end, b) for b in blocks],
                        key=lambda ev: ev[:3])
        section_path, section_ids, selected = active_section(page, page_position, by_part, title, placed)
        selected_ids = [e.get("id") for e in selected if e.get("id") is not None]
        titles = [b for b in blocks if b.kind == "html_heading"]
        index_title_keys = {title_key_exact(str(e.get("title") or "")) for e in page_entries if e.get("title")}
        for offset, kind, _, item in events:
            if kind == 0:
                entry = item
                entry_id = entry.get("id")
                emitted_index.add(entry_id)
                section_path, section_ids, selected = active_section(page, page_position, by_part, title, placed, offset)
                selected_ids = [e.get("id") for e in selected if e.get("id") is not None]
                heading_path = list(section_path)
                heading_ids = list(section_ids)
                entry_title = str(entry.get("title") or "").strip()
                if entry_title and entry_title not in heading_path:
                    heading_path.append(entry_title)
                    heading_ids.append(f"index:{entry_id}")
                part_orders[part_id] += 1
                heading_row = element(
                    summary=summary,
                    element_id=f"elem:{summary['book_id']}:index:{entry_id}",
                    source_record_id=f"index:{entry_id}",
                    element_type="heading",
                    level=max(2, min(5, 1 + (safe_int(entry.get("title_level")) or 1))),
                    parent_element_id=parent_id,
                    ancestor_ids=ancestors,
                    order_in_parent=part_orders[part_id],
                    section_path=heading_path,
                    section_path_ids=heading_ids,
                    text_original=entry_title,
                    char_span=[0, 0],
                    refs=refs_for(record_id=f"index:{entry_id}", record_index=entry.get("order"), package_id=summary["package_id"], page=page, index_entry_ids=[entry_id], title_level=safe_int(entry.get("title_level"))),
                    flags=list(summary.get("review_flags", [])),
                    parse_confidence=0.98,
                )
                output.write(json.dumps(heading_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                global_stats["elements"] += 1
                global_stats["headings"] += 1
                continue

            block = item
            if block.kind == "html_heading":
                cleaned_title, _ = make_embedding_text(block.raw, "ketabonline")
                if title_key_exact(cleaned_title) in index_title_keys:
                    global_stats["html_headings_reconciled_to_index"] += 1
                    continue
                element_type = "heading"
                text_original = block.raw
                block_flag = ["html_title_waypoint"]
                block_id_part = block.attrs.get("id") or f"order-{block.order}"
            else:
                body_raw = remove_nested_titles(block, titles, str(page.get("content") or ""))
                if not body_raw.strip():
                    global_stats["empty_blocks_skipped"] += 1
                    continue
                element_type = "paragraph"
                text_original = body_raw
                block_flag = []
                block_id_part = block.attrs.get("id") or f"order-{block.order}"

            part_orders[part_id] += 1
            record_id = f"page:{page_id}:{'title' if element_type == 'heading' else 'p'}:{block_id_part}"
            eid = f"elem:{summary['book_id']}:page:{page_id}:{'title' if element_type == 'heading' else 'p'}:{sha1_text(record_id + text_original)[:12]}"
            row = element(
                summary=summary,
                element_id=eid,
                source_record_id=record_id,
                element_type=element_type,
                level=3,
                parent_element_id=parent_id,
                ancestor_ids=ancestors,
                order_in_parent=part_orders[part_id],
                section_path=section_path,
                section_path_ids=section_ids,
                text_original=text_original,
                char_span=[block.start, block.end],
                refs=refs_for(record_id=record_id, record_index=page_position, package_id=summary["package_id"], page=page, html_id=block.attrs.get("id"), html_classes=block.classes, index_entry_ids=selected_ids),
                flags=block_flag + list(summary.get("review_flags", [])),
                parse_confidence=0.95 if blocks else 0.75,
            )
            if not row["text_clean"].strip():
                global_stats["empty_blocks_skipped"] += 1
                continue
            normalized = row["normalized_text"]
            if len(normalized) >= 80:
                if sha1_text(normalized) in current_block_hashes:
                    summary["exact_current_element_matches"] = summary.get("exact_current_element_matches", 0) + 1
                    row["quality_flags"] = sorted(set(row["quality_flags"] + ["exact_element_match_current_corpus"]))
                if sha1_text(normalized) in today_block_hashes:
                    summary["exact_today_element_matches"] = summary.get("exact_today_element_matches", 0) + 1
                    row["quality_flags"] = sorted(set(row["quality_flags"] + ["exact_element_match_today_sources"]))
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            global_stats["elements"] += 1
            global_stats["headings" if element_type == "heading" else "paragraphs"] += 1

    # Preserve native index entries that point to a page omitted by the package.
    for entry in entries:
        if entry.get("id") in emitted_index:
            continue
        entry_id = entry.get("id")
        row = element(
            summary=summary,
            element_id=f"elem:{summary['book_id']}:index:{entry_id}:orphan",
            source_record_id=f"index:{entry_id}:orphan",
            element_type="heading",
            level=max(2, min(5, 1 + (safe_int(entry.get("title_level")) or 1))),
            parent_element_id=book_element_id,
            ancestor_ids=[book_element_id],
            order_in_parent=0,
            section_path=[title, str(entry.get("title") or "")],
            section_path_ids=[f"book:{summary['book_id']}", f"index:{entry_id}"],
            text_original=str(entry.get("title") or ""),
            char_span=[0, 0],
            refs=refs_for(record_id=f"index:{entry_id}:orphan", record_index=entry.get("order"), package_id=summary["package_id"], index_entry_ids=[entry_id], title_level=safe_int(entry.get("title_level"))),
            flags=list(summary.get("review_flags", [])) + ["orphan_index_entry"],
            parse_confidence=0.8,
        )
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        global_stats["elements"] += 1
        global_stats["headings"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output directory; a relative path is taken under the data dir "
             "(default: %(default)s). Not structured/ketabonline: that is the "
             "deduplicated handoff the extractor reads, and this would overwrite it",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N packages for a smoke test")
    parser.add_argument("--raw", type=Path, default=KETAB_ROOT,
                        help="prepare every ZIP package under this directory (default: %(default)s)")
    parser.add_argument("--only", action="append", default=[], metavar="PACKAGE_ID",
                        help="prepare only these packages under --raw (repeatable)")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="the packages' catalog (default: %(default)s)")
    parser.add_argument("--download-manifest", type=Path, default=DOWNLOAD_MANIFEST_PATH,
                        help="the packages' download records (default: %(default)s)")
    parser.add_argument("--author", action="append", default=[], metavar="SLUG=AUTHOR_ID",
                        help="author id of a package folder the built-in table lacks (repeatable)")
    parser.add_argument("--no-db-overlap", action="store_true",
                        help="skip the title and block overlap labels against the search DB")
    parser.add_argument("--no-collection-overlap", action="store_true",
                        help="skip the title and block overlap labels against the collection's other sources")
    args = parser.parse_args(argv)
    output_root: Path = args.output if args.output.is_absolute() else DATA_DIR / args.output
    output_root.mkdir(parents=True, exist_ok=True)
    for spec in args.author:
        slug, author_id = spec.split("=", 1)
        AUTHOR_IDS[slug] = author_id
    raw_root: Path = args.raw.resolve()

    catalog_by_id, download_by_id = load_catalog(args.catalog, args.download_manifest)
    paths = package_paths(raw_root)
    if args.only:
        paths = [p for p in paths if p.name.split(".")[0] in set(args.only)]
    if args.limit is not None:
        paths = paths[: max(args.limit, 0)]
    if not paths:
        raise SystemExit(f"No Ketabonline ZIP packages found under {raw_root}")

    print(f"Reading {len(paths)} packages for summaries...", flush=True)
    summaries: list[dict[str, Any]] = []
    for idx, zip_path in enumerate(paths, start=1):
        data = read_package(zip_path)
        package_id = str(data.get("id") or zip_path.stem.split(".")[0])
        summary = make_book_summary(zip_path, data, catalog_by_id.get(package_id, {}), download_by_id.get(package_id, {}))
        summaries.append(process_book_first_pass(summary, data))
        if idx % 25 == 0 or idx == len(paths):
            print(f"  summarized {idx}/{len(paths)}", flush=True)

    current_titles, current_block_hashes = (defaultdict(list), set()) if args.no_db_overlap else load_current_title_catalog()
    today_titles, today_block_hashes = (defaultdict(list), set()) if args.no_collection_overlap else load_today_title_catalog()
    for summary in summaries:
        finalize_book_status(summary, summaries, current_titles, today_titles)
        if summary.get("exact_current_element_matches", 0) >= 3:
            summary.setdefault("review_flags", []).append("content_overlap_current_review")
        if summary.get("exact_today_element_matches", 0) >= 3:
            summary.setdefault("review_flags", []).append("content_overlap_today_review")
        summary["review_flags"] = sorted(set(summary.get("review_flags", [])))

    manifest_path = output_root / "books_manifest.jsonl"
    dedup_path = output_root / "dedup_report.jsonl"
    jsonl_write(manifest_path, summaries)
    jsonl_write(dedup_path, summaries)

    element_path = output_root / "elements.jsonl"
    global_stats: Counter[str] = Counter()
    print("Emitting structure-preserving elements...", flush=True)
    with element_path.open("w", encoding="utf-8") as output:
        for idx, (summary, zip_path) in enumerate(zip(summaries, paths), start=1):
            data = read_package(zip_path)
            emit_elements(summary, data, output, current_block_hashes, today_block_hashes, global_stats)
            if idx % 10 == 0 or idx == len(paths):
                print(f"  emitted {idx}/{len(paths)} packages ({global_stats['elements']} elements)", flush=True)

    # Rewrite manifests with block-level overlap counts collected during emission.
    jsonl_write(manifest_path, summaries)
    jsonl_write(dedup_path, summaries)

    try:
        output_root_label = str(output_root.relative_to(DATA_DIR))
    except ValueError:
        output_root_label = str(output_root)
    summary_report = {
        "schema_version": "dalil.ketabonline_preparation.v1",
        "source_root": str(raw_root.relative_to(DATA_DIR)) if raw_root.is_relative_to(DATA_DIR) else str(raw_root),
        "output_root": output_root_label,
        "packages": len(summaries),
        "package_status_counts": dict(Counter(s.get("status") for s in summaries)),
        "languages": dict(Counter(s.get("language") for s in summaries)),
        "authors": dict(Counter(s.get("author_slug") for s in summaries)),
        "categories": dict(Counter(s.get("category") for s in summaries)),
        "text_roles": dict(Counter(s.get("text_role") for s in summaries)),
        "pages": sum(int(s.get("pages_actual") or 0) for s in summaries),
        "declared_pages": sum(int(s.get("pages_declared") or 0) for s in summaries if str(s.get("pages_declared") or "").isdigit()),
        "index_entries": sum(int(s.get("index_entries_actual") or 0) for s in summaries),
        "raw_chars": sum(int(s.get("raw_chars") or 0) for s in summaries),
        "clean_chars": sum(int(s.get("clean_chars") or 0) for s in summaries),
        "elements": dict(global_stats),
        "flags": dict(Counter(flag for s in summaries for flag in s.get("review_flags", []))),
        "page_quality_flags": dict(Counter(flag for s in summaries for flag, n in (s.get("page_quality_counts") or {}).items() for _ in range(int(n)))),
        "same_content_groups": [
            {"content_sha256": digest, "package_ids": [s.get("package_id") for s in group]}
            for digest, group in group_by(summaries, "content_sha256").items()
            if len(group) > 1 and sum(int(s.get("normalized_content_chars") or 0) for s in group) >= 2000
        ],
        "current_title_key_count": len(current_titles),
        "today_title_key_count": len(today_titles),
        "current_block_hash_count": len(current_block_hashes),
        "today_block_hash_count": len(today_block_hashes),
        "chunking_performed": False,
        "embedding_performed": False,
        "faiss_touched": False,
        "sqlite_touched": False,
    }
    json_dump(output_root / "qa_report.json", summary_report)

    readme = """# Ketabonline structured preparation — 2026-08-15

This directory is the output of `dalil.s2_prepare.ketabonline`.

It contains:

- `books_manifest.jsonl`: one labelled record per downloaded package.
- `elements.jsonl`: book, part, native-index heading, HTML-title, and paragraph elements.
- `dedup_report.jsonl`: conservative title/content overlap decisions.
- `qa_report.json`: aggregate counts and quality flags.

The raw ZIP packages remain authoritative under `data/books/original/ketabonline/`. Each
element keeps original HTML in `text_original`, cleaned retrieval text in
`text_clean`, normalized text, page/part/index provenance, and quality flags.

This stage performs no parent construction, fixed-size chunking, embedding,
FAISS writes, or SQLite writes. `exclude_*` and `review_*` statuses are labels
for the next review stage, not deletion of raw evidence.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def group_by(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "")].append(row)
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
