"""Metadata filters: CLI/dict values -> SQL WHERE clauses for each store."""
from __future__ import annotations

from typing import Any


def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            vals = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    vals.extend(part.strip() for part in item.split(",") if part.strip())
                else:
                    vals.append(item)
            if vals:
                clean[key] = sorted(set(vals)) if all(isinstance(v, str) for v in vals) else vals
        else:
            clean[key] = value
    return clean


def format_filter_summary(filters: dict[str, Any]) -> str:
    parts = []
    for key in sorted(filters):
        value = filters[key]
        if isinstance(value, list) and len(value) > 4:
            rendered = ", ".join(str(v) for v in value[:4]) + f", +{len(value) - 4}"
        elif isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def add_in_filter(clauses: list[str], params: list[Any], column: str, values: list[Any] | None) -> None:
    if not values:
        return
    placeholders = ",".join("?" * len(values))
    clauses.append(f"{column} IN ({placeholders})")
    params.extend(values)


def add_like_filter(clauses: list[str], params: list[Any], column: str, values: list[str] | None) -> None:
    if not values:
        return
    sub = []
    for value in values:
        sub.append(f"{column} LIKE ?")
        params.append(f"%{value}%")
    clauses.append("(" + " OR ".join(sub) + ")")


def add_json_in_filter(
    clauses: list[str],
    params: list[Any],
    json_column: str,
    path: str,
    values: list[Any] | None,
) -> None:
    if not values:
        return
    placeholders = ",".join("?" * len(values))
    clauses.append(f"json_extract({json_column}, ?) IN ({placeholders})")
    params.append(path)
    params.extend(values)


def add_json_like_filter(
    clauses: list[str],
    params: list[Any],
    json_column: str,
    values: list[str] | None,
) -> None:
    if not values:
        return
    sub = []
    for value in values:
        sub.append(f"{json_column} LIKE ?")
        params.append(f"%{value}%")
    clauses.append("(" + " OR ".join(sub) + ")")


def add_numeric_range(
    clauses: list[str],
    params: list[Any],
    column: str,
    min_value: int | None,
    max_value: int | None,
) -> None:
    if min_value is not None:
        clauses.append(f"{column} >= ?")
        params.append(min_value)
    if max_value is not None:
        clauses.append(f"{column} <= ?")
        params.append(max_value)


def build_child_filter_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    add_in_filter(clauses, params, "category", filters.get("category"))
    add_in_filter(clauses, params, "subcategory", filters.get("subcategory"))
    add_in_filter(clauses, params, "book_id", filters.get("book_id"))
    add_in_filter(clauses, params, "parent_id", filters.get("parent_id"))
    add_in_filter(clauses, params, "view_id", filters.get("view_id"))
    add_in_filter(clauses, params, "source", filters.get("source"))
    add_in_filter(clauses, params, "view_type", filters.get("view_type"))
    add_in_filter(clauses, params, "text_role", filters.get("text_role"))
    add_numeric_range(clauses, params, "qwen_tokens", filters.get("min_tokens"), filters.get("max_tokens"))
    add_numeric_range(
        clauses,
        params,
        "source_line",
        filters.get("min_source_line"),
        filters.get("max_source_line"),
    )

    metadata_paths = {
        "source_corpus": "$.source_corpus",
        "work_id": "$.work_id",
        "author_id": "$.author_id",
        "file_format": "$.file_format",
        "parser_profile": "$.parser_profile",
    }
    for key, path in metadata_paths.items():
        add_json_in_filter(clauses, params, "metadata_json", path, filters.get(key))

    refs_paths = {
        "quran_surah": "$.quran.surah",
        "hadith_collection": "$.hadith.collection",
        "hadith_number": "$.hadith.hadith_number",
        "hadith_book_number": "$.hadith.book_number",
        "hadith_chapter_id": "$.hadith.chapter_id",
        "source_record_id": "$.source_record.record_id",
    }
    for key, path in refs_paths.items():
        add_json_in_filter(clauses, params, "refs_json", path, filters.get(key))

    quran_ayah = filters.get("quran_ayah")
    if quran_ayah is not None:
        clauses.append(
            "CAST(json_extract(refs_json, '$.quran.ayah_start') AS INTEGER) <= ? "
            "AND CAST(json_extract(refs_json, '$.quran.ayah_end') AS INTEGER) >= ?"
        )
        params.extend([quran_ayah, quran_ayah])

    quality_flags = filters.get("quality_flag")
    if quality_flags:
        placeholders = ",".join("?" * len(quality_flags))
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(child_metadata.metadata_json, '$.quality_flags') "
            f"WHERE json_each.value IN ({placeholders}))"
        )
        params.extend(quality_flags)

    section_ids = filters.get("section_id")
    if section_ids:
        placeholders = ",".join("?" * len(section_ids))
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(child_metadata.section_path_ids_json) "
            f"WHERE json_each.value IN ({placeholders}))"
        )
        params.extend(section_ids)
    add_json_like_filter(clauses, params, "section_path_json", filters.get("section_contains"))
    add_like_filter(clauses, params, "book_id", filters.get("title"))

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(f"({clause})" for clause in clauses), params


def build_parent_filter_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    add_in_filter(clauses, params, "parent_type", filters.get("parent_type"))
    add_like_filter(clauses, params, "citation", filters.get("citation_contains"))
    add_json_in_filter(clauses, params, "metadata_json", "$.parent_policy", filters.get("parent_policy"))
    add_json_in_filter(clauses, params, "metadata_json", "$.view_policy", filters.get("view_policy"))
    add_like_filter(clauses, params, "book_id", filters.get("title"))
    add_json_like_filter(clauses, params, "section_path_json", filters.get("section_contains"))
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(f"({clause})" for clause in clauses), params


def build_bm25_where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key in [
        "category",
        "subcategory",
        "book_id",
        "parent_id",
        "source",
        "view_type",
        "text_role",
        "source_corpus",
        "work_id",
        "author_id",
        "file_format",
        "parser_profile",
        "quran_surah",
        "hadith_collection",
        "hadith_number",
        "hadith_book_number",
        "hadith_chapter_id",
        "source_record_id",
    ]:
        add_in_filter(clauses, params, key, filters.get(key))
    quran_ayah = filters.get("quran_ayah")
    if quran_ayah is not None:
        clauses.append("CAST(quran_ayah_start AS INTEGER) <= ? AND CAST(quran_ayah_end AS INTEGER) >= ?")
        params.extend([quran_ayah, quran_ayah])
    if not clauses:
        return "", []
    return " AND " + " AND ".join(f"({clause})" for clause in clauses), params

