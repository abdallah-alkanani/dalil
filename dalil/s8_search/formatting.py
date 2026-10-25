"""Plain-text rendering of search results for the CLI."""
from __future__ import annotations

from typing import Any


def format_results(output: dict[str, Any], show_scores: bool = False, group_by_tier: bool = False) -> str:
    results = output.get("results", [])
    if not results:
        lines = ["No results found."]
        if output.get("error"):
            lines.append(f"Error: {output['error']}")
        return "\n".join(lines)

    if group_by_tier:
        return format_results_grouped(results, show_scores)

    lines = []
    for i, result in enumerate(results, 1):
        lines.extend(format_result_block(result, i, show_scores))
    return "\n".join(lines)


def format_results_grouped(results: list[dict[str, Any]], show_scores: bool) -> str:
    labels = {
        "QURAN": "━━━ القرآن الكريم ━━━",
        "HADITH": "━━━ الحديث الشريف ━━━",
        "TAFSIR": "━━━ التفسير ━━━",
        "FIQH": "━━━ الفقه ━━━",
        "USUL AL-FIQH": "━━━ أصول الفقه ━━━",
        "AQEEDAH": "━━━ العقيدة ━━━",
        "SEERAH": "━━━ السيرة النبوية ━━━",
        "OTHER": "━━━ مصادر أخرى ━━━",
    }
    tier_order = ["QURAN", "HADITH", "TAFSIR", "FIQH", "USUL AL-FIQH", "AQEEDAH", "SEERAH", "OTHER"]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for rank, result in enumerate(results, 1):
        grouped.setdefault(result.get("tier_tag", "OTHER"), []).append((rank, result))

    lines = []
    for tag in tier_order:
        group = grouped.get(tag)
        if not group:
            continue
        lines.append(labels.get(tag, f"━━━ {tag} ━━━"))
        lines.append("")
        for original_rank, result in group:
            lines.extend(format_result_block(result, original_rank, show_scores))
        lines.append("")
    return "\n".join(lines)


def format_result_block(result: dict[str, Any], rank: int, show_scores: bool) -> list[str]:
    citation = result.get("citation") or result.get("book_label") or result.get("parent_id") or "Unknown"
    header = f"[{rank}] {result.get('tier_label', 'N/A')} | {citation}"
    if show_scores:
        parts = []
        if result.get("semantic_sim") is not None:
            parts.append(f"Sem:{float(result['semantic_sim']):.4f}")
        if result.get("hybrid_score") is not None:
            parts.append(f"Hybrid:{float(result['hybrid_score']):.4f}")
        if result.get("bm25_rank") is not None:
            parts.append(f"BM25:{int(result['bm25_rank'])}")
        if result.get("dense_rank") is not None:
            parts.append(f"Dense:{int(result['dense_rank'])}")
        if result.get("weighted_score") is not None:
            parts.append(f"W:{float(result['weighted_score']):.4f}")
        if result.get("mmr_rank") is not None:
            parts.append(f"MMR:{int(result['mmr_rank']) + 1}")
        if result.get("listwise_rank") is not None:
            parts.append(f"LW:{int(result['listwise_rank']) + 1}")
        if result.get("n_child_hits", 1) > 1:
            parts.append(f"children:{result['n_child_hits']}")
        if result.get("faiss_row") is not None:
            parts.append(f"row:{result['faiss_row']}")
        if parts:
            header += f" [{' | '.join(parts)}]"

    lines = [header]
    section = result.get("section_path") or []
    if section:
        lines.append(f"    Section: {' > '.join(str(x) for x in section)}")

    meta_bits = []
    if result.get("parent_id"):
        meta_bits.append(f"Parent: {result['parent_id']}")
    if result.get("view_type"):
        meta_bits.append(f"View: {result['view_type']}")
    if meta_bits:
        lines.append("    " + " | ".join(meta_bits))

    refs_line = format_refs(result.get("refs") or {})
    if refs_line:
        lines.append(f"    Refs: {refs_line}")

    text = result.get("text_for_synthesis") or result.get("text_original") or ""
    display = text if result.get("text_for_synthesis_truncated") else text[:2000]
    if not result.get("text_for_synthesis_truncated") and len(text) > 2000:
        display += "..."
    lines.append(display)
    lines.append("")
    return lines


def format_refs(refs: dict[str, Any]) -> str:
    parts = []
    hadith = refs.get("hadith")
    if hadith:
        parts.append(
            f"hadith collection={hadith.get('collection', '?')} "
            f"#{hadith.get('hadith_number', '?')}"
        )
    quran = refs.get("quran")
    if quran:
        surah = quran.get("surah", "?")
        ayah = quran.get("ayah_start") or quran.get("ayah") or "?"
        parts.append(f"quran surah={surah} ayah={ayah}")
    return "; ".join(parts)
