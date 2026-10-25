"""Rank-level helpers: RRF fusion, MMR diversity, listwise parsing, lexical query
sanitising, and sentence-snapped parent windows."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from dalil.arabic.normalize import normalize_for_search


def fuse_parent_hits_rrf(
    dense_hits: dict[str, dict[str, Any]],
    bm25_hits: dict[str, dict[str, Any]],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> dict[str, dict[str, Any]]:
    """Weighted reciprocal-rank fusion of dense and lexical parent hits."""
    fused: dict[str, dict[str, Any]] = {}
    rrf_k = max(1, int(rrf_k))

    dense_ranked = sorted(
        dense_hits.items(),
        key=lambda item: (float(item[1].get("dense_score", item[1].get("best_score", 0.0))), item[0]),
        reverse=True,
    )
    for rank, (pid, hit) in enumerate(dense_ranked, 1):
        rec = hit.copy()
        rec["child_scores"] = list(hit.get("child_scores", []))
        rec["child_rows"] = list(hit.get("child_rows", []))
        rec["dense_rank"] = rank
        rec["retrieval_sources"] = ["dense"]
        rec["hybrid_score"] = dense_weight / (rrf_k + rank)
        fused[pid] = rec

    bm25_ranked = sorted(
        bm25_hits.items(),
        key=lambda item: (int(item[1].get("bm25_rank", 10**9)), item[0]),
    )
    for rank, (pid, hit) in enumerate(bm25_ranked, 1):
        add_score = bm25_weight / (rrf_k + rank)
        if pid not in fused:
            rec = hit.copy()
            rec["child_scores"] = list(hit.get("child_scores", []))
            rec["child_rows"] = list(hit.get("child_rows", []))
            rec["bm25_rank"] = rank
            rec["hybrid_score"] = add_score
            rec["retrieval_sources"] = ["bm25"]
            fused[pid] = rec
            continue
        rec = fused[pid]
        rec["hybrid_score"] = float(rec.get("hybrid_score", 0.0)) + add_score
        rec["bm25_rank"] = rank
        rec["bm25_score"] = hit.get("bm25_score")
        rec["retrieval_sources"] = sorted(set(rec.get("retrieval_sources", [])) | {"bm25"})
        rec["child_scores"].extend(hit.get("child_scores", []))
        rec["child_rows"].extend(hit.get("child_rows", []))
        if "dense_score" not in rec and "dense_score" in hit:
            rec["dense_score"] = hit["dense_score"]
        if hit.get("bm25_score") is not None and rec.get("dense_rank") is None:
            rec["best_score"] = hit.get("best_score", rec.get("best_score", 0.0))
            rec["best_faiss_row"] = hit["best_faiss_row"]
            rec["child"] = hit["child"]

    for rec in fused.values():
        rec["best_score"] = float(rec.get("dense_score", rec.get("best_score", 0.0)))
    return dict(
        sorted(
            fused.items(),
            key=lambda item: float(item[1].get("hybrid_score", item[1].get("best_score", 0.0))),
            reverse=True,
        )
    )


def mmr_select(
    doc_vecs: dict[int, np.ndarray],
    rel_scores: dict[int, float],
    candidate_rows: list[int],
    lam: float,
    k: int,
) -> list[tuple[int, float]]:
    if not candidate_rows:
        return []
    lam = max(0.0, min(1.0, lam))
    selected: list[int] = []
    selected_vecs: list[np.ndarray] = []
    remaining = set(candidate_rows)

    for _ in range(min(k, len(candidate_rows))):
        best_row = None
        best_mmr = -float("inf")
        for row in remaining:
            rel = rel_scores[row]
            if selected_vecs:
                div = float(np.max(np.asarray(selected_vecs) @ doc_vecs[row]))
            else:
                div = 0.0
            mmr_score = lam * rel - (1.0 - lam) * div
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        selected_vecs.append(doc_vecs[best_row])
        remaining.remove(best_row)

    return [(row, rel_scores[row]) for row in selected]


def parse_listwise_ranking(response: str, n_passages: int) -> list[int]:
    if not response:
        return list(range(n_passages))
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    ranking = []
    seen = set()
    for number in re.findall(r"\d+", response):
        idx = int(number) - 1
        if 0 <= idx < n_passages and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    ranking.extend(i for i in range(n_passages) if i not in seen)
    return ranking


def sanitize_fts_query(text: str, max_terms: int = 32) -> str:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text, flags=re.UNICODE)
    seen = set()
    clean = []
    for token in tokens:
        token = token.strip("_").lower()
        if not token or token in seen:
            continue
        seen.add(token)
        clean.append(token)
        if len(clean) >= max_terms:
            break
    return " OR ".join(clean)


# Sentence ends, in the punctuation this corpus actually uses. Arabic full stop is
# the ASCII '.', but the question mark and semicolon are the Arabic codepoints.
_SENTENCE_END_RE = re.compile(r"[.؟!؛?]+[\s\u00a0]|\n\s*\n")


def snap_window(text: str, start: int, end: int, slack: int) -> tuple[int, int]:
    """Move a character window out to the nearest sentence boundaries.

    The window is chosen by character count, which lands mid-word as often as not:
    the parent handed to the answering model then opens halfway through a sentence
    and stops halfway through another. That is the same damage the whole boundary
    project exists to prevent, applied at query time instead of index time, and it
    is worse here because it happens on every hit rather than once per parent.

    Both edges move outward when a boundary is within `slack`, so the excerpt grows
    slightly rather than dropping text that was inside the window. If no boundary is
    near, the original offset stands — a hard cut is still better than returning a
    window so wide it defeats the cap.
    """
    if start > 0:
        head = text.rfind("\n\n", max(0, start - slack), start)
        if head < 0:
            matches = list(_SENTENCE_END_RE.finditer(text, max(0, start - slack), start))
            head = matches[-1].end() if matches else -1
        if head >= 0:
            start = head
    if end < len(text):
        tail = _SENTENCE_END_RE.search(text, end, min(len(text), end + slack))
        end = tail.end() if tail else end
    return start, min(end, len(text))


def window_parent_text(result: dict[str, Any], anchors: list[str], max_chars: int) -> None:
    """Cap ``text_for_synthesis`` at about ``max_chars``, centred on the match.

    The window opens a third of the budget before the first anchor (the matched
    child text) found in the parent, then snaps both edges to sentence
    boundaries. Records the original length and window in the result.
    """
    text = result.get("text_for_synthesis") or ""
    if len(text) <= max_chars:
        return
    anchor = next(
        (position for item in anchors if (position := text.find(item[:120])) >= 0),
        -1,
    )
    start = 0 if anchor < 0 else max(0, anchor - max_chars // 3)
    end = start + max_chars
    start, end = snap_window(text, start, end, max_chars // 10)
    excerpt = text[start:end]
    result["text_for_synthesis"] = (
        ("…" if start > 0 else "") + excerpt + ("…" if end < len(text) else "")
    )
    result["text_for_synthesis_truncated"] = True
    result["text_for_synthesis_full_chars"] = len(text)
    result["text_for_synthesis_window"] = [start, min(end, len(text))]


def normalize_arabic_for_search(text: str) -> str:
    """Query-side search form; identical to what the BM25 index was built with.

    ``clean_markers`` is off: source debris (html tags, AUTO/CHECK, page
    milestones) is stripped from corpus text at index time and never occurs in a
    user's question.
    """
    return normalize_for_search(text, clean_markers=False)
