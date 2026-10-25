"""Verify every retrieval view really lives inside the parent it points at.

Splitting a parent rewrites both sides of the relationship: the parent's text is
divided and its views are re-emitted against the pieces. If a view ends up attached
to the wrong piece, nothing crashes and nothing looks wrong — the corpus validates,
the vectors build, and retrieval quietly returns a parent that does not contain the
text that matched. That failure is invisible until someone reads an answer and finds
the cited passage missing from its own context.

So this checks the containment directly, on the text rather than on the ids:

  1. every view's `parent_id` exists
  2. every parent's `child_view_ids` all exist
  3. the two agree — no view claims a parent that disowns it
  4. **the view's text actually occurs inside its parent's text**
  5. a parent is at least as long as the sum of its children

(4) is the one that catches a bad split. Comparison is on whitespace-normalised
text, since the view layer re-joins lines; and on `normalize_for_search` as a
fallback, because a parent may carry tashkeel where a view does not.

Exit code is non-zero when anything fails, so this can gate the pipeline.

    python -m dalil.s5_audit.parent_child \\
        --parents data/build/corpora/collection/parents.jsonl \\
        --views data/build/corpora/collection/retrieval_views.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from dalil.arabic.normalize import normalize_for_search


def norm(text: str) -> str:
    return " ".join((text or "").split())


def squeeze(text: str) -> str:
    """Whitespace-free form, for containment only.

    A view is not a byte-slice of its parent. `split_paragraph_packs` breaks a unit
    after sentence punctuation, so a view can carry a space the parent does not:
    `"...قال.وقال..."` in the parent becomes `"...قال. وقال..."` in the view. Measured
    on pass-1 output, that accounts for 2,992 of 12,201 paragraph views — all of them
    genuinely present, none of them matching byte for byte. Comparing on the squeezed
    form removes the false alarm without weakening the check: the characters still
    have to be there, in order.
    """
    return re.sub(r"\s+", "", text or "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--max-report", type=int, default=15)
    ap.add_argument("--sample", type=int, default=0,
                    help="only check this many views (0 = all); text containment is "
                         "the expensive part on a 3 GB view file")
    args = ap.parse_args(argv)

    parents: dict[str, dict] = {}
    for line in args.parents.open(encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        body = p.get("text_for_synthesis") or p.get("text_original") or ""
        parents[p["parent_id"]] = {
            "text": norm(body),
            "children": list(p.get("child_view_ids") or []),
            # Compared on the whitespace-free form, for the same reason `squeeze`
            # exists: `split_paragraph_packs` breaks a unit after sentence
            # punctuation, so a view can carry a space its parent does not. On word
            # counts that space splits one token into two and the children appear to
            # outweigh the parent by a few words — 5,134 parents on pass-3 output,
            # every one of them a miscount. Not one of them exceeds its parent on
            # characters, which is the thing that actually has to hold.
            "chars": len(squeeze(body)),
        }

    counts = collections.Counter()
    bad_missing_parent, bad_not_contained, bad_disowned = [], [], []
    child_chars: dict[str, int] = collections.defaultdict(int)
    # A hadith's full views (text_role "full") repeat its sanad and matn views, so
    # where a parent has both, only the sanad and matn views count toward (5).
    full_chars: dict[str, int] = collections.defaultdict(int)
    has_parts: set[str] = set()
    seen_views: set[str] = set()

    for line in args.views.open(encoding="utf-8"):
        if not line.strip():
            continue
        if args.sample and counts["views"] >= args.sample:
            break
        v = json.loads(line)
        counts["views"] += 1
        vid, pid = v["view_id"], v.get("parent_id")
        seen_views.add(vid)
        parent = parents.get(pid)
        if parent is None:
            counts["missing_parent"] += 1
            bad_missing_parent.append((vid, pid))
            continue
        if vid not in parent["children"]:
            counts["disowned"] += 1
            bad_disowned.append((vid, pid))
        text = norm(v.get("display_text") or "")
        # A heading view holds the section title, which is deliberately not part of
        # the parent's body text. Containment does not apply to it.
        if (v.get("text_role") == "heading" or v.get("view_type") == "heading_title"):
            counts["heading_view_skipped"] += 1
            continue
        if v.get("text_role") == "full":
            full_chars[pid] += len(squeeze(text))
        else:
            child_chars[pid] += len(squeeze(text))
            if v.get("text_role") in ("sanad", "matn"):
                has_parts.add(pid)
        if not text:
            counts["empty_view"] += 1
            continue
        if text in parent["text"]:
            counts["contained"] += 1
        elif squeeze(text) in squeeze(parent["text"]):
            counts["contained_ignoring_space"] += 1
        elif normalize_for_search(text) in normalize_for_search(parent["text"]):
            counts["contained_normalised"] += 1
        else:
            counts["NOT_CONTAINED"] += 1
            bad_not_contained.append((vid, pid, text[:70]))

    for pid, chars in full_chars.items():
        if pid not in has_parts:
            child_chars[pid] += chars
    oversized_children = [
        (pid, child_chars[pid], parents[pid]["chars"])
        for pid in child_chars
        if child_chars[pid] > parents[pid]["chars"]
    ]
    dangling = [(pid, c) for pid, p in parents.items() for c in p["children"]
                if c not in seen_views] if not args.sample else []

    print(f"parents: {len(parents):,}   views checked: {counts['views']:,}")
    print(f"  contained verbatim        : {counts['contained']:,}")
    print(f"  contained ignoring spaces : {counts['contained_ignoring_space']:,}")
    print(f"  contained after normalise : {counts['contained_normalised']:,}")
    print(f"  heading views (not applicable): {counts['heading_view_skipped']:,}")
    print(f"  NOT CONTAINED             : {counts['NOT_CONTAINED']:,}")
    print(f"  view points at missing parent: {counts['missing_parent']:,}")
    print(f"  parent disowns its view      : {counts['disowned']:,}")
    print(f"  empty view text              : {counts['empty_view']:,}")
    print(f"  children longer than parent  : {len(oversized_children):,}  (chars)")
    print(f"  child_view_ids with no view  : {len(dangling):,}")

    for label, rows in (("NOT CONTAINED", bad_not_contained),
                        ("MISSING PARENT", bad_missing_parent),
                        ("DISOWNED", bad_disowned),
                        ("CHILDREN > PARENT", oversized_children),
                        ("DANGLING child_view_id", dangling)):
        if rows:
            print(f"\n{label} ({len(rows):,}):")
            for row in rows[: args.max_report]:
                print("   ", row)

    failed = (counts["NOT_CONTAINED"] or counts["missing_parent"]
              or counts["disowned"] or oversized_children or dangling)
    print("\n" + ("FAIL — the parent/child relationship is broken" if failed
                  else "OK — every view lives inside its parent"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
