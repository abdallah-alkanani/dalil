"""Ask DeepSeek where an oversized parent should be cut, and record the answer.

A parent is the unit handed whole to the answering model, so *where* it is cut
decides whether it is usable. Cut in the middle of an argument and both halves are
ruined: the first states a question it never answers, the second gives a ruling
whose subject is missing. Classical Arabic prose is frequently unpunctuated, so
there is often no mechanical seam to find — the previous splitter fell back to
cutting at a word count, which lands mid-sentence, and that is what this replaces.

The decision is therefore made by a reader. It is made *here*, out of band, and
written to a file; `dalil.s4_extract` only applies what this recorded. That
keeps extraction deterministic — same inputs plus same boundary file, same corpus,
no model call in the extraction path — and it makes every cut reviewable before a
single vector is computed.

The model never sees a whole parent; some run to 604,000 words. The text is instead
covered by PROBES — contiguous slices, each asked one question: within this slice,
where does the subject change? It answers with a verbatim anchor phrase, which is
located in the text and stored as a word offset. Probes tile rather than sample:
width equals spacing, so every word is inside exactly one probe and no boundary can
fall in a blind gap.

Nothing is trusted on the model's word. An anchor that does not appear in its probe,
or appears more than once, is discarded and *no cut is made there* — the piece
simply runs on to the next accepted boundary. There is deliberately no mechanical
fallback: a cut that looks deliberate but is not is worse than a parent that is
visibly too long, because the oversized parent is at least detectable.

Matching is done on a normalised token stream, not on bytes. The corpus carries
tashkeel and source debris that the model drops when it quotes, so a byte
comparison reports a correct anchor as absent. What gets *recorded* is the corpus's
own wording at the resolved offset, so the applier's exact check still means
something.

Output is one JSON object per parent:

    {"parent_id": …, "book_id": …, "original_words": 604295,
     "cuts": [{"word_index": 3981, "anchor": "<verbatim>", "reason": "…"}, …],
     "stats": {"asked": 151, "accepted": 143, "rejected": 8}}

Resumable: parents already present in --out are skipped, so an interrupted run
continues where it stopped.

    python -m dalil.s3_structure.semantic_boundaries --parents parents.jsonl --out boundaries.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType

from dalil.arabic.normalize import normalize_for_search

# Three transports reach the same model, and they differ by more than an order of
# magnitude. Measured on one identical 2,500-word chunk:
#
#   openrouter    (openrouter.ai)      4s     separate quota, so it can run alongside
#   ollama_cli    (`ollama run`)      66s
#   ollama_cloud  (ollama.com HTTP)   40-500s, wildly variable
#
# The transport is chosen at runtime rather than imported, so a run can be pointed
# at whichever is healthy without editing code.
from dalil.llm import TRANSPORTS, LLMError, get_transport

PROMPT = """You are dividing a long Arabic Islamic text into coherent retrieval units.

Below is a CHUNK of a much longer work. List EVERY place in this chunk where the
SUBJECT CHANGES — where the author finishes one point and begins another. Think: a
new مسألة, a new باب/فصل, a new hadith with its isnad, a shift from one ruling to
the next, or the start of a new line of argument.

Rules:
- Each cut must be a place where a reader would say "a new topic starts here".
- NEVER cut in the middle of a sentence, a hadith, an isnad, a Quranic quote, or
  between a question and its answer.
- Give each cut as a VERBATIM phrase of 4-8 words copied EXACTLY from the chunk,
  marking the FIRST words of the NEW topic. Each will be searched for, so it must
  appear in the chunk exactly as written, and appear only ONCE — if a phrase would
  be ambiguous, extend it until it is unique.
- List them in the order they appear. Aim for a real boundary roughly every
  %(target)s words, but follow the text: give more where the text really turns more
  often, fewer where it does not.
- If the chunk is one continuous topic with no good break, return an empty list.
  An empty list is correct and expected; a bad cut is far worse than no cut.

Return STRICT JSON only, no markdown fence:
{"anchors": ["<verbatim 4-8 words>", "<verbatim 4-8 words>", ...],
 "reason": "<=100 chars, ENGLISH ONLY"}

CHUNK:
<<<
%(chunk)s
>>>"""

MAX_ANCHORS_PER_REPLY = 1024
MIN_ANCHOR_WORDS = 3
MAX_ANCHOR_WORDS = 14


def body_words(row: dict) -> list[str]:
    """The words of a parent's own text: what `dalil.s4_extract` splits and measures.

    Not `text_for_synthesis`, which also carries the Shamela footnotes appended
    after `الحواشي:`: a 12,917-word section with 28,000 words of notes is not
    oversized, and a cut among its notes can never apply.
    """
    return (row.get("text_original") or row["text_for_synthesis"]).split()


def norm_stream(words: list[str]) -> tuple[list[str], list[int]]:
    """Normalised token stream plus, for each token, the word it came from.

    Matching cannot be done on the raw words. The corpus carries tashkeel and
    source debris that the model silently drops when it quotes: asked for an
    anchor it answered `كتاب ( إقامة الدليل على إبطال التحليل )` where the text
    reads `كتاب ( إقامة ~~الدليل على ...` — the same phrase, and a byte comparison
    calls it absent. Normalisation also splits or deletes tokens (punctuation
    becomes whitespace), so the two sides can disagree on word count as well; the
    owner list is what maps a match back to a real word offset in the original.
    """
    tokens: list[str] = []
    owners: list[int] = []
    for i, word in enumerate(words):
        for tok in normalize_for_search(word, clean_markers=False).split():
            tokens.append(tok)
            owners.append(i)
    return tokens, owners


def find_unique(haystack: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [i for i in range(len(haystack) - len(needle) + 1)
            if haystack[i:i + len(needle)] == needle]


def resolve_chunk(words: list[str], lo: int, hi: int, target: int,
                  transport: ModuleType, model: str, timeout: int,
                  retries: int = 6) -> tuple[list[dict], collections.Counter]:
    """Ask one question about words[lo:hi] and get back every boundary it holds.

    One call per chunk rather than one call per cut. A 555,464-word parent needed
    222 round trips when each asked for a single boundary; asking each chunk for
    all of its boundaries turns that into ~28, and the model is reading the same
    text either way. Each returned anchor is resolved and verified independently,
    so one bad anchor in a reply costs that cut and nothing else.
    """
    chunk = words[lo:hi]
    stats: collections.Counter = collections.Counter()
    try:
        reply = transport.ask_json(PROMPT % {"chunk": " ".join(chunk), "target": f"{target:,}"},
                                   model=model, timeout=timeout, retries=retries)
    except LLMError as exc:
        stats["error"] += 1
        # Keep the message. Counting failures tells you a parent broke; only the
        # message tells you whether it was a timeout, a rate limit, or a reply that
        # would not parse — and therefore whether retrying can possibly help.
        stats["last_error"] = str(exc)[:160]
        return [], stats

    anchors = reply.get("anchors")
    if isinstance(anchors, str):          # a single string instead of a list
        anchors = [anchors]
    if not isinstance(anchors, list) or not anchors:
        stats["no_boundary"] += 1
        return [], stats
    reason = (reply.get("reason") or "")[:100]

    hay, owners = norm_stream(chunk)
    out: list[dict] = []
    # Whole-parent calls legitimately return hundreds of boundaries.
    for raw in anchors[:MAX_ANCHORS_PER_REPLY]:
        if not isinstance(raw, str) or not raw.strip():
            stats["bad_length"] += 1
            continue
        if not MIN_ANCHOR_WORDS <= len(raw.split()) <= MAX_ANCHOR_WORDS:
            stats["bad_length"] += 1
            continue
        needle, _ = norm_stream(raw.split())
        hits = find_unique(hay, needle)
        if len(hits) != 1:
            stats["absent" if not hits else "ambiguous"] += 1
            continue
        # Store the corpus's own words, not the model's rendering of them. The
        # applier re-checks the anchor at this offset, so what is recorded has to
        # be what is actually there.
        first = owners[hits[0]]
        last = owners[min(hits[0] + len(needle) - 1, len(owners) - 1)]
        out.append({"word_index": lo + first,
                    "anchor": " ".join(chunk[first:last + 1]),
                    "model_anchor": " ".join(raw.split()), "reason": reason})
    stats["returned"] += len(anchors)
    out.sort(key=lambda c: c["word_index"])
    return out, stats


def gap_ranges(rec: dict, above: int, min_gap: int) -> list[list[int]]:
    """Word ranges of one recorded parent that still have to be asked about.

    Two kinds of hole end up in a boundary file and they look nothing alike.
    A parent whose chunk failed is marked `partial` and lists its `missing_ranges`
    honestly, so those come back verbatim. The dangerous kind is silent: an early
    run recorded a failed chunk as a clean "no boundaries here", which is
    indistinguishable from a genuinely continuous stretch — three pieces of 46k,
    83k and 29k words in one 555,464-word parent were exactly that. So any piece
    still longer than `above` is re-asked too. Re-asking is cheap and cannot do
    harm: every anchor is verified against the text, a piece that really is
    continuous simply comes back empty a second time, and the alternative is
    shipping an 83,000-word parent because a transport blipped once.
    """
    out = [list(r) for r in (rec.get("missing_ranges") or [])]
    cuts = sorted(c["word_index"] for c in rec.get("cuts") or [])
    edges = [0] + cuts + [rec["original_words"]]
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo <= above:
            continue
        if any(lo < b and a < hi for a, b in out):   # already listed as missing
            continue
        out.append([lo, hi])
    out.sort()
    return [r for r in out if r[1] - r[0] >= min_gap]


def tile(lo: int, hi: int, width: int, min_gap: int) -> list[tuple[int, int]]:
    """Cover [lo, hi) with contiguous chunks; fold a final sliver into its neighbour.

    Chunks tile the text end to end rather than sampling it. An earlier version
    asked about a 900-word window every 4,000 words, which meant 78% of the text
    was never looked at: a boundary that fell in a gap could not be found, and the
    parent came back "continuous" on the strength of a fifth of its own content.
    """
    spans = [(x, min(x + width, hi)) for x in range(lo, hi, width)]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_gap:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parents", type=Path, required=True, help="parents.jsonl from a pass-1 extract")
    ap.add_argument("--out", type=Path, required=True, help="boundary decisions (jsonl, appended)")
    ap.add_argument("--chunk-words", type=int, default=8000,
                    help="how much text goes in one question; chunks tile the parent, "
                         "so every word is looked at exactly once")
    ap.add_argument("--target-words", type=int, default=4000,
                    help="the spacing of boundaries suggested to the model; guidance "
                         "only, the text decides")
    ap.add_argument("--min-words", type=int, default=4000, help="only parents longer than this")
    ap.add_argument("--min-gap", type=int, default=600,
                    help="reject a cut that would leave a piece shorter than this. Keep "
                         "this small: it records dense candidates, and the parent size "
                         "actually wanted is chosen when the boundaries are applied")
    ap.add_argument("--flag", default=None,
                    help="only parents carrying this quality_flag (e.g. needs_semantic_split)")
    ap.add_argument("--limit", type=int, default=0, help="max parents this run (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent boundary questions")
    ap.add_argument("--transport", default="openrouter", choices=TRANSPORTS,
                    help="which path to the model; they differ ~20x in speed")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the right one for the transport")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the queue smallest-first instead of biggest-first, so a "
                         "second run can work the same list from the other end")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=6,
                    help="attempts per chunk before giving up on it. Keep this low when "
                         "running few workers: a chunk that keeps timing out blocks a "
                         "worker for retries x timeout, and a skipped chunk only costs "
                         "the cuts in that slice, never a bad cut")
    ap.add_argument("--regap", type=Path, default=None,
                    help="repair mode: an existing boundary file whose holes are to be "
                         "filled. Only the unasked ranges are sent to the model — a "
                         "partial record's own missing_ranges, plus any piece still "
                         "longer than --regap-above, which is how a chunk that failed "
                         "silently and was banked as 'no boundaries' gets caught. The "
                         "repaired records are written to --out; everything else in the "
                         "file is copied across unchanged")
    # Must equal PARENT_SPLIT_ABOVE in dalil.s4_extract; change the two together.
    ap.add_argument("--regap-above", type=int, default=22500,
                    help="in --regap mode, re-ask any recorded piece longer than this. "
                         "Defaults to PARENT_SPLIT_ABOVE: a piece the size policy would "
                         "have split has no business carrying zero boundaries")
    args = ap.parse_args(argv)
    # --out is appended to and its finished parents count as done, so repairing a
    # file in place would skip its silent holes and leave a partial parent's old
    # record next to the repaired one.
    if args.regap and args.out.resolve() == args.regap.resolve():
        ap.error("--out must be a new file, not the --regap file being repaired")
    transport = get_transport(args.transport)
    if args.model is None:
        args.model = transport.DEFAULT_MODEL

    # Two ways in. A normal run walks parents.jsonl and asks about every parent it
    # has not already banked. A --regap run walks an existing boundary file instead
    # and asks only about the holes in it, so closing a 46,000-word gap costs 6
    # calls rather than the 70 a whole re-run of that parent would.
    seed: dict[str, dict] = {}          # parent_id -> the record being repaired
    ranges_by_parent: dict[str, list[list[int]]] = {}
    carried: list[dict] = []            # regap records that need no work
    done: set[str] = set()

    if args.regap:
        for line in args.regap.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            holes = gap_ranges(rec, args.regap_above, args.min_gap)
            if holes:
                seed[rec["parent_id"]] = rec
                ranges_by_parent[rec["parent_id"]] = holes
            else:
                carried.append(rec)
        print(f"{len(seed):,} parents with holes, {len(carried):,} already complete; "
              f"{sum(h[1] - h[0] for hs in ranges_by_parent.values() for h in hs):,} "
              f"words to re-ask", flush=True)

    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            # A partial record still has gaps; it must not count as done.
            if not row.get("partial"):
                done.add(row["parent_id"])

    todo: list[dict] = []
    for line in args.parents.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["parent_id"] in done:
            continue
        if args.regap:
            if row["parent_id"] in seed:
                todo.append(row)
            continue
        if args.flag and args.flag not in (row.get("quality_flags") or []):
            continue
        if len(body_words(row)) <= args.min_words:
            continue
        todo.append(row)
    todo.sort(key=lambda r: -len(body_words(r)))
    if args.limit:
        todo = todo[: args.limit]

    if args.regap:
        missing = sorted(set(seed) - {r["parent_id"] for r in todo} - done)
        if missing:
            print(f"WARNING: {len(missing)} parent(s) in --regap are not in --parents; "
                  f"their holes cannot be closed: {missing[0][-30:]}...", flush=True)
        # A repaired file has to be a whole file, not a diff, or the next step gets
        # a corpus with most of its parents missing.
        with args.out.open("a", encoding="utf-8") as fh:
            for rec in carried:
                if rec["parent_id"] not in done:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(rec["parent_id"])

    print(f"{len(todo):,} parents to resolve ({len(done):,} already done), "
          f"{args.workers} workers", flush=True)
    if not todo:
        return 0

    # Every chunk of every parent is one independent question, so they all go into
    # a single pool. Doing a parent at a time instead would leave most workers idle
    # for most of the run: the majority of these parents are one or two chunks long,
    # and only the handful of enormous ones can fill a pool on their own.
    tasks: list[tuple[int, int, int]] = []
    words_by_parent: list[list[str]] = []
    for idx, row in enumerate(todo):
        words = body_words(row)
        words_by_parent.append(words)
        if args.regap:
            for gap_lo, gap_hi in ranges_by_parent[row["parent_id"]]:
                tasks.extend((idx, lo, hi) for lo, hi in
                             tile(gap_lo, min(gap_hi, len(words)),
                                  args.chunk_words, args.min_gap))
        else:
            tasks.extend((idx, lo, hi)
                         for lo, hi in tile(0, len(words), args.chunk_words, args.min_gap))

    # Biggest parents first, because they are the whole point: a 4,000-word parent
    # is already inside the wanted size and gains nothing from being cut, while a
    # 555,464-word one is the reason this tool exists. Ordering smallest-first banks
    # records sooner but spends the run on parents that need no splitting.
    #
    # The cost is that a big parent writes nothing until all its chunks are back, so
    # its work is lost if the run dies mid-parent. That is the right trade: chunks
    # within a parent are still ordered, so the loss is bounded by one parent, and
    # the alternative is finishing a night's run with every large parent untouched.
    order = 1 if args.reverse else -1
    tasks.sort(key=lambda t: (order * len(words_by_parent[t[0]]), t[0], t[1]))

    print(f"{len(tasks):,} chunks of ~{args.chunk_words:,} words to ask", flush=True)
    totals = collections.Counter()
    found: dict[int, list[dict]] = collections.defaultdict(list)
    if args.regap:
        # A repair keeps everything the earlier run already found — only the holes
        # were re-asked, and the old cuts are the bulk of the answer. They go in
        # before the new ones so the min-gap pass below sees one merged sequence.
        for idx, row in enumerate(todo):
            found[idx].extend(seed[row["parent_id"]]["cuts"])
    pending = collections.Counter(idx for idx, _, _ in tasks)
    failed: set[int] = set()
    errors_by_parent: dict[int, str] = {}
    failed_spans: dict[int, list] = {}
    completed = [0]

    def work(task: tuple[int, int, int]):
        idx, lo, hi = task
        cuts, st = resolve_chunk(words_by_parent[idx], lo, hi, args.target_words,
                                 transport, args.model, args.timeout, args.retries)
        return idx, cuts, st, lo, hi

    started = time.time()
    with args.out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, t) for t in tasks]
            # as_completed, not map: map yields strictly in submission order, so one
            # slow chunk stalls the reporting and the writing of every parent behind
            # it even when their work is already done.
            for fut in as_completed(futures):
                idx, cuts, st, task_lo, task_hi = fut.result()
                message = st.pop("last_error", None)
                if message:
                    print(f"    chunk failed: {message}", flush=True)
                    errors_by_parent.setdefault(idx, message)
                totals.update(st)
                totals["asked"] += 1
                if st.get("error"):
                    failed_spans.setdefault(idx, []).append([task_lo, task_hi])
                    # A transport failure is not evidence that the text has no
                    # boundaries, but a record written with zero cuts is
                    # indistinguishable from one that says so — and resume then
                    # skips the parent forever. Whole-parent calls produced exactly
                    # that: an HTTP 400 and an empty reply, both banked as "no
                    # boundaries found". Mark the parent instead, and skip writing it.
                    failed.add(idx)
                found[idx].extend(cuts)
                completed[0] += 1
                pending[idx] -= 1
                # Every chunk: at low worker counts anything coarser is blind.
                rate = completed[0] / max(1e-9, time.time() - started) * 60
                left = (len(tasks) - completed[0]) / max(1e-9, rate)
                print(f"    …{completed[0]}/{len(tasks)} chunks  "
                      f"{rate:.1f}/min  ~{left:.0f} min left", flush=True)
                if pending[idx]:
                    continue
                # Every chunk of this parent is in; thin and write it now rather
                # than at the end, so an interrupted run keeps what it finished.
                row = todo[idx]
                ordered = sorted(found.pop(idx), key=lambda c: c["word_index"])
                partial = idx in failed
                kept: list[dict] = []
                for cut in ordered:
                    # Two chunks can propose boundaries a few words apart at their
                    # shared edge, and a cut at word 0 is not a cut. Keep only ones
                    # that advance the text.
                    prev = kept[-1]["word_index"] if kept else 0
                    if cut["word_index"] <= prev or cut["word_index"] - prev < args.min_gap:
                        totals["too_close"] += 1
                        continue
                    kept.append(cut)
                    totals["accepted"] += 1
                # Keep what succeeded. A parent whose 40 chunks answered and whose
                # 3rd chunk 402'd is not "no boundaries found" — it is mostly solved,
                # and discarding it throws away every good chunk to punish one bad
                # one. Record the cuts we have, mark the parent partial, and list the
                # word ranges still unasked so a retry re-asks only those.
                rec = {"parent_id": row["parent_id"], "book_id": row["book_id"],
                       "original_words": len(words_by_parent[idx]), "cuts": kept}
                if partial:
                    rec["partial"] = True
                    rec["missing_ranges"] = sorted(failed_spans.get(idx, []))
                    rec["error"] = errors_by_parent.get(idx, "a chunk failed")
                    totals["parents_partial"] += 1
                    print(f"[{completed[0]}/{len(tasks)} chunks] "
                          f"{len(words_by_parent[idx]):>8,}w  PARTIAL {len(kept)} cuts, "
                          f"{len(rec['missing_ranges'])} gap(s) — {rec['error'][:40]}",
                          flush=True)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{completed[0]}/{len(tasks)} chunks] "
                      f"{rec['original_words']:>8,}w -> {len(kept):>4} cuts  "
                      f"{row['parent_id'][-26:]}", flush=True)

    asked = totals.get("asked", 0)
    acc = totals.get("accepted", 0)
    print(f"\nasked {asked:,} | accepted {acc:,} ({100 * acc / max(1, asked):.0f}%)")
    for why in ("no_boundary", "absent", "ambiguous", "too_close", "bad_length", "error", "returned"):
        if totals.get(why):
            print(f"  rejected {why:<12} {totals[why]:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
