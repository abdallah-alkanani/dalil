"""Embed the structured corpus retrieval views with an 8B Arabic embedding model.

Reads `retrieval_views.jsonl` and embeds the `embedding_text` field, which
already carries the book / category / section-path context header, so the
vector reflects where the passage sits rather than the bare chunk.

The job is sharded and resumable: each shard writes `<shard>.npy` (float16,
L2-normalised) plus `<shard>.ids.json`, and completed shards are skipped on
restart. Within a shard, texts are sorted by length so each batch pads to a
similar size, which is a large throughput win on a single GPU.

Needs a CUDA GPU plus ``torch`` and ``sentence-transformers``, or with ``--engine vllm``
the vLLM engine (``vllm>=0.30``; run one process per GPU). Both follow the model card:
last-token pooling, L2-normalised, documents without an instruction. On an NVIDIA L4
vLLM embeds 8.1 views/s against 5.7/s for sentence-transformers, with vectors at
cosine >= 0.996 to its (2,944 views, 2026-10-17). Engines are imported only once the
run starts, so ``--help`` works without them. The shards are then appended
to the search stores with ``dalil.s7_index.append``.

    python -m dalil.s6_embed.embed_views \\
        --views data/build/corpora/collection/retrieval_views.jsonl \\
        --out embeddings/
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

KNOWN_REVISIONS = {
    "nvidia/Nemotron-3-Embed-8B-BF16": "dd9d06b1852ef836d3dc2fbd2b71a34356912759",
    "Qwen/Qwen3-Embedding-8B": "4e423935c619ae4df87b646a3ce949610c66241c",
}
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"


ARABIC_RE = re.compile(r"[ء-ي]")


def too_little_arabic(r: dict, min_arabic: int) -> bool:
    """True for a view whose display_text has fewer than min_arabic Arabic letters.

    A Quran verse is never too short: طه, يسٓ and the seven حمٓ are whole verses
    of two letters, and dropping them left their parents out of the stores.
    """
    content = r.get("display_text")
    if content is None or r.get("view_type") == "ayah_exact":
        return False
    return len(ARABIC_RE.findall(content)) < min_arabic


def iter_views(path: Path, text_field: str, skip_types: set[str], min_arabic: int):
    """Yield (view_id, text) for every view worth embedding.

    Views whose *content* is only markers, numerals or punctuation ('#', '١',
    '؟', '٣٠ - ٣١ -') are skipped. They are legitimate structural records, but
    as vectors they carry no retrievable text — only the shared context header —
    so embedding them spends GPU time producing near-duplicate vectors that can
    outrank real content. The filter looks at display_text, since embedding_text
    always contains Arabic in its header.
    """
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("view_type") in skip_types:
                skipped += 1
                continue
            text = r.get(text_field) or ""
            if not text.strip():
                skipped += 1
                continue
            if too_little_arabic(r, min_arabic):
                skipped += 1
                continue
            yield r["view_id"], text
    print(f"skipped {skipped:,} views (no embeddable content)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--engine", choices=("sentence-transformers", "vllm"), default="sentence-transformers")
    ap.add_argument("--revision", default=None,
                    help="defaults to the pinned revision for known models")
    ap.add_argument("--text-field", default="embedding_text")
    ap.add_argument("--skip-view-types", default="",
                    help="comma-separated view_type values to leave unembedded")
    ap.add_argument("--shard-size", type=int, default=2_000,
                    help="views per saved shard; a crash loses at most one shard")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="unused when --token-budget drives batching; kept for reference")
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens per batch; caps VRAM independently of length")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--prompt-name", default="document")
    ap.add_argument("--min-arabic", type=int, default=3,
                    help="skip views whose display_text has fewer Arabic letters")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    revision = args.revision or KNOWN_REVISIONS.get(args.model)

    # Optional heavy dependencies, imported only once a run actually starts.
    import torch
    if args.engine == "vllm":
        if args.prompt_name != "document":
            raise SystemExit("--engine vllm embeds documents only (no instruction prefix)")
        from vllm import LLM
    else:
        from sentence_transformers import SentenceTransformer

    args.out.mkdir(parents=True, exist_ok=True)
    skip_types = {t for t in args.skip_view_types.split(",") if t}

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = None
    shard_idx = 0
    buf_ids: list[str] = []
    buf_txt: list[str] = []
    seen = 0
    t_start = time.time()
    total_embedded = 0

    def flush(idx: int, ids: list[str], texts: list[str]) -> None:
        nonlocal model, total_embedded
        if not ids:
            return
        npy = args.out / f"shard_{idx:05d}.npy"
        idsf = args.out / f"shard_{idx:05d}.ids.json"
        if npy.exists() and idsf.exists():
            done = json.loads(idsf.read_text())
            if len(done) == len(ids):
                print(f"shard {idx:05d}: already done ({len(ids)}), skipping", flush=True)
                total_embedded += len(ids)
                return
        if args.engine == "vllm":
            if model is None:
                t0 = time.time()
                model = LLM(model=args.model, revision=revision, runner="pooling", dtype="bfloat16",
                            max_model_len=args.max_seq_length)
                print(f"vLLM engine ready in {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
            vecs = np.asarray([o.outputs.embedding for o in model.embed(texts, use_tqdm=False)], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            np.save(npy, vecs.astype(np.float16))
            idsf.write_text(json.dumps(ids), encoding="utf-8")
            total_embedded += len(ids)
            dt = time.time() - t0
            print(f"shard {idx:05d}: {len(ids)} views in {dt/60:.1f} min ({len(ids)/dt:.1f}/s) dim={vecs.shape[1]} "
                  f"| total {total_embedded} in {(time.time()-t_start)/60:.1f} min", flush=True)
            return
        if model is None:
            t0 = time.time()
            model = SentenceTransformer(
                args.model, revision=revision, device="cuda",
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            model.max_seq_length = args.max_seq_length
            print(f"model loaded in {time.time()-t0:.0f}s, "
                  f"max_seq_length={model.max_seq_length}", flush=True)

        # Exact token lengths first: they drive both the sort and the batch sizing,
        # and batch-encoding 50k texts with the fast tokenizer costs only seconds.
        tokenizer = model.tokenizer
        tok_lens = [
            min(len(token_ids), args.max_seq_length)
            for token_ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
        ]
        order = sorted(range(len(texts)), key=lambda i: tok_lens[i])

        # Token-budget batching. A fixed batch size has to be tuned for the
        # longest sequence and then wastes the GPU on the short ones; budgeting
        # padded tokens instead keeps memory flat whatever the length mix, so a
        # high --max-seq-length costs time on long chunks rather than OOM-ing.
        batches: list[list[int]] = []
        current: list[int] = []
        for i in order:
            longest = max(tok_lens[i], tok_lens[current[0]] if current else 0)
            if current and (longest * (len(current) + 1) > args.token_budget
                            or len(current) >= args.max_batch):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)

        def encode(idxs: list[int]):
            return model.encode(
                [texts[i] for i in idxs],
                batch_size=len(idxs),
                prompt_name=args.prompt_name,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        t0 = time.time()
        restored = None
        oom_splits = 0
        for batch in batches:
            # A hard OOM should cost one batch, not the whole run: free the
            # cache and retry on halves, down to a single sequence.
            queue = [batch]
            while queue:
                chunk = queue.pop(0)
                try:
                    vecs = encode(chunk)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(chunk) == 1:
                        raise
                    mid = len(chunk) // 2
                    queue[:0] = [chunk[:mid], chunk[mid:]]
                    oom_splits += 1
                    continue
                if restored is None:
                    restored = np.empty((len(texts), vecs.shape[1]), dtype=vecs.dtype)
                restored[np.asarray(chunk)] = vecs
        np.save(npy, restored.astype(np.float16))
        idsf.write_text(json.dumps(ids), encoding="utf-8")
        total_embedded += len(ids)
        dt = time.time() - t0
        rate = len(ids) / dt if dt else 0
        elapsed = time.time() - t_start
        print(f"shard {idx:05d}: {len(ids)} views in {dt/60:.1f} min "
              f"({rate:.1f}/s) dim={restored.shape[1]} batches={len(batches)} "
              f"maxtok={max(tok_lens)} oom_splits={oom_splits} | total {total_embedded} in {elapsed/60:.1f} min",
              flush=True)

    for vid, text in iter_views(args.views, args.text_field, skip_types, args.min_arabic):
        buf_ids.append(vid)
        buf_txt.append(text)
        seen += 1
        if len(buf_ids) >= args.shard_size:
            flush(shard_idx, buf_ids, buf_txt)
            shard_idx += 1
            buf_ids, buf_txt = [], []
        if args.limit and seen >= args.limit:
            break
    flush(shard_idx, buf_ids, buf_txt)

    manifest = {
        "model": args.model,
        "revision": revision,
        "engine": args.engine,
        "text_field": args.text_field,
        "skipped_view_types": sorted(skip_types),
        "min_arabic_letters": args.min_arabic,
        "max_seq_length": args.max_seq_length,
        "token_budget": args.token_budget,
        "prompt_name": args.prompt_name,
        "normalized": True,
        "dtype": "float16",
        "total_views": total_embedded,
        "shards": sorted(p.name for p in args.out.glob("shard_*.npy")),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("total_views", "elapsed_min", "model")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
