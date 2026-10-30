"""Build a compact in-RAM copy of the dense index for serving.

The full index is 927,744 x 4096 float32 (15.2 GB) and exact search scans all of
it per query (``dalil.s8_search.dense``). That is fine on a workstation with fast NVMe
but not on a host whose disk is capped at a few thousand IOPS. Qwen3-Embedding is
trained with Matryoshka representation learning, so the first ``dims`` coordinates
of each vector, renormalised, are themselves an embedding. This module writes
them as float16 rows (``dims`` = 1024 -> 1.9 GB) for a first pass held in RAM;
the top candidates are then rescored exactly against the full vectors
(``dalil.s8_search.dense.CompactDenseIndex``).

    python -m dalil.s7_index.compact --dims 1024
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dalil.paths import DB_DIR

BLOCK_ROWS = 65536


def compact_path(db_dir: Path, dims: int) -> Path:
    """The compact index: a FAISS ``IndexScalarQuantizer`` (fp16, inner product)."""
    return db_dir / f"dense_compact_{dims}_f16.index"


def write_faiss(rows: np.ndarray, path: Path) -> None:
    """Wrap float16 rows as a FAISS fp16 scalar-quantizer index, which stores
    exactly these bytes as its codes and scans them without a float32 copy."""
    import faiss

    index = faiss.IndexScalarQuantizer(rows.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    faiss.copy_array_to_vector(np.ascontiguousarray(rows, dtype=np.float16).view(np.uint8).ravel(), index.codes)
    index.ntotal = rows.shape[0]
    part = path.with_name(path.name + ".part")
    faiss.write_index(index, str(part))
    part.replace(path)


def build(index_path: Path, dims_list: list[int], db_dir: Path) -> list[Path]:
    import faiss

    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
    d, n = index.d, index.ntotal
    if type(index) not in (faiss.IndexFlatIP, faiss.IndexFlat):
        raise SystemExit(f"expected a flat index, got {type(index).__name__}")
    offset = os.path.getsize(index_path) - n * d * 4
    raw = np.memmap(index_path, dtype=np.float32, mode="r", offset=offset, shape=(n, d))

    outs, tmp = [], []
    for dims in dims_list:
        path = compact_path(db_dir, dims)
        part = path.with_suffix(".npy.part")
        tmp.append((part, path))
        outs.append(np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(n, dims)))

    for start in range(0, n, BLOCK_ROWS):
        block = np.asarray(raw[start:start + BLOCK_ROWS])
        for dims, out in zip(dims_list, outs):
            head = block[:, :dims]
            norms = np.linalg.norm(head, axis=1, keepdims=True)
            out[start:start + len(block)] = head / np.maximum(norms, 1e-12)
        print(f"\r{min(start + BLOCK_ROWS, n):,}/{n:,}", end="", flush=True)
    print()

    for out, (part, path) in zip(outs, tmp):
        out.flush()
        write_faiss(np.load(part, mmap_mode="r"), path)
        part.unlink()
    for dims, (_, path) in zip(dims_list, tmp):
        meta = {"source": index_path.name, "ntotal": n, "source_dims": d, "dims": dims}
        path.with_name(path.stem + "_metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return [p for _, p in tmp]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db-dir", type=Path, default=DB_DIR)
    parser.add_argument("--dims", type=int, nargs="+", default=[1024])
    args = parser.parse_args()
    for path in build(args.db_dir / "dense_faiss_ar.index", args.dims, args.db_dir):
        print(path, f"{path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
