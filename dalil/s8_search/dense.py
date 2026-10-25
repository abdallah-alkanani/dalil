"""Exact dense search over the flat FAISS index, streamed from disk.

The production index is an ``IndexFlatIP`` of 880k x 4096 float32 vectors:
14.4 GB. Reading it into RAM per query (``faiss.read_index``) needs more memory
than a 16 GB machine has free, and memory-mapping it (``IO_FLAG_MMAP_IFC``) turns
the scan into 4 KB page faults: 27 s per query measured on NVMe that streams the
same file at 4.3 GB/s.

So the index is memory-mapped only for its header and ``reconstruct`` (MMR), and
searches stream the vectors in large sequential blocks. When the file cannot fit
in the page cache anyway, blocks are read with O_DIRECT: buffered reads of a file
that only thrashes the cache measured 1.9 GB/s, direct reads 3.7 GB/s. Each block is scored with
FAISS's own exact kernel (``faiss.knn``) one query at a time, the same code path
``IndexFlatIP.search`` takes for a single query, and the per-block top-k are
merged with ``faiss.ResultHeap``, the merge FAISS uses for sliced datasets
(``faiss.contrib.exhaustive_search.knn_ground_truth``). Results are the same
scores and ids as ``IndexFlatIP.search``. Memory stays at a few blocks, and one
pass from disk serves every query variant.

Indexes that are not a plain float32 ``IndexFlat`` fall back to ``index.search``.
"""
from __future__ import annotations

import mmap
import os
import queue
import threading
from pathlib import Path

import numpy as np

# Multiple of 4 so FAISS's 4-wide inner-product kernels group rows the same way
# as in a whole-index scan. 32768 x 4096 x 4 B = 512 MB per block.
BLOCK_ROWS = 32768
BUFFERS = 3  # one being read, one being scored, one spare
ALIGN = 4096  # O_DIRECT offset, length and buffer alignment


def _fits_in_page_cache(nbytes: int) -> bool:
    """Whether the OS could keep ``nbytes`` cached; True when unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return nbytes < int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    return True


class DenseIndex:
    def __init__(self, path: Path):
        import faiss

        self._faiss = faiss
        self.path = path
        self.index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP_IFC | faiss.IO_FLAG_READ_ONLY)
        self.d = self.index.d
        self.ntotal = self.index.ntotal
        self.metric_type = self.index.metric_type
        vector_bytes = self.ntotal * self.d * 4
        self._offset = os.path.getsize(path) - vector_bytes
        # Flat float32 codes are the last ntotal*d*4 bytes of the file.
        self._streamable = type(self.index) in (faiss.IndexFlatIP, faiss.IndexFlat) and self._offset > 0

    def reconstruct(self, row: int) -> np.ndarray:
        return self.index.reconstruct(row)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k for each query row. Returns (scores, ids), each (nq, k)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.d)
        k = min(int(k), self.ntotal)
        if not self._streamable:
            return self.index.search(queries, k)

        faiss = self._faiss
        keep_max = faiss.is_similarity_metric(self.metric_type)
        heaps = [faiss.ResultHeap(1, k, keep_max=keep_max) for _ in range(len(queries))]
        for start, block in self._blocks():
            kb = min(k, len(block))
            for qi, heap in enumerate(heaps):
                scores, ids = faiss.knn(queries[qi:qi + 1], block, kb, metric=self.metric_type)
                heap.add_result(scores, ids + start)
        for heap in heaps:
            heap.finalize()
        return np.vstack([h.D for h in heaps]), np.vstack([h.I for h in heaps])

    def _open(self) -> tuple[int, bool]:
        """Open the index for streaming: (fd, direct)."""
        if hasattr(os, "O_DIRECT") and not _fits_in_page_cache(self.ntotal * self.d * 4):
            try:
                return os.open(self.path, os.O_RDONLY | os.O_DIRECT), True
            except OSError:  # e.g. tmpfs: no O_DIRECT
                pass
        fd = os.open(self.path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        return fd, False

    def _blocks(self):
        """Yield (first_row, block) in order; a reader thread keeps the next block loading."""
        block_bytes = BLOCK_ROWS * self.d * 4
        free: queue.Queue = queue.Queue()
        ready: queue.Queue = queue.Queue()
        for _ in range(BUFFERS):
            free.put(mmap.mmap(-1, block_bytes + 2 * ALIGN))  # page-aligned, as O_DIRECT needs
        stop = threading.Event()
        fd, direct = self._open()

        def read_into(view: memoryview, pos: int, nbytes: int) -> None:
            got = 0
            while got < nbytes:
                n = os.preadv(fd, [view[got:nbytes]], pos + got)
                if n <= 0:
                    raise OSError(f"short read in {self.path} at byte {pos + got}")
                got += n

        def reader() -> None:
            try:
                for start in range(0, self.ntotal, BLOCK_ROWS):
                    rows = min(BLOCK_ROWS, self.ntotal - start)
                    buf = free.get()
                    if stop.is_set():
                        return
                    pos, nbytes = self._offset + start * self.d * 4, rows * self.d * 4
                    if direct:
                        # Read the enclosing aligned span; the vectors start mid-page.
                        lo = pos - pos % ALIGN
                        hi = min(-(-(pos + nbytes) // ALIGN) * ALIGN, lo + len(buf))
                        view = memoryview(buf)
                        got = 0
                        while got < pos + nbytes - lo:
                            n = os.preadv(fd, [view[got:hi - lo]], lo + got)
                            if n <= 0:
                                raise OSError(f"short read in {self.path} at byte {lo + got}")
                            got += n
                        skip = pos - lo
                    else:
                        read_into(memoryview(buf), pos, nbytes)
                        skip = 0
                    block = np.frombuffer(buf, dtype=np.float32, count=rows * self.d, offset=skip)
                    ready.put((start, buf, block.reshape(rows, self.d)))
                ready.put(None)
            except BaseException as exc:  # surface reader failures to the consumer
                ready.put(exc)

        thread = threading.Thread(target=reader, name="dense-block-reader", daemon=True)
        thread.start()
        try:
            while (item := ready.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                start, buf, block = item
                yield start, block
                del block
                free.put(buf)
        finally:
            stop.set()
            free.put(None)  # unblock a reader waiting for a buffer
            thread.join(timeout=5)
            os.close(fd)
