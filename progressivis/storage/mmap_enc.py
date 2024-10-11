"""
Use mmap to manage strings
"""
from __future__ import annotations

import mmap as mm
import os
import os.path
import marshal
import numpy as np
from functools import lru_cache
from ..core.pintset import PIntSet

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    Sizes = np.ndarray[Any, Any]

PAGESIZE = mm.PAGESIZE
WB = 4
MAX_SHORT = 32
MAX_SHORT_BIT_LENGTH = MAX_SHORT.bit_length()
LRU_MAX_SIZE = 128
FREELIST_SIZE = 63
MAX_OVERSIZE = (
    3  # we can reuse a chunk at most 2**MAX_OVERSIZE times greater than required
)


def _ofs(idx: int) -> int:
    return (idx + 1) * WB


class MMapObject(object):
    def __init__(self, filename: str) -> None:
        if os.path.isfile(filename):
            self._file = open(filename, "r+b")
            self._new_file = False
        else:
            self._file = open(filename, "wb+")
            os.ftruncate(self._file.fileno(), PAGESIZE)
            self._new_file = True
        self.mmap = mm.mmap(self._file.fileno(), 0)
        self.sizes: Sizes
        self.sizes = np.frombuffer(self.mmap, np.uint32)
        if self._new_file:
            self.sizes[0] = 1
        self._freelist = [PIntSet() for _ in range(FREELIST_SIZE)]

    def _allocate(self, size: int) -> None:
        if self.sizes is not None and self.sizes.base is not None and hasattr(self.sizes.base, "release"):
            self.sizes.base.release()
        self.mmap.resize(size * PAGESIZE)
        self.sizes = np.frombuffer(self.mmap, np.uint32)

    def resize(self, size: int) -> None:
        mod = size % WB
        if mod:
            size += WB - mod
        # size *= WB # size is already in bytes so ...
        if size > len(self.mmap):
            # pages = (size + PAGESIZE-1) // PAGESIZE * (1+PAGESIZE)
            pages = (size + PAGESIZE - 1) // PAGESIZE + 1
            self._allocate(pages)

    def __len__(self) -> int:
        return int(self.sizes[0])

    def encode(self, obj: Any) -> bytes:
        return marshal.dumps(obj)

    def decode(self, buf: bytes) -> Any:
        return marshal.loads(buf)

    def get(self, idx: int) -> Any:
        if idx < 0 or idx > len(self):
            raise IndexError("index %d is out of range" % idx)
        size = self.sizes[idx]
        off = _ofs(idx)
        buf = self.mmap[off : off + size * WB]
        return self.decode(buf)

    def __getitem__(self, idx: int) -> Any:
        return self.get(idx)

    def add(self, obj: Any) -> int:
        buf = self.encode(obj)
        lb = len(buf)
        if lb >= MAX_SHORT:
            return self._add_long(buf, lb, with_reuse=False)
        return self._add_short(buf, lb)

    @lru_cache(maxsize=LRU_MAX_SIZE)
    def _add_short(self, buf: bytes, lb: int) -> int:
        return self._add_long(buf, lb, with_reuse=False)

    def release(self, idx: int) -> None:
        if not idx:
            return
        lb = self.sizes[idx] * WB
        if lb <= MAX_SHORT:
            return
        pos = int(lb).bit_length() - 1
        self._freelist[pos].add(idx)

    def _get_from_freelist(self, lb: int) -> int:
        pos = int(lb).bit_length() - 1
        lw = lb // WB + 1
        for i in self._freelist[pos]:
            if self.sizes[i] >= lw:
                self._freelist[pos].remove(i)
                return i
        for i in range(pos + 1, min(pos + MAX_OVERSIZE, FREELIST_SIZE - 1)):
            bm = self._freelist[i].pop()
            if bm:
                return bm[0]
        return -1

    def _add_long(self, buf: bytes, lb: int, with_reuse: bool) -> int:
        bufsize = lb + WB - lb % WB
        assert bufsize % 4 == 0
        idx = -1
        alloc_ = True
        if with_reuse:
            idx = self._get_from_freelist(lb + 1)
        if idx == -1:
            idx = len(self)
        else:
            alloc_ = False
        off = _ofs(idx)
        if alloc_:
            self.resize(off + WB + bufsize)
            self.sizes[idx] = bufsize // WB
        self.mmap[off : off + lb] = buf
        self.mmap[off + lb : off + bufsize] = b"\x00" * (
            bufsize - lb
        )  # np.zeros(bufsize-lb, dtype=np.uint8)
        if alloc_:
            self.sizes[0] = idx + 1 + bufsize // WB
            self.sizes[idx] = bufsize // WB
        return idx

    def set_at(self, idx: int, obj: Any) -> int:
        if idx < 0 or idx > len(self):
            raise IndexError("index %d is out of range" % idx)
        buf = self.encode(obj)
        lb = len(buf)
        self.release(idx)
        if lb <= MAX_SHORT:
            return self._add_short(buf, lb)
        # long string case
        return self._add_long(buf, lb, with_reuse=True)

    def close(self) -> None:
        if not self.mmap.closed:
            if self.sizes is not None and self.sizes.base is not None and hasattr(self.sizes.base, "release"):
                self.sizes.base.release()
            self.mmap.close()
            # self.mmap = None
        if not self._file.closed:
            self._file.close()
            # self._file = None
