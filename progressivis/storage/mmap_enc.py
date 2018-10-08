"""
Use mmap to manage strings
"""
from __future__ import absolute_import
import mmap
import os
import os.path
from resource import getpagesize
import marshal
import six
import numpy as np
from functools import lru_cache

PAGESIZE = getpagesize()
WB = 4
MAX_SHORT = 32
LRU_MAX_SIZE = 128

def _ofs(idx):
    return (idx+1)*WB

class MMapObject(object):
    def __init__(self, filename):
        if os.path.isfile(filename):
            self._file = open(filename, "r+b")
            self._new_file = False
        else:
            self._file = open(filename, "wb+")
            os.ftruncate(self._file.fileno(), PAGESIZE)
            self._new_file = True            
        self.mmap = mmap.mmap(self._file.fileno(), 0)
        self.sizes = np.frombuffer(self.mmap, np.uint32)
        if self._new_file:
            self.sizes[0] = 1
        self._freelist = []

    def _allocate(self, size):
        self.mmap.resize(size*PAGESIZE)
        self.sizes = np.frombuffer(self.mmap, np.uint32)

    def resize(self, size):
        mod = size%WB
        if mod:
            size += WB - mod
        #size *= WB # size is already in bytes so ...
        if size > len(self.mmap):
            #pages = (size + PAGESIZE-1) // PAGESIZE * (1+PAGESIZE)
            pages = (size + PAGESIZE-1) // PAGESIZE + 1
            self._allocate(pages)

    def __len__(self):
        return self.sizes[0]

    def encode(self, obj):
        return marshal.dumps(obj)

    def decode(self, buf):
        return marshal.loads(buf)


    def get(self, idx):
        if idx < 0 or idx > len(self):
            raise IndexError('index %d is out of range' % idx)
        size = self.sizes[idx]
        off = _ofs(idx)
        buf = self.mmap[off:off+size*WB]
        return self.decode(buf)

    def __getitem__(self, idx):
        return self.get(idx)


    def ad_d(self, obj):
        buf = self.encode(obj)
        lb = len(buf)
        if lb >= MAX_SHORT:
            return self._add_long(buf, lb)
        return self._add_short(buf, lb)
    @lru_cache(maxsize=LRU_MAX_SIZE)
    def _add_short(self, buf, lb):
        return self._add_long(buf, lb, with_reuse=False)
    #def _add_long(self, buf, lb):
    #    return self._add_string(buf, lb)
    def _add_to_freelist(self, idx):
        self._freelist.append(idx)
    def _get_from_freelist(self, lb):
        lw = lb//4+1
        for idx in self._freelist:
            if self.sizes[idx] >= lw:
                #print("HIT: ", lb, lw, self.sizes[idx])
                return idx
        #print("MISS: ", lb, lw)
        return -1
    def _add_long(self, buf, lb, with_reuse):
        bufsize = lb + WB - lb%WB
        assert bufsize % 4 == 0
        idx = -1
        alloc_ = True
        if with_reuse:
            idx = self._get_from_freelist(lb+1)
        if idx == -1:
            idx = len(self)
        else:
            alloc_ = False
        off = _ofs(idx)
        if alloc_:
            self.resize(off+WB+bufsize)
            self.sizes[idx] = bufsize//WB
        self.mmap[off:off+lb] = buf
        self.mmap[off+lb:off+bufsize] = b'\x00'*(bufsize-lb)  #np.zeros(bufsize-lb, dtype=np.uint8)
        if alloc_:
            self.sizes[0] = idx + 1 + bufsize//WB
            self.sizes[idx] = bufsize//WB
        return idx
    def set_at(self, idx, obj):
        if idx < 0 or idx > len(self):
            raise IndexError('index %d is out of range' % idx)
        buf = self.encode(obj)
        lb = len(buf)
        if idx > 0 and self.sizes[idx]*WB > MAX_SHORT:
            self._add_to_freelist(idx)
        if lb <= MAX_SHORT:
            return self._add_short(buf, lb)
        # long string case
        return self._add_long(buf, lb, with_reuse=True)

    def close(self):
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
