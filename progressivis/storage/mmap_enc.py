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

PAGESIZE = getpagesize()
WB = 4

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
            

    def _allocate(self, size):
        self.mmap.resize(size*PAGESIZE)
        self.sizes = np.frombuffer(self.mmap, np.uint32)

    def resize(self, size):
        mod = size%WB
        if mod:
            size += WB - mod
        size *= WB
        if size > len(self.mmap):
            pages = (size + PAGESIZE-1) // PAGESIZE * (1+PAGESIZE)
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


    def add(self, obj):
        buf = self.encode(obj)
        lb = len(buf)
        bufsize = lb + WB - lb%WB
        assert bufsize % 4 == 0
        ret = len(self) 
        off = _ofs(ret)
        self.resize(off+WB+bufsize)        
        self.sizes[ret] = bufsize//WB
        self.mmap[off:off+lb] = buf
        self.mmap[off+lb:off+bufsize] = b'\x00'*(bufsize-lb)  #np.zeros(bufsize-lb, dtype=np.uint8)
        self.sizes[0] = ret + 1 + bufsize//4
        return ret

        
    def set_at(self, idx, obj):
        if idx < 0 or idx > len(self):
            raise IndexError('index %d is out of range' % idx)
        if idx == 0:
            return self.add(obj)
        buf = self.encode(obj)
        lb = len(buf)
        size = lb + 4 - lb%4
        if size <= self.sizes[idx]:
            off = _ofs(idx)
            self.mmap[off:off+lb] = buf
            self.mmap[off+lb:off+size] = b'\x00'*(size-lb) #np.zeros(size-lb, dtype=np.uint8)
            #self.sizes[idx] = size
            return idx
        return self.add(obj)

    def close(self):
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
