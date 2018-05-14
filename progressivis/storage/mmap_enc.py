"""
Use mmap to manage strings
"""
import mmap
import os.path
from resource import getpagesize

import six

import numpy as np

PAGESIZE = getpagesize()


class MMapObject(object):
    def __init__(self, filename):
        if os.path.isfile(filename):
            file = open(filename, "r+b")
        else:
            file = open(filename, "wb")
        self.mmap = mmap.mmap(file.fileno(), 0)
        self.sizes = None
        self.view = None
        if not self.mmap:
            self._allocate(PAGESIZE)
            self.sizes[0] = 0
        else:
            self.sizes = np.frombuffer(mmap, np.unint32)

    def _allocate(self, size):
        assert (size%PAGESIZE) == 0
        self.mmap.resize(size)
        self.sizes = np.frombuffer(mmap, np.unint32)

    def resize(self, size):
        mod = size%4
        if mod:
            size += 4 - mod
        size *= 4
        if size > len(self.mmap):
            pages = (size + PAGESIZE-1) // PAGESIZE * (1+PAGESIZE)
            self._allocate(pages)

    def __len__(self):
        return self.sizes[0]

    def encode(self, obj):
        assert isinstance(obj, six.text_type)
        return obj.encode('utf-8')

    def decode(self, buf):
        return buf.decode('utf-8')

    def add(self, obj):
        buf = self.encode(obj)
        size = len(buf + 4)
        ret = len(self)
        self.resize(ret+size)
        self.sizes[ret] = size
        self.mmap[(ret+1)*4:] = buf
        self.sizes[0] = ret+1
        return ret

    def get(self, idx):
        if idx < 0 or idx > len(self):
            raise IndexError('index %d is out of range' % idx)
        size = self.sizes[idx]
        buf = self.mmap[(idx+1)*4:(idx+1)*4+size]
        return self.decode(buf)

    def __getitem__(self, idx):
        return self.get(idx)

    def __setitem__(self, idx, obj):
        if idx < 0 or idx > len(self):
            raise IndexError('index %d is out of range' % idx)
        buf = self.encode(obj)
        size = len(buf) + 4
        if size <= self.sizes[idx]:
            self.mmap[(idx+1)*4:] = buf
            self.sizes[idx] = size
        else:
            self.sizes[idx] = size
        self.mmap[(ret+1)*4:] = buf
