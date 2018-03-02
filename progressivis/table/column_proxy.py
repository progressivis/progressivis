from __future__ import absolute_import, division, print_function
from .column_base import BaseColumn

class ColumnProxy(BaseColumn):
    def __init__(self, base, index=None, name=None):
        super(ColumnProxy, self).__init__(name, base=base, index=index)
    
    @property
    def chunks(self):
        return self._base.chunks

    @property
    def shape(self):
        return self._base.shape

    def set_shape(self, shape):
        self._base.set_shape(shape)

    def __delitem__(self, index):
        raise RuntimeError('Cannot delete in %s'%type(self))

    @property
    def maxshape(self):
        return self._base.maxshape
    
    @property
    def dtype(self):
        return self._base.dtype

    @property
    def dshape(self):
        return self._base.dshape

    @property
    def size(self):
        return self._base.size

    def __len__(self):
        return len(self._base)

    @property
    def fillvalue(self):
        return self._base.fillvalue

    @property
    def value(self):
        return self._base[:]

    @property
    def values(self):
        return self.value
    
    def __getitem__(self, index):
        return self._base[index]

    def __setitem__(self, index, val):
        self._base[index] = val

    def resize(self, newsize):
        self._base.resize(newsize)

    def tolist(self):
        return list(self.values)


