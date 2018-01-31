from __future__ import absolute_import, division, print_function

from .base import StorageEngine, Dataset, Attribute
from .hierarchy import GroupImpl
from ..utils import integer_types
from progressivis.table.dshape import VSTRING, OBJECT

import bcolz
from bcolz.attrs import attrs

import numpy as np

# For now (April 21st 2017), bcolz has fatal bugs related to strings.
# I have submitted to issues to the bcolz github project, waiting for them to
# be solved:
# Issues are #342 and #343:
# https://github.com/Blosc/bcolz/issues/342
# https://github.com/Blosc/bcolz/issues/343


class BCOLZGroup(GroupImpl):
    def __init__(self, name, parent=None):
        super(BCOLZGroup, self).__init__(name, parent=parent)

    def _create_attribute(self, dict_values=None):
        return BCOLZAttrs(None, 'w')

    def create_dataset(self, name, shape=None, dtype=None, data=None, fillvalue=None, chunks=None, maxshape=None, **kwds):
        _ = maxshape
        if name in self.dict:
            raise KeyError('name %s already defined', name)
        if fillvalue is None:
            fillvalue=0
        if chunks is None:
            chunklen=None
        elif isinstance(chunks, integer_types):
            chunklen=int(chunks)
        elif isinstance(chunks, tuple):
            chunklen=1
            for m in chunks:
                chunklen *= m
        if dtype is VSTRING:
            dtype = OBJECT
            #print("Fixing VSTRING")
            fillvalue=''
        elif dtype is not None:
            dtype = np.dtype(dtype)
        if data is None:
            if shape is None:
                data=np.ndarray([], dtype=dtype)
            elif fillvalue==0:
                data=np.zeros(shape, dtype=dtype)
            else:
                data=np.full(shape, fillvalue, dtype=dtype)

        arr = BCOLZDataset(data,
                           cparams=self._cparams,
                           dtype=dtype,
                           dflt=fillvalue,
                           chunklen=chunklen,
                           mode='w',
                           **kwds)
        self.dict[name] = arr
        return arr

    def _create_group(self, name, parent):
        return BCOLZGroup(name, parent=parent)

class BCOLZStorageEngine(StorageEngine, BCOLZGroup):
    def __init__(self):        
        StorageEngine.__init__(self, "bcolz")
        BCOLZGroup.__init__(self, '/', None)

    def open(self, name, flags, **kwds):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def __contains__(self, name):
        return BCOLZGroup.__contains__(self, name)

class BCOLZDataset(bcolz.carray):
    @property
    def fillvalue(self):
        return self.dflt

    def resize(self, size, axis=None):
        _ = axis
        if isinstance(size, tuple):
            size = size[0]
        super(BCOLZDataset, self).resize(int(size))

class BCOLZAttrs(attrs, Attribute):
    def __contains__(self, key):
        return key in self.attrs

    def get(self, key, default=None):
        return self.attrs.get(key, default)



Dataset.register(BCOLZDataset)
