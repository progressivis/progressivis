from __future__ import absolute_import, division, print_function

import logging

import numpy as np

from progressivis.core.utils import integer_types
from progressivis.utils.fast import next_pow2
from .base import StorageEngine, Dataset
from .hierarchy import GroupImpl, AttributeImpl

logger = logging.getLogger(__name__)

class NumpyDataset(Dataset):
    def __init__(self, name, shape=None, dtype=None, data=None, **kwds):
        self._name = name
        if data is not None:
            self.base = np.array(data, dtype=dtype)
        else:
            self.base = np.empty(shape=shape, dtype=dtype)
        if 'maxshape' in kwds:
            del kwds['maxshape']
        if 'fillvalue' in kwds:
            self._fillvalue = kwds.pop('fillvalue')
            #print('fillvalue specified for %s is %s'%(self.base.dtype, self._fillvalue))
        else:
            if isinstance(self.base.dtype, np.int):
                self._fillvalue = 0
            else:
                self._fillvalue = np.nan
            #print('fillvalue for %s defaulted to %s'%(self.base.dtype, self._fillvalue))
        if kwds:
            logger.warning('Ignored keywords in NumpyDataset: %s', kwds)
        self.view = self.base
        self._attrs = AttributeImpl()

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self.view.dtype

    @property
    def maxshape(self):
        return self.view.shape

    @property
    def fillvalue(self):
        return self._fillvalue

    @property
    def chunks(self):
        return self.view.shape

    @property
    def size(self):
        return self.view.shape[0]

    def resize(self, size, axis=None):
        if isinstance(size, integer_types):
            size = np.array(tuple([size]+list(self.base.shape[1:])))
        else:
            size = np.array(size)
        baseshape = np.array(self.base.shape)
        viewshape = self.view.shape
        if ((size > baseshape).any()):
            self.view = None
            newsize = []
            for s,shape in zip(size, baseshape):
                if s > shape:
                    s = next_pow2(s)
                newsize.append(s)
            self.base = np.resize(self.base, tuple(newsize))
        # fill new areas with fillvalue
        if ((size > viewshape).any() and (size!=0).all()):
            newarea = [np.s_[0:os] for os in viewshape]
            for i in range(len(viewshape)):
                s = size[i]
                os = viewshape[i]
                if s > os:
                    newarea[i] = np.s_[os:s]
                    self.base[tuple(newarea)] = self._fillvalue
                newarea[i] = np.s_[0:s]
        else:
            newarea = [np.s_[0:s] for s in size]
        self.view = self.base[tuple(newarea)]
                
    def __getitem__(self, args):
        return self.view[args]

    def __setitem__(self, args, val):
        self.view[args] = val

    def __len__(self):
        return self.view.shape[0]

    @property
    def attrs(self):
        return self._attrs

    @property
    def name(self):
        return self._name

        
class NumpyGroup(GroupImpl):
    def __init__(self, name='numpy', parent=None):
        super(NumpyGroup, self).__init__(name, parent=parent)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if name in self.dict:
            raise KeyError('name %s already defined', name)
        chunks = kwds.pop('chunks', None)
        if chunks is None:
            chunklen=None
        elif isinstance(chunks, integer_types):
            chunklen=int(chunks)
        elif isinstance(chunks, tuple):
            chunklen=1
            for m in chunks:
                chunklen *= m
        if dtype is not None:
            dtype = np.dtype(dtype)
        fillvalue = kwds.pop('fillvalue', None)
        if fillvalue is None:
            if dtype==np.object:
                fillvalue=''
            else:
                fillvalue=0
        if data is None:
            if shape is None:
                data=np.ndarray([], dtype=dtype)
                shape = data.shape
            elif fillvalue==0:
                data=np.zeros(shape, dtype=dtype)
            else:
                data=np.full(shape, fillvalue, dtype=dtype)

        arr = NumpyDataset(name,
                           data=data,
                           shape=shape,
                           dtype=dtype,
                           fillvalue=fillvalue,
                           **kwds)
        self.dict[name] = arr
        return arr

    def _create_group(self, name, parent):
        return NumpyGroup(name, parent=parent)

class NumpyStorageEngine(StorageEngine, NumpyGroup):
    def __init__(self):        
        StorageEngine.__init__(self, "numpy")
        NumpyGroup.__init__(self, '/', None)

    def open(self, name, flags, **kwds):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def __contains__(self, name):
        return NumpyGroup.__contains__(self, name)
    @staticmethod
    def create_group(name=None, create=True):
        _ = create # for pylint
        return NumpyGroup(name)
