from __future__ import absolute_import, division, print_function

from progressivis.core.utils import integer_types
from progressivis.core.fast import next_pow2
from .base import StorageEngine, Dataset
from .hierarchy import GroupImpl, AttributeImpl

import numpy as np
#import pptable as pypmm

import logging
logger = logging.getLogger(__name__)

class PPTableDataset(Dataset):
    def __init__(self, name, chunks, shape=None, data=None, **kwds):
        self._name = name
        self._chunks = chunks

        #Copies the vector given as input
        if data is not None:
            #Only linear chunks at the moment
            chunk_shape = np.array(data.shape,dtype=np.uint)
            chunk_shape[:] = 1
            chunk_shape[len(chunk_shape)-1] = 4000
            # print(dtype)
            ###### COPY IS MISSING ######
            self.base = pypmm.ChunkedMemory()
            self.base.initialize(np.array(data.shape,dtype=np.uint),chunk_shape)
            #NP not working for multidim...
            if data.ndim != 1:
                print(data.shape)
                print(np.zeros(data.shape))
                raise Warning("NOT IMPLEMENTED")
        else:
            raise Warning ("empty table... chunk size?")
            if shape is None:
                raise Warning("NOT IMPLEMENTED")
            self.base = pypmm.ChunkedMemory()
            self.base.initialize(shape,shape)

        #maxshape is not supported by the pypmm class (it is unlimited)
        if 'maxshape' in kwds:
            del kwds['maxshape']

        #setting the fill value
        if 'fillvalue' in kwds:
            self._fillvalue = kwds.pop('fillvalue')
            #print('fillvalue specified for %s is %s'%(self.base.dtype, self._fillvalue))
        else:
            if isinstance(self.base.dtype, np.int):
                self._fillvalue = 0
                raise Warning("NOT IMPLEMENTED")
            else:
                self._fillvalue = np.nan
            #print('fillvalue for %s defaulted to %s'%(self.base.dtype, self._fillvalue))

        #printing the remaining keywords
        if kwds:
            logger.warning('Ignored keywords in PPTableDataset: %s', kwds)

        self.view = self.base #TOREMOVE
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
    def size(self):
        return self.view.shape[0]

    @property
    def chunks(self):
        return self._chunks

    def resize(self, size, axis=None):
        self.base.resize(size)

    def __getitem__(self, args):
        # print(args)
        return self.view[args]

    def __setitem__(self, args, val):
        # print(str(args) + " = " + str(val))
        self.view[args] = val

    def __len__(self):
        return self.view.shape[0]

    @property
    def attrs(self):
        return self._attrs

    @property
    def name(self):
        return self._name


class PPTableGroup(GroupImpl):
    def __init__(self, name, parent=None):
        super(PPTableGroup, self).__init__(name, parent=parent)

    def create_dataset(self, name, shape=None, dtype=None, data=None, fillvalue=None, chunks=None, maxshape=None, **kwds):
        # print("CHUUUUUUUUUNKS")
        # print(chunks)
        # print("SHAAAAAAAAAAAPE")
        # print(shape)
        # print("MAAAAAAAAAAAAAAAAAX")
        # print(maxshape)

        _ = maxshape
        if name in self.dict:
            raise KeyError('name %s already defined', name)
        if chunks is None:
            chunklen=None
        elif isinstance(chunks, integer_types):
            chunklen=int(chunks)
            chunks=(chunks,)
        elif isinstance(chunks, tuple):
            chunklen=1
            for m in chunks:
                chunklen *= m
        if dtype is not None:
            dtype = np.dtype(dtype)
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

        arr = PPTableDataset(name,
                             chunks=chunks,
                             data=data,
                             shape=shape,
                             dtype=dtype,
                             fillvalue=fillvalue,
                             **kwds)
        arr._chunks = chunks
        self.dict[name] = arr
        return arr

    def _create_group(self, name, parent):
        return PPTableGroup(name, parent=parent)

class PPTableStorageEngine(StorageEngine, PPTableGroup):
    def __init__(self):
        StorageEngine.__init__(self, "pptable")
        PPTableGroup.__init__(self, '/', None)

    def open(self, name, flags, **kwds):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def __contains__(self, name):
        return PPTableGroup.__contains__(self, name)
