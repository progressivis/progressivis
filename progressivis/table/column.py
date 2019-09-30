from __future__ import absolute_import, division, print_function

from collections import Iterable
import logging

import numpy as np

from progressivis.storage import Group
from progressivis.core.utils import integer_types, get_random_name
from progressivis.utils.fast import indices_to_slice
from .column_base import BaseColumn
from .dshape import dshape_to_h5py, np_dshape, dshape_create
from . import metadata

logger = logging.getLogger(__name__)

__all__ = ["Column"]


class Column(BaseColumn):
    def __init__(self, name, index, base=None, storagegroup=None,
                 dshape=None, fillvalue=None,
                 shape=None, chunks=None, data=None, indices=None):
        """Create a new column.

        if index is None and self.index return None, a new index and dataset are created.
        """
        super(Column, self).__init__(name, index, base=base)
        if storagegroup is None:
            if index is not None:
                storagegroup = index.storagegroup
            else:
                storagegroup = Group.default(name=get_random_name('column_'))
        self._storagegroup = storagegroup
        self.dataset = None
        self._dshape = None
        if self.index is None:
            if data is not None: # check before creating everything
                l = len(data)
                if indices and l != len(indices):
                    raise ValueError('Bad index length (%d/%d)', len(indices), l)
            self._complete_column(dshape, fillvalue, shape, chunks, data)
            if data is not None:
                self.append(data, indices)

    @property
    def storagegroup(self):
        return self._storagegroup

    def _allocate(self, count, indices=None):
        indices = self.index._allocate(count, indices)
        newsize = self.index.size
        self.resize(newsize)
        return indices

    def append(self, data, indices=None):
        if data is None:
            return
        l = len(data)
        is_array = isinstance(data, (np.ndarray, list, BaseColumn))
        if indices is not None and len(indices) != l:
            raise ValueError('Bad index length (%d/%d)', len(indices), l)
        indices = self._allocate(len(data), indices)
        if is_array:
            indices = indices_to_slice(indices)
            self[indices] = data[0:l]
        else:
            for i in range(l):
                self[indices[i]] = data[i]

    def add(self, value, index=None):
        if index is None:
            index = self._allocate(1)
        else:
            index = self._allocate(1, [index])
        index = index[0]
        self[index] = value

    def _complete_column(self, dshape, fillvalue, shape, chunks, data):
        if dshape is None:
            if data is None:
                raise ValueError('Cannot create column "%s" without dshape nor data', self.name)
            elif hasattr(data, 'dshape'):
                dshape = data.dshape
            elif hasattr(data, 'dtype'):
                dshape = np_dshape(data)
            else:
                raise ValueError('Cannot create column "%s" from data %s', self.name, data)
        dshape = dshape_create(dshape) # make sure it is valid 
        if shape is None and data is not None:
            shape=dshape.shape
        from .column_id import IdColumn
        self._index = IdColumn(self._storagegroup)
        self._index.create_dataset()
        self.create_dataset(dshape=dshape, fillvalue=fillvalue,
                            shape=shape, chunks=chunks)

    def create_dataset(self, dshape, fillvalue, shape=None, chunks=None):
        dshape = dshape_create(dshape) # make sure it is valid 
        self._dshape = dshape
        dtype = dshape_to_h5py(dshape)
        if shape is None:
            maxshape = (None,)
            shape=dshape.shape
            shape = (0,)
            if chunks is None:
                chunks = (128*1024/np.dtype(dtype).itemsize,)
        else:
            maxshape = tuple([None]+list(shape))
            shape=tuple([0]+[0 if s is None else s for s in shape])
            if chunks is None:
                dims = list(shape)[1:]
                # count 16 entries for each variable dimension
                #TODO find a smarter way to allocate chunk size
                chunks = [64]
                for d in dims:
                    chunks.append(d if d != 0 else 64)
                chunks = tuple(chunks)
        if not isinstance(chunks, tuple):
            chunks = tuple([chunks])
        logger.debug('column=%s, shape=%s, chunks=%s, dtype=%s',
                     self._name, shape, chunks, str(dtype))

        group = self._storagegroup
        if self.name in group:
            logger.warning('Deleting dataset named "%s"', self.name)
            del group[self.name]
        dataset = group.create_dataset(self.name, 
                                       shape=shape,
                                       dtype=dtype,
                                       chunks=chunks,
                                       maxshape=maxshape,
                                       fillvalue=fillvalue)
        dataset.attrs[metadata.ATTR_COLUMN] = True
        dataset.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        dataset.attrs[metadata.ATTR_DATASHAPE] = str(dshape)
        self.dataset = dataset
        return dataset

    def load_dataset(self, dshape, nrow, shape=None, is_id=False):
        self._dshape = dshape
        if shape is None:
            shape = (nrow,)
        else:
            shape=tuple([nrow]+shape)
        dtype = dshape_to_h5py(dshape)
        group = self._storagegroup
        if is_id and not self.name in group: # for lazy ID column creation 
            return None
        dataset = group.require_dataset(self.name,
                                        dtype=dtype,
                                        shape=shape)
        assert dataset.attrs[metadata.ATTR_COLUMN] == True \
          and dataset.attrs[metadata.ATTR_VERSION] == metadata.VALUE_VERSION \
          and dataset.attrs[metadata.ATTR_DATASHAPE] == str(dshape)
        self.dataset = dataset
        return dataset

    @property
    def chunks(self):
        return self.dataset.chunks

    @property
    def shape(self):
        return self.dataset.shape

    def set_shape(self, shape):
        if not isinstance(shape, list):
            shape = list(shape)
        myshape = list(self.shape[1:])
        if len(myshape)!=len(shape):
            raise ValueError('Specified shape (%s) does not match colum shape (%s)'%(shape,myshape))
        if myshape==shape:
            return
        logger.debug('Changing size from (%s) to (%s)',myshape,shape)
        self.dataset.resize(tuple([len(self)]+shape))

    @property
    def maxshape(self):
        return self.dataset.maxshape

    @property
    def dtype(self):
        return self.dataset.dtype

    @property
    def dshape(self):
        return self._dshape

    @property
    def size(self):
        return self.dataset.size

    def __len__(self):
        return len(self.index)

    @property
    def fillvalue(self):
        return self.dataset.fillvalue

    @property
    def value(self):
        return self.dataset[:]

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            index = list(index)
        try: # EAFP
            return self.dataset[index]
        except TypeError:
            if isinstance(index, Iterable):
                return np.array([self.dataset[e] for e in index])
            raise

    def read_direct(self, array, source_sel=None, dest_sel=None):
        if hasattr(self.dataset, 'read_direct'):
            if isinstance(source_sel, np.ndarray) and source_sel.dtype==np.int:
                source_sel = list(source_sel)
#            if is_fancy(source_sel):
#                source_sel = fancy_to_mask(source_sel, self.shape)
            return self.dataset.read_direct(array, source_sel, dest_sel)
        else:
            return super(Column, self).read_direct(array, source_sel, dest_sel)

    def __setitem__(self, index, val):
        if isinstance(index, integer_types):
            self.dataset[index] = val 
        else:
            if hasattr(val, 'values') and isinstance(val.values, np.ndarray):
                val = val.values
            if not hasattr(val, 'shape'):
                val = np.asarray(val, dtype=self.dtype)

            if isinstance(index, np.ndarray) and index.dtype==np.int:
                index = list(index)
            try:
                self.dataset[index] = val
            except TypeError as e:
                #TODO distinguish between unsupported fancy indexing and real error
                if isinstance(index, Iterable):
                    if isinstance(val, (np.ndarray, list)):
                        for e in index:
                            self.dataset[e] = val[e]
                    else:
                        for e in index:
                            self.dataset[e] = val
                else:
                    raise
        self.index.touch(index)

    def resize(self, newsize):
        assert isinstance(newsize, integer_types)
        if self.size==newsize:
            return
        shape = self.shape
        if len(shape)==1:
            self.dataset.resize((newsize,))
        else:
            shape = tuple([newsize]+list(shape[1:]))
            self.dataset.resize(shape)
        if self.index is not None:
            self.index.resize(newsize)

    def __delitem__(self, index):
        del self.index[index]
        self.dataset[index] = self.fillvalue # cannot propagate that to other columns
        self.dataset.resize(self.index.size)


