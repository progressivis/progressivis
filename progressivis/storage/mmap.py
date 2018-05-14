"""
Numpy array using mmap that can resize without copy.
See https://stackoverflow.com/questions/20932361/resizing-numpy-memmap-arrays.
"""
from __future__ import absolute_import, division, print_function

import os
import os.path
from resource import getpagesize
import marshal
import shutil
from mmap import mmap
import logging

import numpy as np

from progressivis.core.utils import integer_types
from .base import StorageEngine, Dataset
from .hierarchy import GroupImpl, AttributeImpl


logger = logging.getLogger(__name__)

METADATA_FILE = ".metadata"
PAGESIZE = getpagesize()


class MMapDataset(Dataset):
    """
    Dataset implemented using the mmap file system function.
    Can grow as desired without needing any copy.
    """
    def __init__(self, path, name, shape=None, dtype=None, data=None, **kwds):
        "Create a MMapDataset."
        self._name = name
        self._filename = os.path.join(path, name)
        length = 0
        if dtype is not None:
            dtype = np.dtype(dtype)
        if data is not None:
            shape = data.shape
            if dtype is None:
                dtype = data.dtype
        if dtype is None:
            raise ValueError('dtype requied when no data is provided')
        self._dtype = dtype
        if dtype == OBJECT:
            self._strings = MMapDataset(path, name+"_strings", shape=shape, dtype=np.int8)
            dtype = np.int64
        else:
            self._strings = None
        
        if shape:
            length = 1
            for shap in shape:
                length *= shap
            length *= dtype.itemsize
        else:
            shape = (0,)

        last = max(0, length - 1)
        length = (last // PAGESIZE + 1) * PAGESIZE

        self._file = open(self._filename, 'wb+') # can raise many exceptions
        os.ftruncate(self._file.fileno(), length)
        self._buffer = mmap(self._file.fileno(), 0)

        if 'maxshape' in kwds:
            #TODO check if another dimension than 0 is growable to raise an exception
            del kwds['maxshape']
        if 'fillvalue' in kwds:
            self._fillvalue = kwds.pop('fillvalue')
            #print('fillvalue specified for %s is %s'%(self.base.dtype, self._fillvalue))
        else:
            if dtype == OBJECT:
                self._fillvalue = ''
            if np.issubdtype(dtype, np.int):
                self._fillvalue = 0
            elif np.issubdtype(dtype, np.bool):
                self._fillvalue = False
            else:
                self._fillvalue = np.nan
            #print('fillvalue for %s defaulted to %s'%(self.base.dtype, self._fillvalue))
        if kwds:
            logger.warning('Ignored keywords in MMapDataset: %s', kwds)
        self.base = np.frombuffer(self._buffer, dtype=dtype)

        if self.base.shape == shape:
            self.view = self.base
        else:
            self.view = self.base[:shape[0]]
            assert self.view.shape == shape
        if data is not None:
            self._fill(0, data)

        self._attrs = AttributeImpl()

    def _fill(self, data, start=0, end=None):
        if end is None:
            end = start + len(data)
        if self.base.dtype == OBJECT:
            for i, v in enumerate(data):
                self._set_value_at(start+i, v)
        else:
            self.view[start:end] = np.asarray(data)

    def _set_value_at(self, i, v):
        #TODO free current value
        if v is None:
            self.view[i] = -1
        else:
            data = v.encode('utf-8')
            offset = len(self._strings)
            self._strings.append(np.frombuffer(data, dtype=np.int8))
            self._strings.append([0])

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self._dtype

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
        shape = self.base.shape
        if isinstance(size, integer_types):
            length = 1
            for shap in shape[1:]:
                length *= shap
            length *= size
            shape = tuple([size]+list(shape[1:]))
        else:
            length = 1
            for shap in size:
                length *= shap
        length *= self.dtype.itemsize
        last = max(0, length - 1)
        length = (last // PAGESIZE + 1) * PAGESIZE
        self._buffer.resize(length)
        self.base = np.frombuffer(self._buffer, self.dtype)
        if self.base.shape != shape:
            self.base = self.base.reshape(shape)
        baseshape = np.array(self.base.shape)
        viewshape = self.view.shape
        size = np.asarray(size)
        if (size > baseshape).any():
            self.view = None
            newsize = []
            for shap, shape in zip(size, baseshape):
                if shap > shape:
                    shap = next_pow2(shap)
                newsize.append(shap)
            self.base = np.resize(self.base, tuple(newsize))
        # fill new areas with fillvalue
        if any(size > viewshape) and (size != 0).all():
            newarea = [np.s_[0:os] for os in viewshape]
            for i, oldsiz in enumerate(viewshape):
                siz = size[i]
                if siz > oldsiz:
                    newarea[i] = np.s_[oldsiz:siz]
                    self.base[tuple(newarea)] = self._fillvalue
                newarea[i] = np.s_[0:siz]
        else:
            newarea = [np.s_[0:s] for s in size]
        self.view = self.base[newarea]

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


OBJECT = np.dtype('O')

class MMapGroup(GroupImpl):
    """
    Group of mmap-based groups and datasets.
    """
    def __init__(self, name, parent=None):
        super(MMapGroup, self).__init__(name, parent=parent)
        self._directory = self.path()
        metadata = os.path.join(self._directory, METADATA_FILE)
        if os.path.exists(self._directory):
            if not os.path.isdir(self._directory):
                raise OSError('Cannot create group %s'%self._directory)
            if not os.path.isfile(metadata):
                raise ValueError('Cannot create group %s, "unsuitable directory'%
                                 self._directory)
            _read_attributes(self._attrs.attrs, metadata)
        else:
            os.mkdir(self._directory) # can raise exceptions
            _write_attributes(self._attrs.attrs, metadata)

    def path(self):
        "Return the path of the directory for that group"
        if self.parent is None:
            return self._name
        return os.path.join(self.parent.path(), self._name)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if name in self.dict:
            raise KeyError('name %s already defined' % name)
        chunks = kwds.pop('chunks', None)
        if chunks is None:
            chunklen = None
        elif isinstance(chunks, integer_types):
            chunklen = int(chunks)
        elif isinstance(chunks, tuple):
            chunklen = 1
            for dsize in chunks:
                chunklen *= dsize
        if dtype is not None:
            dtype = np.dtype(dtype)
        fillvalue = kwds.pop('fillvalue', None)
        if fillvalue is None:
            if dtype == OBJECT:
                fillvalue = ''
            else:
                fillvalue = 0
        if data is None:
            if shape is None:
                shape = (0,)
            arr = MMapDataset(self.path(),
                              name,
                              data=data,
                              shape=shape,
                              dtype=dtype,
                              fillvalue=fillvalue,
                              **kwds)
        self.dict[name] = arr
        return arr

    def _create_group(self, name, parent):
        return MMapGroup(name, parent=parent)

    def delete(self):
        "Delete the group and resources associated. Do it at your own risk"
        shutil.rmtree(self._directory)


class MMapStorageEngine(StorageEngine, MMapGroup):
    "StorageEngine for mmap-based storage"
    def __init__(self, root='mmap_storage'):
        """
        Create a storage manager from a specified root directory.
        """
        StorageEngine.__init__(self, "mmap")
        MMapGroup.__init__(self, root, None)

    def __contains__(self, name):
        return MMapGroup.__contains__(self, name)

def _read_attributes(attrs, filename):
    with open(filename, 'rb') as inf:
        dictionary = marshal.load(inf)
    if not isinstance(dictionary, dict):
        raise ValueError('metadata contains invalid data %s'%filename)
    attrs.clear()
    attrs.update(dictionary)
    return attrs

def _write_attributes(attrs, filename):
    with open(filename, 'wb') as outf:
        marshal.dump(attrs, outf)
