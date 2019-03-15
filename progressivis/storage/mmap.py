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
import six
import numpy as np

from progressivis.core.utils import integer_types, get_random_name
from .base import StorageEngine, Dataset
from .hierarchy import GroupImpl, AttributeImpl
from .mmap_enc import MMapObject
import sys
import json
logger = logging.getLogger(__name__)

METADATA_FILE = ".metadata"
PAGESIZE = getpagesize()
FACTOR = 1


def _shape_len(shape):
    length = 1
    for dim in shape:
        length *= dim
    return length

class MMapDataset(Dataset):
    """
    Dataset implemented using the mmap file system function.
    Can grow as desired without needing any copy.
    """
    #datasets = []
    def __init__(self, path, name, shape=None, dtype=None, data=None, **kwds):
        "Create a MMapDataset."
        self._name = name
        self._filename = os.path.join(path, name)
        self._metafile = os.path.join(path, METADATA_FILE+'.'+name)
        self._attrs = AttributeImpl()        
        length = 0
        if dtype is not None:
            dtype = np.dtype(dtype)
        if data is not None:
            shape = data.shape
            if dtype is None:
                dtype = data.dtype
        if dtype is None:
            raise ValueError('dtype required when no data is provided')
        self._dtype = dtype

        if shape:
            length = 1
            for shap in shape:
                length *= shap
            length *= dtype.itemsize
        else:
            shape = (0,)
            length = 0
        if dtype == OBJECT:
                # NB: shape=(1,) means single empty string, offset=0, so shared by all
                # entries in self.base
            #self._strings = MMapDataset(path, name+"_strings", shape=(1,), dtype=np.int8)
            self._strings = MMapObject(self._filename+"_strings")
            if data is not None:
                pass # TODO: ...
            dtype = np.dtype(np.int64)
        else:
            self._strings = None
        nb_item = length//dtype.itemsize
        last = max(0, length - 1)
        length = (last // PAGESIZE + 1) * PAGESIZE * 10
        if os.path.exists(self._filename):
            self._file = open(self._filename, 'ab+')
            _read_attributes(self._attrs.attrs, self._metafile)
        else:
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
            if np.issubdtype(dtype, np.int):
                self._fillvalue = 0
            elif np.issubdtype(dtype, np.bool):
                self._fillvalue = False
            else:
                self._fillvalue = np.nan
            #print('fillvalue for %s defaulted to %s'%(self.base.dtype, self._fillvalue))
        if kwds:
            logger.warning('Ignored keywords in MMapDataset: %s', kwds)
        self.base = np.frombuffer(self._buffer, dtype=dtype, count=nb_item)
        if self.base.shape != shape:
            self.base = self.base.reshape(shape)
        self.view = self.base
        #if self.base.shape == shape:
        #    self.view = self.base
        #else:
        #    self.view = self.base[:shape[0]]
        assert self.view.shape == shape
        if data is not None:
            self._fill(0, data)
        #MMapDataset.datasets.append(self)

    def _fill(self, data, start=0, end=None):
        assert self._buffer is not None
        if end is None:
            end = start + len(data)
        if self.base.dtype == OBJECT:
            for i, v in enumerate(data):
                self._set_value_at(start+i, v)
        else:
            self.view[start:end] = np.asarray(data)

    def append(self, val):
        assert self._buffer is not None        
        #assert isinstance(val, bytes)
        assert isinstance(self.shape, tuple) and len(self.shape) == 1
        assert self.dtype == np.int8
        last = len(self.view)
        lval = len(val)
        self.resize(last+lval)
        self.base[last:last+lval] = val

    def release(self, ids):
        if  self._strings is None:
            return
        if isinstance(ids, integer_types):
            offset = self.view[ids]
            if offset == -1:
                return None
            return self._strings.release(offset)
        elif isinstance(ids, slice):
            stop_ = ids.stop if ids.stop is not None else len(self.view)
            ids = range(*ids.indices(stop_))
        for k, i in enumerate(ids):
            offset = self.view[i]
            if offset == -1:
                continue
            self._strings.release(offset)

    def _set_value_at(self, i, v):
        #TODO free current value
        if v is None:
            self.view[i] = -1
        else:
            self.view[i] = self._strings.add(v)

    def close(self, recurse=True):
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if recurse and self._strings is not None:
            self._strings.close()
        _write_attributes(self._attrs.attrs, self._metafile)
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
        assert self._buffer is not None
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
            shape = size
        dtype_ = np.dtype(np.int64) if self.dtype==OBJECT else self.dtype
        nb_item = length
        length *= dtype_.itemsize
        last = max(0, length - 1)
        length = (last // PAGESIZE + 1) * PAGESIZE *10
        if length != len(self._buffer):
            self._buffer.resize(length)
        self.base = np.frombuffer(self._buffer, dtype=dtype_, count=nb_item) # returns an 1D array
        if self.base.shape != shape:
            self.base = self.base.reshape(shape)
        baseshape = np.array(self.base.shape)
        viewshape = self.view.shape
        size = np.asarray(shape)
        if (size > baseshape).any():
            self.view = None
            newsize = []
            for shap, shape in zip(size, baseshape):
                if shap > shape:
                    shap = next_pow2(shap)
                newsize.append(shap)
            self.base = np.resize(self.base, tuple(newsize))
        # fill new areas with fillvalue
        fillvalue_ = 0 if self.dtype==OBJECT else self._fillvalue
        if any(size > viewshape) and (size != 0).all():
            newarea = [np.s_[0:os] for os in viewshape]
            for i, oldsiz in enumerate(viewshape):
                siz = size[i]
                if siz > oldsiz:
                    newarea[i] = np.s_[oldsiz:siz]
                    self.base[tuple(newarea)] = fillvalue_
                newarea[i] = np.s_[0:siz]
        else:
            newarea = [np.s_[0:s] for s in size]
        self.view = self.base[tuple(newarea)]

    def __getitem__(self, args):
        if self.dtype != OBJECT:
            return self.view[args]
        if isinstance(args, integer_types):
            offset = self.view[args]
            if offset == -1:
                return None
            return self._strings.get(offset)
        elif isinstance(args, slice):
            stop_ = args.stop if args.stop is not None else len(self.view)
            args = range(*args.indices(stop_))
        res = np.empty((len(args),),dtype=OBJECT)
        for k, i in enumerate(args):
            offset = self.view[i]
            if offset == -1:
                res[k] = None
                continue
            res[k] = self._strings.get(offset)
        return np.array(res, dtype=OBJECT)

    def __setitem__(self, args, val):
        if self.dtype != OBJECT:
            self.view[args] = val
            return
        if isinstance(args, integer_types):
            args = (args,)
            val = (val,)
        elif isinstance(args, slice):
            stop_ = args.stop if args.stop is not None else len(self.view)
            args = range(*args.indices(stop_))
        for i, k in enumerate(args):
            self.view[k]  = self._strings.set_at(self.view[k], val[i])
            #if new_idx != k:
            #    view[k] = new_idx

    def __len__(self):
        return self.view.shape[0]

    @property
    def attrs(self):
        return self._attrs

    @property
    def name(self):
        return self._name


OBJECT = np.dtype('O')
ROOT_NAME = "mmap_storage_py2" if six.PY2 else "mmap_storage"
class MMapGroup(GroupImpl):
    """
    Group of mmap-based groups and datasets.
    """
    #all_instances = []
    def __init__(self, name=None, parent=None):
        if name is None:
            name = get_random_name("mmapstorage_")
        super(MMapGroup, self).__init__(name, parent=parent)
        self._directory = self.path()
        metadata = os.path.join(self._directory, METADATA_FILE)
        self._metadata = metadata
        if os.path.exists(self._directory):
            if not os.path.isdir(self._directory):
                raise OSError('Cannot create group %s'%self._directory)
            if not os.path.isfile(metadata):
                raise ValueError('Cannot create group %s, "unsuitable directory'%
                                 self._directory)
            _read_attributes(self._attrs.attrs, metadata)
        else:
            os.makedirs(self._directory) # can raise exceptions
            _write_attributes(self._attrs.attrs, metadata)
        if parent is not None:
            if name in parent.dict:
                raise ValueError('Cannot create group {}, already exists'.format(name))
            parent.dict[name] = self
            
    def path(self):
        "Return the path of the directory for that group"
        if self.parent is None:
            return self._name
        return os.path.join(self.parent.path(), self._name)
    def has_files(self):
        if not os.path.isdir(self._directory):
            return False
        if not os.path.isfile(self._metadata):
            return False
        return True
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
        if os.path.exists(self._directory):
            shutil.rmtree(self._directory)
        if self.parent is not None and self in self.parent.dict:
            del self.parent.dict[self]

            
    def delete_children(self):
        for f in os.listdir(self._directory):
            if f == '.metadata':
                continue
            child = os.path.join(self._directory, f)
            if os.path.isdir(child):
                shutil.rmtree(child)
            else:
                os.unlink(child)
            
    def close_all(self):
        for ds in self.dict.values():
            if isinstance(ds, MMapDataset):
                ds.close()
            elif isinstance(ds, MMapGroup):
                ds.close_all()
        metadata = os.path.join(self._directory, METADATA_FILE)
        _write_attributes(self._attrs.attrs, metadata)

    def release(self, ids):
        for ds in self.dict.values():
            ds.release(ids)


    
            

class MMapStorageEngine(StorageEngine, MMapGroup):
    "StorageEngine for mmap-based storage"
    def __init__(self, root=ROOT_NAME):
        """
        Create a storage manager from a specified root directory.
        """
        StorageEngine.__init__(self, "mmap")
        MMapGroup.__init__(self, root, None)

        
    @staticmethod
    def create_group(name=None, create=True):
        root = StorageEngine.engines()['mmap']
        if name in root.dict:
            if create:
                name = get_random_name(name[:16]+'_')
            else:
                return root.dict[name]
            # TODO : specify this behaviour
            #grp = root.dict[name]
            #if not isinstance(grp, MMapGroup):
            #     raise ValueError("{} already exists and is not a group".format(name))
            #return grp
        return MMapGroup(name, parent=root)
    
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
