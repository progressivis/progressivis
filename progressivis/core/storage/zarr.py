from __future__ import absolute_import, division, print_function

from .base import StorageEngine, Group, Attribute, Dataset
from ..config import get_option

import zarr
from zarr.attrs import Attributes
from zarr.storage import init_group, contains_group
import numcodecs
zarr.codecs.codec_registry[numcodecs.Pickle.codec_id] = numcodecs.Pickle
zarr.codecs.codec_registry[numcodecs.MsgPack.codec_id] = numcodecs.MsgPack
from collections import Iterable
import numpy as np

# For now (April 21st 2017), zarr misses two important features: fancy indexing and Boolean indexing.
# These two features are scheduled for inclusion in future releases of zarr.


Group.register(zarr.Group)
Attribute.register(Attributes)
Dataset.register(zarr.Array)

class ZARRGroup(zarr.Group):
    def __init__(self, store, **kwds):
        super(ZARRGroup, self).__init__(store, **kwds)

    def create_dataset(self, name, shape=None, dtype=None, data=None, maxshape=None, **kwds):
        _ = maxshape
        if kwds.get('compression') is None:
            kwds.update(get_option('storage.zarr.filter', {}))
        filters = kwds.get('filters', [])
        filters.append(numcodecs.MsgPack())
        kwds['filters'] = filters
        return super(ZARRGroup, self).create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype=None, exact=False, **kwds):
        if kwds.get('compression') is None:
            kwds.update(get_option('storage.zarr.filter', {}))
        return super(ZARRGroup, self).require_dataset(name, shape=shape, exact=exact, **kwds)

    # override
    def _create_group_nosync(self, name, overwrite=False):
        path = self._item_path(name)

        # create terminal group
        init_group(self._store, path=path, chunk_store=self._chunk_store,
                   overwrite=overwrite)

        return ZARRGroup(self._store, path=path, read_only=self._read_only,
                         chunk_store=self._chunk_store,
                         synchronizer=self._synchronizer)
        
    # override
    def _require_group_nosync(self, name, overwrite=False):
        path = self._item_path(name)

        # create terminal group if necessary
        if not contains_group(self._store, path):
            init_group(store=self._store, path=path,
                       chunk_store=self._chunk_store,
                       overwrite=overwrite)

        return ZARRGroup(self._store, path=path, read_only=self._read_only,
                         chunk_store=self._chunk_store,
                         synchronizer=self._synchronizer)


class ZARRStorageEngine(StorageEngine):
    def __init__(self):
        super(ZARRStorageEngine, self).__init__("zarr")
        self.store = zarr.DictStore()
        init_group(self.store)
        self._zarr = ZARRGroup(self.store)

    @property
    def zarr(self):
        return self._zarr

    def set_default(self, f):
        assert isinstance(f, zarr.Group)
        if self._zarr is not None:
            self.close()
        self._zarr = f

    def open(self, name, flags, **kwds):
        if self.store is not None:
            self.close()
        if name is None:
            self.store = zarr.DictStore()
        elif name.endsWith('/'):
            self.store = zarr.DirectoryStore(name)
        else:
            self.store = zarr.ZipStore(name, flags)
        self._zarr = zarr.Group(self.store)

    def close(self):
        if self.store is not None:
            if hasattr(self.store, 'close'):
                self.store.close()
            self.store = None
            self._zarr = None

    def create_dataset(self, name, shape=None, dtype=None, data=None, maxshape=None, **kwds):
        _ = maxshape
        return self.zarr.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, maxshape=None, **kwds):
        _ = maxshape
        return self.zarr.require_dataset(name, shape=shape, dtype=dtype, exact=exact, **kwds)

    def require_group(self, name):
        return self.zarr.require_group(name)

    def __getitem__(self, name):
        try:
            return self.zarr[name]
        except TypeError:
            if isinstance(name, Iterable):
                return np.array([self.zarr[e] for e in name])
            else:
                raise
    def __delitem__(self, name):
        del self.zarr[name]

    def __contains__(self, name):
        return name in self.zarr
    
    def __len__(self):
        return len(self.zarr)

    @property
    def attrs(self):
        return self.zarr.attrs()
    
    def flush(self):
        self.zarr.flush()

    
