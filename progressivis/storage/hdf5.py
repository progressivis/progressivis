from __future__ import absolute_import, division, print_function

from progressivis.core.storagemanager import StorageManager
from progressivis.core.config import get_option
from .base import StorageEngine, Group, Attribute, Dataset

import h5py

Group.register(h5py.Group)
Attribute.register(h5py.AttributeManager)
Dataset.register(h5py.Dataset)

class HDF5StorageEngine(StorageEngine):
    def __init__(self):
        super(HDF5StorageEngine, self).__init__("hdf5")
        self._h5py = None
        self._root = None

    @property
    def h5py(self):
        if self._h5py is None:
            self._h5py = self.open('default.h5', 'w')
        return self._h5py

    @property
    def root(self):
        if self._root is None:
            self._root = self.h5py['/']
        return self._root

    def set_default(self, f):
        assert isinstance(f, h5py.File)
        if self._h5py is not None:
            self.close()
        self._h5py = f

    def open(self, name, flags, **kwds):
        if not kwds:
            kwds = get_option('storage.hdf5.open', kwds)
        return h5py.File(StorageManager.default.filename(name), flags, **kwds)

    def close(self):
        if self._h5py is not None:
            self._h5py.close()
            self._h5py = None
            self._root = None

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if kwds.get('compression') is None:
            kwds.update(get_option('storage.hdf5.compression', {}))
        return self.root.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        if kwds.get('compression') is None:
            kwds.update(get_option('storage.hdf5.compression', {}))
        return self.root.require_dataset(name, shape=shape, dtype=dtype, exact=exact, **kwds)

    def require_group(self, name):
        return self.root.require_group(name)

    def __getitem__(self, name):
        return self.root[name]

    def __delitem__(self, name):
        del self.root[name]

    def __contains__(self, name):
        return name in self.root
    
    def __len__(self):
        return len(self.root)

    @property
    def attrs(self):
        return self.root.attrs()
    
    def flush(self):
        self.h5py.flush()

    
