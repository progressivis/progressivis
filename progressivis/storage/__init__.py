from __future__ import absolute_import, division, print_function

from progressivis.core.config import get_option
from .base import Group, StorageEngine

__all__ = ["Group"]

from .numpy import NumpyStorageEngine
numpyengine = NumpyStorageEngine()

from .mmap import MMapStorageEngine
mmapengine = MMapStorageEngine()

if get_option('storage.default'):
    StorageEngine.default = get_option('storage.default')

Group.default = staticmethod(NumpyStorageEngine.create_group)
Group.default_internal = staticmethod(NumpyStorageEngine.create_group)
#Group.default = staticmethod(MMapStorageEngine.create_group)
IS_PERSISTENT = Group.default.__module__=='progressivis.storage.mmap' # TODO consider all other persistent storage (HDF5 etc.)
