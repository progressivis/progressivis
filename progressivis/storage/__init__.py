
from progressivis.core.config import get_option
from .base import Group, StorageEngine

__all__ = ["Group"]

from .numpy import NumpyStorageEngine
numpyengine = NumpyStorageEngine()

from .mmap import (MMapStorageEngine, init_temp_dir_if,
                   cleanup_temp_dir, temp_dir, Persist)
mmapengine = MMapStorageEngine()

if get_option('storage.default'):
    StorageEngine.default = get_option('storage.default')
else:
    StorageEngine.default = 'numpy'

if StorageEngine.default == 'mmap':
    Group.default = staticmethod(MMapStorageEngine.create_group)
elif StorageEngine.default == 'numpy':
    Group.default = staticmethod(NumpyStorageEngine.create_group)
else:
    raise ValueError(f"Unknown storage {StorageEngine.default}")
Group.default_internal = staticmethod(NumpyStorageEngine.create_group)
#Group.default = staticmethod(MMapStorageEngine.create_group)
IS_PERSISTENT = Group.default.__module__=='progressivis.storage.mmap'
