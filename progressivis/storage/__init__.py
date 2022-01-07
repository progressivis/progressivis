from __future__ import annotations

from progressivis.core.config import get_option
from .base import Group, StorageEngine, Dataset
from .numpy import NumpyStorageEngine
from .mmap import MMapStorageEngine, init_temp_dir_if, cleanup_temp_dir, temp_dir


__all__ = ["Group", "Dataset", "init_temp_dir_if", "cleanup_temp_dir", "temp_dir"]


numpyengine = NumpyStorageEngine()
mmapengine = MMapStorageEngine()

if get_option("storage.default"):
    StorageEngine._default = get_option("storage.default")
else:
    StorageEngine._default = "numpy"

if StorageEngine._default == "mmap":
    Group.default = MMapStorageEngine.create_group
elif StorageEngine._default == "numpy":
    Group.default = NumpyStorageEngine.create_group
else:
    raise ValueError(f"Unknown storage {StorageEngine.default}")
Group.default_internal = NumpyStorageEngine.create_group


IS_PERSISTENT = Group.default.__module__ == "progressivis.storage.mmap"
