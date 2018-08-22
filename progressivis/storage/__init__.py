from __future__ import absolute_import, division, print_function

from progressivis.core.config import get_option
from .base import Group, StorageEngine

__all__ = ["Group"]

try:
    from .numpy import NumpyStorageEngine, NumpyGroup
    numpyengine = NumpyStorageEngine()
except ImportError:
    numpyengine = None

try:
    from .mmap import MMapStorageEngine, MMapGroup
    mmapengine = MMapStorageEngine()
except ImportError:
    mmapengine = None


# try:
#     from .pptable import PPTableStorageEngine
#     pptableengine = PPTableStorageEngine()
# except ImportError:
#     pptableengine = None

try:
    from .hdf5 import HDF5StorageEngine
    hdf5engine = HDF5StorageEngine()
except ImportError:
    hdf5engine = None

try:
    from .bcolz import BCOLZStorageEngine
    bcolzengine = BCOLZStorageEngine()
except ImportError:
    bcolzengine = None

try:
    from .zarr import ZARRStorageEngine
    zarrengine = ZARRStorageEngine()
except ImportError:
    zarrengine = None

if get_option('storage.default'):
    StorageEngine.default = get_option('storage.default')

#Group.default = MMapGroup #NumpyGroup
Group.default = MMapStorageEngine.create_mmap_group
