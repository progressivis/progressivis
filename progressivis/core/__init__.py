from __future__ import absolute_import, division, print_function

from ._version import get_versions
from .utils import (type_fullname, ProgressiveError, fix_loc, indices_len, integer_types)
from .scheduler import Scheduler
from .scheduler_base import BaseScheduler
from .slot import Slot, SlotDescriptor
from .storagemanager import StorageManager
from .module import Module, Every, Print
from .bitmap import bitmap
from .wait import Wait
# pylint: disable=unused-import
from .changemanager_bitmap import BitmapChangeManager


__version__ = get_versions()['version']
version = __version__
short_version = __version__
del get_versions


__all__ = ["ProgressiveError", "type_fullname", "fix_loc", "indices_len",
           "integer_types",
           "BaseScheduler", "Scheduler", "bitmap",
           "version", "__version__", "short_version",
           "Slot", "SlotDescriptor", "Module", "StorageManager",
           "Every", "Print", "Wait" ]
#           "get_option", "set_option", "option_context", "config_prefix" ]
