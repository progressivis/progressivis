import logging
# Magic spell to avoid the message 'No handlers could be found for logger X.Y.Z'
logging.getLogger('progressive').addHandler(logging.NullHandler())

def log_level(level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('progressive').addHandler(ch)

__all__ = ["log_level", "ProgressiveError", "Scheduler", "Slot",
           "SlotDescriptor", "Module", "connect", "DataFrameModule",
           "StorageManager", "Constant", "Print", "Wait", "Merge" ]

from progressive.core.version import version as __version__
from progressive.core.common import ProgressiveError
from progressive.core.scheduler import Scheduler
from progressive.core.slot import Slot, SlotDescriptor
from progressive.core.storagemanager import StorageManager
from progressive.core.module import Module, connect, Print
from progressive.core.dataframe import DataFrameModule, Constant
from progressive.core.wait import Wait
from progressive.core.merge import Merge
