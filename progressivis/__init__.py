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

from progressivis.core.version import version as __version__
from progressivis.core.common import ProgressiveError
from progressivis.core.scheduler import Scheduler
from progressivis.core.slot import Slot, SlotDescriptor
from progressivis.core.storagemanager import StorageManager
from progressivis.core.module import Module, connect, Print
from progressivis.core.dataframe import DataFrameModule, Constant
from progressivis.core.wait import Wait
from progressivis.core.merge import Merge
