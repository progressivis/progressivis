import logging
# Magic spell to avoid the message 'No handlers could be found for logger X.Y.Z'
logging.getLogger('progressive').addHandler(logging.NullHandler())

__all__ = ["ProgressiveError", "Scheduler", "Slot",
           "SlotDescriptor", "Module", "connect", "DataFrameModule",
           "Constant", "Print", "Wait", "Merge" ]

from progressive.core.version import version as __version__
from progressive.core.common import ProgressiveError
from progressive.core.scheduler import Scheduler
from progressive.core.slot import Slot, SlotDescriptor
from progressive.core.module import Module, connect, Print
from progressive.core.dataframe import DataFrameModule, Constant
from progressive.core.wait import Wait
from progressive.core.merge import Merge
