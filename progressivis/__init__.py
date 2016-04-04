import logging

from progressivis.core import (ProgressiveError, NIL,
                               version, __version__, short_version,
                               Scheduler, MTScheduler, Slot, SlotDescriptor,
                               Module, connect, StorageManager, Every, Print,
                               DataFrameModule, Constant, Select,
                               Wait, RangeQuery, Merge, Join, CombineFirst, LastRow,
                               NIL_INDEX, index_diff, index_changes)



__all__ = ["log_level",
           "ProgressiveError", "Scheduler", "MTScheduler",
           "version", "__version__", "short_version",
           "Slot", "SlotDescriptor", "Module", "connect", "StorageManager",
           "DataFrameModule", "Constant", "Every", "Print", "Wait", "Select",
           "RangeQuery", "Merge", "Join", "CombineFirst", "NIL", "LastRow",
           "NIL_INDEX", 'index_diff', 'index_changes']


# Magic spell to avoid the message 'No handlers could be found for logger X.Y.Z'
#logging.getLogger('progressivis').addHandler(logging.NullHandler())

def log_level(level=logging.DEBUG, package='progressivis'):
    stream = logging.StreamHandler()
    stream.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream.setFormatter(formatter)
    logging.getLogger(package).addHandler(stream)
    logging.getLogger(package).setLevel(level)

# Usage example
#log_level(level=logging.INFO)
