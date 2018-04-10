"""
Main imports from progressivis.
"""
import logging

from progressivis.core import (ProgressiveError,
                               version, __version__, short_version,
                               BaseScheduler, Scheduler, Slot, SlotDescriptor,
                               Module, StorageManager, Every, Print)

from progressivis.table import Table, Column, Row

__all__ = ["log_level",
           "ProgressiveError", "BaseScheduler", "Scheduler",
           "version", "__version__", "short_version",
           "Slot", "SlotDescriptor", "Module", "StorageManager",
           "Every", "Print",
           "Table", "Column", "Row"]

# Avoids the message 'No handlers could be found for logger X.Y.Z'
# logging.getLogger('progressivis').addHandler(logging.NullHandler())


def log_level(level=logging.DEBUG, package='progressivis'):
    "Set the logging level for progressivis."
    stream = logging.StreamHandler()
    stream.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                  '%(levelname)s - %(message)s')
    stream.setFormatter(formatter)
    logging.getLogger(package).addHandler(stream)
    logging.getLogger(package).setLevel(level)

# Usage example
# log_level(level=logging.INFO)
