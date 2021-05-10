"""
Main imports from progressivis.
"""
import logging
import yaml,sys

POLL_INTERVAL = None
DO_ONE_ITERATION = None

from progressivis.core import (version, __version__, short_version,
                               Scheduler,
                               Slot, SlotDescriptor,
                               StorageManager,
                               Module, Every, Print)
from progressivis.utils import ProgressiveError
from progressivis.core import aio
from progressivis.table import Table, Column, Row
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic, needs_local_scope)

__all__ = ["log_level",
           "ProgressiveError", "Scheduler",
           "version", "__version__", "short_version",
           "Slot", "SlotDescriptor", "Module", "StorageManager",
           "Every", "Print",
           "Table", "Column", "Row", "POLL_INTERVAL", "DO_ONE_ITERATION"]


def s():
    "Shortcut to get the default scheduler."
    return Scheduler.default

# Avoids the message 'No handlers could be found for logger X.Y.Z'
# logging.getLogger('progressivis').addHandler(logging.NullHandler())


def log_level(level=logging.DEBUG, package='progressivis'):
    "Set the logging level for progressivis."
    logger = logging.getLogger(package)
    if logger.handlers:
        return
    stream = logging.StreamHandler()
    stream.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                  '%(levelname)s - %(message)s')
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logging.getLogger(package).setLevel(level)

# Usage example
# log_level(level=logging.INFO)
try:
    from ipykernel.eventloops import register_integration
    @register_integration('progressivis')
    def loop_progressivis(kernel):
        global POLL_INTERVAL, DO_ONE_ITERATION
        POLL_INTERVAL = kernel._poll_interval
        DO_ONE_ITERATION = kernel.do_one_iteration
except ImportError:
    pass


# https://gist.github.com/nkrumm/2246c7aa54e175964724
@magics_class
class ProgressivisMagic(Magics):
    @line_cell_magic
    @needs_local_scope
    def progressivis(self, line, cell=None, local_ns=None):
        from IPython.display import clear_output
        if cell is None:
            clear_output()
            for ln in yaml.dump(dict(eval(line, local_ns))).split('\n'):
                print(ln)
            sys.stdout.flush()
        else:
            ps_dict = eval(line, local_ns)
            ps_dict.update(yaml.safe_load(cell))
            return ps_dict

    @cell_magic
    @needs_local_scope
    def from_input(self, line, cell, local_ns=None):
        module = eval(line, local_ns)
        return aio.create_task(module.from_input(yaml.safe_load(cell)))


def load_ipython_extension(ipython):
    ip = get_ipython()
    ip.register_magics(ProgressivisMagic)
