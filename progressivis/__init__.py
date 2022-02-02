"""
Main imports from progressivis.
"""
from __future__ import annotations

import logging
import yaml
import sys

from progressivis.core import (
    version,
    __version__,
    short_version,
    Scheduler,
    Slot,
    SlotDescriptor,
    StorageManager,
    Module,
    Every,
    Print,
)
from progressivis.utils import ProgressiveError
from progressivis.core import aio
from progressivis.table import Table, Column, Row
from IPython.core.magic import (Magics, magics_class,  # type: ignore
                                cell_magic, line_cell_magic,
                                needs_local_scope)

from typing import Dict, Optional, Any

__all__ = [
    "log_level",
    "ProgressiveError",
    "Scheduler",
    "version",
    "__version__",
    "short_version",
    "Slot",
    "SlotDescriptor",
    "Module",
    "StorageManager",
    "Every",
    "Print",
    "Table",
    "Column",
    "Row",
]


LOGGERS: Dict[str, logging.Logger] = {}


def s() -> Scheduler:
    "Shortcut to get the default scheduler."
    return Scheduler.default


# Avoids the message 'No handlers could be found for logger X.Y.Z'
# logging.getLogger('progressivis').addHandler(logging.NullHandler())

# Usage example
# log_level(level=logging.INFO)


def log_level(level: int = logging.DEBUG, package: str = "progressivis") -> None:
    "Set the logging level for progressivis."
    global LOGGERS

    print(f"Setting logging level to {level} for {package}")
    logger = LOGGERS.get(package)
    if logger is None:
        logger = logging.getLogger(package)
        LOGGERS[package] = logger
        logger.propagate = False
    logger.setLevel(level)
    logger.handlers.clear()
    stream = logging.StreamHandler()
    stream.setLevel(level)
    formatter = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
    )
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    # logging.getLogger(package).setLevel(level)


# https://gist.github.com/nkrumm/2246c7aa54e175964724
@magics_class
class ProgressivisMagic(Magics):  # type: ignore
    @line_cell_magic  # type: ignore
    @needs_local_scope  # type: ignore
    def progressivis(self, line: str,
                     cell: Optional[str] = None,
                     local_ns: Any = None) -> Any:
        from IPython.display import clear_output  # type: ignore
        if cell is None:
            clear_output()
            for ln in yaml.dump(dict(eval(line, local_ns))).split('\n'):
                print(ln)
            sys.stdout.flush()
        else:
            ps_dict = eval(line, local_ns)
            ps_dict.update(yaml.safe_load(cell))
            return ps_dict

    @cell_magic  # type: ignore
    @needs_local_scope  # type: ignore
    def from_input(self, line: str, cell: str, local_ns: Optional[Any] = None) -> aio.Task[Any]:
        module = eval(line, local_ns)
        return aio.create_task(module.from_input(yaml.safe_load(cell)))


def load_ipython_extension(ipython: Any) -> None:
    from IPython import get_ipython  # type: ignore
    ip = get_ipython()
    ip.register_magics(ProgressivisMagic)
