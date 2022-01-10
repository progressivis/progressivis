"""
Main imports from progressivis.
"""
from __future__ import annotations

import logging

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
from progressivis.table import Table, Column, Row

from typing import Dict

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
