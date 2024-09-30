"""
Main imports from progressivis.
"""
from __future__ import annotations

import logging

from progressivis.core.api import (
    Scheduler,
    Slot,
    SlotDescriptor,
    StorageManager,
    Module,
    Every,
    Print,
    PIntSet,
    Sink,
)

from progressivis.utils.api import (
    ProgressiveError,
    PDict
)

from progressivis.table.api import (
    BasePColumn,
    PColumn,
    BasePTable,
    IndexPTable,
    PTableSelectedView,
    PTable,
    Constant,
    ConstDict,
    Row,
    RangeQuery2d,
    RangeQuery,
    Join,
)

from typing import Dict

from ._version import __version__
from progressivis.io import (
    CSVLoader,
    ParquetLoader,
    Variable,
)

from progressivis.stats import (
    Histogram1D,
    Histogram2D,
    Max,
    Min,
    Quantiles,
    RandomPTable,
)

from progressivis.vis import (
    Heatmap,
)

version = __version__


__all__ = [
    "log_level",
    "ProgressiveError",
    "PDict",
    "Scheduler",
    "version",
    "__version__",
    # "short_version",
    "Slot",
    "SlotDescriptor",
    "Module",
    "StorageManager",
    "Every",
    "Print",
    "PIntSet",
    "Sink",
    "PColumn",
    "BasePColumn",
    "Constant",
    "ConstDict",
    "BasePTable",
    "IndexPTable",
    "PTableSelectedView",
    "PTable",
    "Row",
    "RangeQuery",
    "RangeQuery2d",
    "Join",
    "CSVLoader",
    "ParquetLoader",
    "Variable",
    "Histogram1D",
    "Histogram2D",
    "Max",
    "Min",
    "Quantiles",
    "RandomPTable",
    "Heatmap",
]


LOGGERS: Dict[str, logging.Logger] = {}


# def s() -> Scheduler:
#     "Shortcut to get the default scheduler."
#     return Scheduler.default


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
