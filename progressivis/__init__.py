"""
Main imports from progressivis.
"""
from __future__ import annotations

import logging

from progressivis.core.api import (
    Scheduler,
    Dataflow,
    Slot,
    SlotDescriptor,
    StorageManager,
    ReturnRunStep,
    Module,
    Every,
    Print,
    process_slot,
    PIntSet,
    Wait,
    Sink,
    def_input,
    def_output,
    def_parameter,
    document,
    indices_len,
    fix_loc,
    notNone,
    run_step_required,
    run_if_all,
    or_all,
    run_if_any,
    and_any,
    run_always,
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
    BinningIndex,
    BinningIndexND,
    RangeQuery2D,
    RangeQuery,
    Join,
    Merge,
    LastRow,
    Select,
)

from typing import Dict

from ._version import __version__
from progressivis.io.api import (
    CSVLoader,
    PACSVLoader,
    VECLoader,
    ParquetLoader,
    SimpleCSVLoader,
    ArrowBatchLoader,
    ThreadedCSVLoader,
    Variable,
)

from progressivis.stats.api import (
    Stats,
    Min,
    ScalarMin,
    Max,
    ScalarMax,
    Var,
    KLLSketch,
    IdxMax,
    IdxMin,
    VarH,
    Quantiles,
    Histogram1D,
    Histogram2D,
    MCHistogram2D,
    Histogram1DCategorical,
    Sample,
    Distinct,
    Corr,
    MinMaxScaler,
    RandomPTable,
    RandomDict,
)

from progressivis.vis import (
    Heatmap,
    MCScatterPlot,
)

from progressivis.datasets import (
    get_dataset,
)

version = __version__


__all__ = [
    "log_level",
    "ProgressiveError",
    "PDict",
    "Scheduler",
    "Dataflow",
    "version",
    "__version__",
    # "short_version",
    "Slot",
    "SlotDescriptor",
    "Module",
    "StorageManager",
    "ReturnRunStep",
    "Every",
    "Print",
    "process_slot",
    "PIntSet",
    "Wait",
    "Sink",
    "def_input",
    "def_output",
    "def_parameter",
    "document",
    "indices_len",
    "fix_loc",
    "run_step_required",
    "run_if_all",
    "or_all",
    "run_if_any",
    "and_any",
    "run_always",
    "notNone",
    "PColumn",
    "BasePColumn",
    "Constant",
    "ConstDict",
    "BasePTable",
    "IndexPTable",
    "PTableSelectedView",
    "PTable",
    "Row",
    "BinningIndex",
    "BinningIndexND",
    "RangeQuery",
    "RangeQuery2D",
    "Join",
    "Merge",
    "LastRow",
    "Select",
    "CSVLoader",
    "PACSVLoader",
    "VECLoader",
    "SimpleCSVLoader",
    "ArrowBatchLoader",
    "ThreadedCSVLoader",
    "ParquetLoader",
    "Variable",
    "Stats",
    "Min",
    "ScalarMin",
    "Max",
    "ScalarMax",
    "KLLSketch",
    "IdxMax",
    "IdxMin",
    "VarH",
    "Var",
    "Quantiles",
    "Histogram1D",
    "Histogram1DCategorical",
    "Histogram2D",
    "MCHistogram2D",
    "Sample",
    "RandomPTable",
    "RandomDict",
    "Distinct",
    "Corr",
    "MinMaxScaler",
    "Heatmap",
    "MCScatterPlot",
    "get_dataset",
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
