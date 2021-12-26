from __future__ import annotations

from progressivis.table.module import TableModule, ReturnRunStep
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor

try:
    from .cxx_max import Max as CxxMax  # type: ignore
except Exception:
    CxxMax = None

import numpy as np

import logging

from typing import List

logger = logging.getLogger(__name__)


class Max(TableModule):
    parameters = [("history", np.dtype(int), 3)]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, columns: List[str] = None, **kwds):
        super(Max, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000
        self.cxx_module = CxxMax(self)

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        return self.cxx_module.run(run_number, step_size, howlong)
