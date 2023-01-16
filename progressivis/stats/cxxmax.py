from __future__ import annotations

from progressivis.core.module import ReturnRunStep
from progressivis.core.slot import SlotDescriptor
from progressivis.table.module import PTableModule
from progressivis.table.table import PTable

try:
    from .cxx_max import Max as CxxMax  # type: ignore
except Exception:
    CxxMax = None

import numpy as np

import logging

from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class Max(PTableModule):
    parameters = [("history", np.dtype(int), 3)]
    inputs = [SlotDescriptor("table", type=PTable, required=True)]

    def __init__(self, columns: Optional[List[str]] = None, **kwds: Any) -> None:
        super(Max, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000
        self.cxx_module = CxxMax(self)

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return self.cxx_module.run(run_number, step_size, howlong)  # type: ignore
