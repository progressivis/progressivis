from __future__ import annotations

from progressivis.core.api import Module, ReturnRunStep, def_input, def_output, def_parameter
from progressivis.table.api import PTable

try:
    from .cxx_max import Max as CxxMax  # type: ignore
except Exception:
    CxxMax = None

import numpy as np

import logging

from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@def_input("table", PTable)
@def_parameter("history", np.dtype(int), 3)
@def_output("result", PTable)
class Max(Module):
    def __init__(self, columns: Optional[List[str]] = None, **kwds: Any) -> None:
        super().__init__(**kwds)
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
