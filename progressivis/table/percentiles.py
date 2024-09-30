from __future__ import annotations

import numpy as np

from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter
from ..utils.psdict import PDict
from .binning_index import BinningIndex
from .api import PTable
from typing import Any, cast


@def_parameter("accuracy", np.dtype(float), 0.5)
@def_input("percentiles", PDict)
@def_input("index", PTable)
@def_output("result", PDict)
class Percentiles(Module):
    """ """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._accuracy = self.params.accuracy
        self.default_step_size = 1000
        self.result = PDict()

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        steps = 0
        percentiles_slot = self.get_input_slot("percentiles")
        if percentiles_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if percentiles_slot.has_buffered():
            steps += 1
            percentiles_slot.clear_buffers()
        if len(percentiles_slot.data()) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        hist_slot = self.get_input_slot("index")
        if hist_slot.has_buffered():
            steps += 1
            hist_slot.clear_buffers()
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        hist_index: BinningIndex = cast(BinningIndex, hist_slot.output_module)
        if not hist_index._impl:
            return self._return_run_step(self.state_blocked, steps_run=0)
        computed = hist_index.compute_percentiles(
            percentiles_slot.data(), self._accuracy
        )
        assert self.result is not None
        self.result.update(computed)
        return self._return_run_step(self.next_state(hist_slot), steps)
