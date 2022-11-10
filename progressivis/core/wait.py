from __future__ import annotations

from progressivis.utils.errors import ProgressiveError
from progressivis.core.module import Module, ReturnRunStep
from progressivis.core.slot import SlotDescriptor

import numpy as np

from typing import Any


class Wait(Module):
    parameters = [("delay", np.dtype(float), np.nan), ("reads", np.dtype(int), -1)]
    inputs = [SlotDescriptor("inp", required=True)]
    outputs = [SlotDescriptor("out", required=False)]

    def __init__(self, **kwds: Any) -> None:
        super(Wait, self).__init__(**kwds)
        if np.isnan(self.params.delay) and self.params.reads == -1:
            raise ProgressiveError(
                "Module %s needs either a delay or a number of reads, not both",
                self.pretty_typename(),
            )

    def is_ready(self) -> bool:
        if not super(Wait, self).is_ready():
            return False
        if self.is_zombie():
            return True  # give it a chance to run before it dies
        delay = self.params.delay
        reads = self.params.reads
        if np.isnan(delay) and reads < 0:
            return False
        inslot = self.get_input_slot("inp")
        assert inslot is not None
        trace = inslot.output_module.tracer.trace_stats()
        if len(trace) == 0:
            return False
        if not np.isnan(delay):
            return bool(len(trace) >= delay)
        elif reads >= 0:
            return bool(len(inslot.data()) >= reads)
        return False

    def get_data(self, name: str) -> Any:
        if name == "inp":
            slot = self.get_input_slot("inp")
            if slot is not None:
                return slot.data()
        return None

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("inp")
        if slot is not None:
            slot.clear_buffers()
        return self._return_run_step(self.state_blocked, steps_run=1)
