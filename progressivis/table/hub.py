from __future__ import annotations

from ..core.module import Module, ReturnRunStep, def_input, def_output
from ..utils.psdict import PDict


@def_input("table", PDict, multiple=True)
@def_output("result", PDict)
class Hub(Module):
    """
    Groups many (dict) outputs in one. Assume there is no clash
    Useful with Switch
    """

    # parameters = []

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if self.result is None:
            self.result = PDict()
        steps = 0
        for name in self.get_input_slot_multiple("table"):
            slot = self.get_input_slot(name)
            if slot.has_buffered():
                d = slot.data()
                steps += len(d)
                assert isinstance(d, PDict)
                self.result.update(d)
            slot.clear_buffers()
        return self._return_run_step(self.state_blocked, steps_run=steps)
