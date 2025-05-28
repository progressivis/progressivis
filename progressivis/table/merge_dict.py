from __future__ import annotations

from ..core.module import Module, ReturnRunStep, def_input, def_output, document
from ..utils.psdict import PDict

from typing import Any


@document
@def_input("table", PDict, multiple=True, doc="Multiple inputs providing dictionaries")
@def_output("result", PDict, doc="Merged dictionary")
class MergeDict(Module):
    """
    Gathers many (dict) outputs in one. Assume there is no clash.
    Complementary to ``Switch`` module
    """

    def __init__(
        self,
        **kwds: Any,
    ) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)

    # parameters = []

    def run_step(
        self, run_number: int, step_size: int, quantum: float
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
