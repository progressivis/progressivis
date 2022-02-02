from __future__ import annotations

import logging

from ..core.module import ReturnRunStep
from ..table.nary import NAry
from ..utils.psdict import PsDict

logger = logging.getLogger(__name__)


class Hub(NAry):
    """
    Groups many (dict) outputs in one. Suppose there are no clashes
    Useful with Switch
    """

    # parameters = []

    def run_step(self, run_number: int, step_size: int, howlong: float) -> ReturnRunStep:
        if self.result is None:
            self.result = PsDict()
        steps = 0
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            if slot.has_buffered():
                d = slot.data()
                steps += len(d)
                assert isinstance(d, PsDict)
                self.psdict.update(d)
            slot.clear_buffers()
        return self._return_run_step(self.state_blocked, steps_run=steps)
