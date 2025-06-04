import numpy as np
from typing import Any
from progressivis import (Module, ReturnRunStep, PTable, PDict,
                          document, def_input, def_output)
from progressivis.core.utils import indices_len, fix_loc


@document
@def_input("table", PTable, doc="The input PTable to process")
@def_output("result", PDict, doc=("PDict where each key represents a column"))
class SimpleMax(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        # Handle the changed input slots
        table_slot = self.get_input_slot("table")
        if table_slot.updated.any() or table_slot.deleted.any():
            table_slot.reset()
            table_slot.update(run_number)
            self.reset()
        # Extract the new chunk
        indices = table_slot.created.next(length=step_size)
        steps = indices_len(indices)
        chunk = table_slot.data().loc[fix_loc(indices)]
        # Apply the operation on the chunk
        op = chunk.max(keepdims=False)
        # Update the result
        if self.result is None:
            self.result = PDict(op)
        else:
            for k, v in self.result.items():
                self.result[k] = np.fmax(op[k], v)
        # Return the next state and number of steps handled
        if table_slot.has_buffered():
            next_state = Module.state_ready
        else:
            next_state = Module.state_blocked
        return self._return_run_step(next_state, steps)

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(-np.inf)
