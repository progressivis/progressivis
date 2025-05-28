import numpy as np
from typing import Any, Sequence
from progressivis import (
    Module, ReturnRunStep, def_input, def_output,
    PTable, PDict, document, process_slot, run_if_any
)
from ..core.utils import indices_len, fix_loc
from progressivis.core.docstrings import INPUT_SEL


@document
@def_input("table", PTable, hint_type=Sequence[str], doc=INPUT_SEL)
@def_output(
    "result",
    PDict,
    doc=("PDict where each key represents a column"),
)
class Max(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(-np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        with self.context as ctx:
            indices = ctx.table.created.next(length=step_size)
            steps = indices_len(indices)
            op = self.filter_slot_columns(
                ctx.table,
                fix_loc(indices)
            ).max(keepdims=False)
            if self.result is None:
                self.result = PDict(op)
            else:
                for k, v in self.result.items():
                    self.result[k] = np.fmax(op[k], v)
            return self._return_run_step(
                self.next_state(ctx.table),
                steps
            )
