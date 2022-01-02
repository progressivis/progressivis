from __future__ import annotations

from ..core.utils import indices_len
from ..table.module import TableModule, ReturnRunStep
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class Counter(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, **kwds):
        super(Counter, self).__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super(Counter, self).is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            data = pd.DataFrame(dict(counter=steps), index=[0])
            if self.result is None:
                self.result = Table(
                    self.generate_table_name("counter"), data=data, create=True
                )
            elif len(self.result) == 0:
                self.table.append(data)
            else:
                self.table["counter"].loc[0] += steps
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
