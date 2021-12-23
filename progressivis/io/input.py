from __future__ import annotations

import numpy as np

from progressivis.table import Table
from progressivis.table.module import TableModule, ReturnRunStep, JSon


class Input(TableModule):
    parameters = [("history", np.dtype(int), 3)]
    schema = "{input: string}"

    def __init__(self, **kwds):
        super(Input, self).__init__(**kwds)
        self.tags.add(self.TAG_INPUT)
        self.result = Table(name=None, dshape=Input.schema, create=True)
        self._last = len(self.result)
        self.default_step_size = 1000000

    def is_ready(self) -> bool:
        return len(self.result) > self._last

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        self._last = len(self.result)
        return self._return_run_step(self.state_blocked, steps_run=0)

    async def from_input(self, msg: JSon) -> str:
        if not isinstance(msg, (list, dict)):
            msg = {"input": msg}
        self.result.add(msg)
        return ""
