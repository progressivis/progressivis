from __future__ import annotations

import numpy as np

from progressivis.table.api import PTable
from progressivis.core.module import Module
from progressivis.core.module import ReturnRunStep, JSon, def_output, def_parameter

from typing import Any


@def_parameter("history", np.dtype(int), 3)
@def_output("result", PTable)
class Input(Module):
    schema = "{input: string}"

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.tags.add(self.TAG_INPUT)
        table = PTable(name=None, dshape=Input.schema, create=True)
        self.result = table
        self._last = len(table)
        self.default_step_size = 1000000

    def is_ready(self) -> bool:
        assert self.result is not None
        return len(self.result) > self._last

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.result is not None
        self._last = len(self.result)
        return self._return_run_step(self.state_blocked, steps_run=0)

    async def from_input(self, msg: JSon, stop_iter: bool = False) -> str:
        if not isinstance(msg, dict):
            msg = {"input": msg}
        assert self.result is not None
        self.result.add(msg)
        return ""
