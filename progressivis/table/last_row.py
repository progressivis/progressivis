from __future__ import annotations

from progressivis.core.module import Module, ReturnRunStep, def_input, def_output
from .table import PTable

from typing import Optional, Any


@def_input("table", PTable)
@def_output("result", PTable)
class LastRow(Module):
    def __init__(self, reset_index: Optional[bool] = True, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._reset_index = reset_index

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        assert slot is not None
        slot.clear_buffers()
        df = slot.data()

        if df is not None:
            last = df.last()
            if self.result is None:
                self.result = PTable(
                    self.generate_table_name("LastRow"), dshape=df.dshape
                )
                if self._reset_index:
                    self.result.add(last)
                else:
                    self.result.add(last, last.index)
            elif self._reset_index:
                self.result.loc[0] = last
            else:
                del self.result.loc[0]
                self.result.add(last, last.index)

        return self._return_run_step(self.state_blocked, steps_run=1)
