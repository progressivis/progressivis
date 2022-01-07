from __future__ import annotations

from progressivis.core.slot import SlotDescriptor
from progressivis.core.module import ReturnRunStep
from .module import TableModule
from .table import Table

from typing import Optional, Any


class LastRow(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, reset_index: Optional[bool] = True, **kwds: Any) -> None:
        super(LastRow, self).__init__(**kwds)
        self._reset_index = reset_index

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        assert slot is not None
        slot.clear_buffers()
        df = slot.data()

        if df is not None:
            last = df.last()
            if self.result is None:
                self.result = Table(
                    self.generate_table_name("LastRow"), dshape=df.dshape
                )
                if self._reset_index:
                    self.table.add(last)
                else:
                    self.table.add(last, last.index)
            elif self._reset_index:
                self.table.loc[0] = last
            else:
                del self.table.loc[0]
                self.table.add(last, last.index)

        return self._return_run_step(self.state_blocked, steps_run=1)
