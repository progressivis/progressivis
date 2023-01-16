from __future__ import annotations

import logging

from ..core.pintset import PIntSet
from ..core.module import ReturnRunStep
from ..core.utils import indices_len
from ..core.slot import SlotDescriptor
from ..table.table import PTable
from ..table.module import PTableModule
from ..table import PTableSelectedView

from typing import Callable, Optional, Any


logger = logging.getLogger(__name__)


class Switch(PTableModule):
    """
    Select the output (result or result_else) ar runtime
    """

    # parameters = []

    inputs = [SlotDescriptor("table", type=PTable, required=True)]
    outputs = [SlotDescriptor("result_else", type=PTable, required=False)]

    def __init__(self, condition: Callable[..., Optional[bool]], **kwds: Any) -> None:
        """
        condition: callable which should return
        * None => undecidable (yet), run_step must return blocked_state
        * True => run_step output is 'result'
        * False => run_step output is 'result_else'
        """
        assert callable(condition)
        super().__init__(**kwds)
        self._condition = condition
        self.result_else: Optional[PTableSelectedView] = None
        self._output: Optional[PTableSelectedView] = None

    def reset(self) -> None:
        if self.result is not None:
            self.selected.selection = PIntSet()
        if self.result_else is not None:
            self.result_else.selection = PIntSet()

    def get_data(self, name: str) -> Any:
        if name == "result_else":
            return self.result_else
        return super().get_data(name)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        input_df = slot.data()
        if input_df is None or not slot.has_buffered():
            return self._return_run_step(self.state_blocked, steps_run=0)
        cond = self._condition(self)
        if cond is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._output is None:
            if cond:
                self._output = self.result = PTableSelectedView(input_df, PIntSet([]))
            else:
                self._output = self.result_else = PTableSelectedView(
                    input_df, PIntSet([])
                )
        steps = 0
        created_ids = PIntSet()
        if slot.created.any():
            created_ids = slot.created.next(as_slice=False)
            steps += indices_len(created_ids)
            self._output.selection |= created_ids
        updated_ids = PIntSet()
        if slot.updated.any():
            updated_ids = slot.updated.next(as_slice=False)
            steps += indices_len(updated_ids)
            print("Updates are ignored in switch")
            # self._output._base.add_updated(updated_ids)
        deleted_ids = PIntSet()
        if slot.deleted.any():
            deleted_ids = slot.deleted.next(as_slice=False)
            steps += indices_len(deleted_ids)
            self._output.selection -= deleted_ids
        return self._return_run_step(self.next_state(slot), steps_run=steps)
