from __future__ import annotations

from ..core.module import Module, ReturnRunStep
from ..core.slot import SlotDescriptor

from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.scheduler import Scheduler
    from .table_base import BaseTable


class WaitForData(Module):
    """
    Calls a function when a table has some data and terminate.

    Module useful to create modules according to the actual types and initial values
    of a table.
    """

    inputs = [
        SlotDescriptor("table", type=BaseTable, required=True),
    ]

    def __init__(
        self, proc: Callable[[Scheduler, BaseTable], None], **kwds: Any
    ) -> None:
        super(WaitForData, self).__init__(**kwds)
        self._proc = proc

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        table = slot.data()
        if table is not None and len(table) > 0:
            self._proc(self.scheduler(), table)
            return self._return_terminate()
        return self._return_run_step(Module.state_blocked, 0)
