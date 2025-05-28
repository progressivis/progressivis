from __future__ import annotations

import logging

from ..core.utils import indices_len
from ..core.module import Module, ReturnRunStep, def_input, def_output
from .api import PTable, PTableSelectedView
from ..core.pintset import PIntSet

from typing import Any


logger = logging.getLogger(__name__)


@def_input("table", PTable)
@def_input("select", PIntSet)
@def_output("result", PTableSelectedView)
class LiteSelect(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 1000

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        step_size = 1000
        table_slot = self.get_input_slot("table")
        assert table_slot is not None
        table = table_slot.data()
        # table_slot.update(run_number,
        #                  buffer_created=False,
        #                  buffer_updated=True,
        #                  buffer_deleted=False,
        #                  manage_columns=False)
        select_slot = self.get_input_slot("select")
        assert select_slot is not None
        # select_slot.update(run_number,
        #                    buffer_created=True,
        #                    buffer_updated=False,
        #                    buffer_deleted=True)

        steps = 0
        if self.result is None:
            self.result = PTableSelectedView(table, PIntSet([]))

        if select_slot.deleted.any():
            indices = select_slot.deleted.next(length=step_size, as_slice=False)
            s = indices_len(indices)
            # print("LITESELECT: -", s)
            logger.info("deleting %s", indices)
            self.result.selection -= PIntSet.aspintset(indices)
            # step_size -= s//2

        if step_size > 0 and select_slot.created.any():
            indices = select_slot.created.next(length=step_size, as_slice=False)
            s = indices_len(indices)
            logger.info("creating %s", indices)
            steps += s
            # step_size -= s
            self.result.selection |= PIntSet.aspintset(indices)

        return self._return_run_step(self.next_state(select_slot), steps_run=steps)
