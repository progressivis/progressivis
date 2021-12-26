from __future__ import annotations

from ..core.utils import indices_len
from ..core.slot import SlotDescriptor
from .module import TableModule, ReturnRunStep
from .table import Table
from ..core.bitmap import bitmap
from . import TableSelectedView

import logging

logger = logging.getLogger(__name__)


class LiteSelect(TableModule):
    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("select", type=bitmap, required=True),
    ]

    def __init__(self, **kwds):
        super(LiteSelect, self).__init__(**kwds)
        self.default_step_size = 1000

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        step_size = 1000
        table_slot = self.get_input_slot("table")
        table = table_slot.data()
        # table_slot.update(run_number,
        #                  buffer_created=False,
        #                  buffer_updated=True,
        #                  buffer_deleted=False,
        #                  manage_columns=False)
        select_slot = self.get_input_slot("select")
        # select_slot.update(run_number,
        #                    buffer_created=True,
        #                    buffer_updated=False,
        #                    buffer_deleted=True)

        steps = 0
        if self.result is None:
            self.result = TableSelectedView(table, bitmap([]))

        if select_slot.deleted.any():
            indices = select_slot.deleted.next(step_size, as_slice=False)
            s = indices_len(indices)
            print("LITESELECT: -", s)
            logger.info("deleting %s", indices)
            self.selected.selection -= bitmap.asbitmap(indices)
            # step_size -= s//2

        if step_size > 0 and select_slot.created.any():
            indices = select_slot.created.next(step_size, as_slice=False)
            s = indices_len(indices)
            logger.info("creating %s", indices)
            steps += s
            # step_size -= s
            self.selected.selection |= bitmap.asbitmap(indices)

        return self._return_run_step(self.next_state(select_slot), steps_run=steps)
