
from ..core.utils import indices_len
from ..core.slot import SlotDescriptor
from .module import TableModule
from .table import Table
from ..core.bitmap import bitmap
from . import TableSelectedView

import logging
logger = logging.getLogger(__name__)

class LiteSelect(TableModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True),
                         SlotDescriptor('select', type=bitmap, required=True)])
        super(LiteSelect, self).__init__(**kwds)
        self.default_step_size = 1000
        
        
    def run_step(self, run_number, step_size, howlong):
        step_size=1000
        table_slot = self.get_input_slot('table')
        table = table_slot.data()
        #table_slot.update(run_number,
        #                  buffer_created=False,
        #                  buffer_updated=True,
        #                  buffer_deleted=False,
        #                  manage_columns=False)
        
        select_slot = self.get_input_slot('select')
        select_slot.update(run_number,
                           buffer_created=True,
                           buffer_updated=False,
                           buffer_deleted=True)
                           
        steps = 0
        if self._table is None:
            self._table = TableSelectedView(table, bitmap([]))

        if select_slot.deleted.any():
            indices = select_slot.deleted.next(step_size, as_slice=False)
            s = indices_len(indices)
            print("LITESELECT: -",s)
            logger.info("deleting %s",indices)
            self._table.selection -= bitmap.asbitmap(indices)
            #step_size -= s//2

        if step_size > 0 and select_slot.created.any():
            indices = select_slot.created.next(step_size, as_slice=False)
            s = indices_len(indices)
            logger.info("creating %s",indices)
            steps += s
            #step_size -= s
            self._table.selection |= bitmap.asbitmap(indices)

        #print('index: ', len(self._table.index))
        return self._return_run_step(self.next_state(select_slot), steps_run=steps)

