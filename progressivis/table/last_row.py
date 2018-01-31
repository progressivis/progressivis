from __future__ import absolute_import, division, print_function

from ..core.slot import SlotDescriptor
from .module import TableModule
from .table import Table


class LastRow(TableModule):
    def __init__(self, reset_index=True, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        super(LastRow, self).__init__(**kwds)
        self._reset_index = reset_index
        
    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        slot = self.get_input_slot('table')
        df = slot.data()

        if df is not None:
            with slot.lock:
                last = df.last()
                if self._table is None:
                    self._table = Table(self.generate_table_name('LastRow'),
                                        dshape=df.dshape)
                    if self._reset_index:
                        self._table.add(last)
                    else:
                        self._table.add(last, last.index)
                elif self._reset_index:
                    self._table.loc[0] = last
                else:
                    del self._table.loc[0]
                    self._table.add(last, last.index)
                    
        return self._return_run_step(self.state_blocked, steps_run=1)
