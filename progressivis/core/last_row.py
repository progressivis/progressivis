from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import pandas as pd

class LastRow(DataFrameModule):
    def __init__(self, reset_index=True, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(LastRow, self).__init__(**kwds)
        self._reset_index = reset_index
        
    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        slot = self.get_input_slot('df')
        df = slot.data()

        if df is not None:
            with slot.lock:
                last = self.last_row(slot.data(), as_series=False)
            last[self.UPDATE_COLUMN] = run_number
            if self._reset_index:
                last.index = [0]
            with self.lock:
                self._df = last
        return self._return_run_step(self.state_blocked, steps_run=1)
