from progressivis.core.utils import indices_len, last_row, fix_loc
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Max(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Max, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(Max, self).is_ready()

    @synchronized
    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():        
            dfslot.reset()
            self._df = None
            dfslot.update(run_number)
        indices = dfslot.next_created(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        op = self.filter_columns(input_df, fix_loc(indices)).max()
        if not op.index.equals(self._columns):
            # some columns are not numerical
            self._columns = op.index

        op[self.UPDATE_COLUMN] = run_number
        if self._df is None:
            self._df = pd.DataFrame([op],index=[run_number])
        else:
            op = pd.concat([last_row(self._df), op], axis=1).max(axis=1)
            # Also computed the max over the UPDATE_COLUMNS so reset it
            op[self.UPDATE_COLUMN] = run_number
            self._df.loc[run_number] = op

        if len(self._df) > self.params.history:
            self._df = self._df.loc[self._df.index[-self.params.history:]]
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
