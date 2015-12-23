from progressivis.core.common import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Min(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Min, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(Min, self).is_ready()

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
        if isinstance(indices,slice):
            indices=slice(indices.start,indices.stop-1) # semantic of slice with .loc
        input_df = dfslot.data()
        if self._df is None:
            # Need to check now which columns exist. Cannot do it before we receive a valid df
            with dfslot.lock:
                cols = input_df.columns.difference([self.UPDATE_COLUMN])
            if self._columns is None:
                self._columns = cols
            else:
                cols = cols.difference(self._columns)
                if cols is None:
                    logger.error('Nonexistant columns %s in dataframe, ignoring', self._columns)
                    self._columns = input_df.columns.difference([self.UPDATE_COLUMN])

        with dfslot.lock:
            op = input_df.loc[indices,self._columns].min()
        if not op.index.equals(self._columns):
            # some columns are not numerical
            self._columns = op.index

        op[self.UPDATE_COLUMN] = run_number
        if self._df is None:
            self._df = pd.DataFrame([op],index=[run_number])
        else:
            op = pd.concat([self.last_row(self._df), op], axis=1).min(axis=1)
            # Also computed the min over the UPDATE_COLUMNS so reset it
            op[self.UPDATE_COLUMN] = run_number
        self._df.loc[run_number] = op

        if len(self._df) > self.params.history:
            self._df = self._df.loc[self._df.index[-self.params.history:]]
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
