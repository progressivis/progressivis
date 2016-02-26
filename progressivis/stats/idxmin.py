from progressivis.core.utils import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class IdxMin(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=False)])
        super(IdxMin, self).__init__(**kwds)
        self._columns = columns
        self._min = None
        self.default_step_size = 10000

    def min(self):
        return self._min

    def get_data(self, name):
        if name=='min':
            return self.min()
        return super(IdxMin,self).get_data(name)

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(IdxMin, self).is_ready()

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
            cols = input_df.columns.difference([self.UPDATE_COLUMN])
            if self._columns is None:
                self._columns = cols
            else:
                cols = cols.difference(self._columns)
                if cols is None:
                    logger.error('Nonexistant columns %s in dataframe, ignoring', self._columns)
                    self._columns = input_df.columns.difference([self.UPDATE_COLUMN])

        op = input_df.loc[indices,self._columns].idxmin()
        if not op.index.equals(self._columns):
            # some columns are not numerical
            self._columns = op.index

        op[self.UPDATE_COLUMN] = run_number
        if self._min is None:
            min = pd.Series([np.nan], index=op.index) # the UPDATE_COLUMN is included
            min[self.UPDATE_COLUMN] = run_number
            for col in op.index:
                if col==self.UPDATE_COLUMN: continue
                min[col] = input_df.loc[op[col], col] # lookup value, is there a better way?
            self._min = pd.DataFrame([min], columns=op.index)
            self._df = pd.DataFrame([op], columns=op.index)
        else:
            prev_min = self.last_row(self._min)
            prev_idx = self.last_row(self._df)
            min = pd.Series(prev_min)
            min[self.UPDATE_COLUMN] = run_number
            for col in op.index:
                if col==self.UPDATE_COLUMN: continue
                val = input_df.loc[op[col], col]
                if np.isnan(val):
                    pass
                elif np.isnan(min[col]) or val < min[col]:
                    op[col] = prev_idx[col]
                    min[col] = val
            op[self.UPDATE_COLUMN] = run_number
            with self.lock:
                self._df = self._df.append(op, ignore_index=True)
                self._min = self._min.append(min, ignore_index=True)
                if len(self._df) > self.params.history:
                    self._df = self._df.loc[self._df.index[-self.params.history:]]
                    self._min = self._min.loc[self._min.index[-self.params.history:]]

        return self._return_run_step(dfslot.next_state(), steps_run=steps)
