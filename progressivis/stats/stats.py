from progressivis.core.common import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

#TODO update with http://www.johndcook.com/blog/skewness_kurtosis/
#Use http://www.grantjenks.com/docs/runstats/ 

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Stats(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, column, min_column=None, max_column=None, reset_index=False, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Stats, self).__init__(dataframe_slot='stats', **kwds)
        self._column = column
        self.default_step_size = 10000

        if min_column is None:
            min_column = str(column) + '.min'
        if max_column is None:
            max_column = str(column) + '.max'
        self._min_column = min_column
        self._max_column = max_column
        self._reset_index = reset_index
        self.schema = [(self._min_column, np.dtype(float), np.nan),
                       (self._max_column, np.dtype(float), np.nan),
                       DataFrameModule.UPDATE_COLUMN_DESC]
        self._df = self.create_dataframe(self.schema)

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(Stats, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        df = self._df
        prev = df.index[-1]
        prev_max = df.at[prev, self._max_column]
        prev_min = df.at[prev, self._min_column]
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():        
            dfslot.reset()
            prev_min = prev_max = np.nan
            dfslot.update(run_number, input_df)
        indices = dfslot.next_created(step_size) # returns a slice
        logger.debug('next_created returned %s', indices)
        steps = indices_len(indices)
        if steps > 0:
            if isinstance(indices,slice):
                indices=slice(indices.start,indices.stop-1) # semantic of slice with .loc
            x = input_df.loc[indices,self._column]
            row = [np.nanmin([prev_min, x.min()]),
                   np.nanmax([prev_max, x.max()]),
                   run_number]
            with self.lock:
                df.loc[run_number] = row
                if len(df) > self.params.history:
                    self._df = df.loc[df.index[-self.params.history:]]
                if self._reset_index:
                    self._df.index = range(0, len(self._df))
        return self._return_run_step(dfslot.next_state(),
                                     steps_run=steps, reads=steps, updates=len(self._df))
