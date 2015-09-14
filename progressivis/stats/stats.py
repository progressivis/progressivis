from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Stats(DataFrameModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, column, min_column=None, max_column=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Stats, self).__init__(dataframe_slot='stats', **kwds)
        self._column = column
        self.default_step_size = 10000

        if min_column is None:
            min_column = str(column) + '.min'
        if max_column is None:
            max_column = str(column) + '.max'
        self.min_column = min_column
        self.max_column = max_column
        self.schema = [(self.min_column, np.dtype(float), np.nan),
                       (self.max_column, np.dtype(float), np.nan),
                       DataFrameModule.UPDATE_COLUMN_DESC]
        self._df = self.create_dataframe(self.schema)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Stats, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        df = self._df
        prev = df.index[-1]
        prev_max = df.at[prev, self.max_column]
        prev_min = df.at[prev, self.min_column]
        if not dfslot.update(run_number, input_df):
            dfslot.reset()
            prev_min = prev_max = np.nan
            if not dfslot.update(run_number, input_df):
                raise ProgressiveError('%s module cannot update', self.__class__.__name__)
        indices = dfslot.next_buffered(step_size) # returns a slice
        logger.debug('next_buffered returned %s', indices)
        steps = indices.stop - indices.start
        if steps > 0:
            x = input_df[self._column].iloc[indices]
            df.loc[run_number] = [np.nanmin([prev_min, x.min()]),
                                  np.nanmax([prev_max, x.max()]),
                                  run_number]
            if len(df) > self.params.history:
                self._df = df.loc[df.index[-self.params.history:]]
        return self._return_run_step(dfslot.next_state(),
                                     steps_run=steps, reads=steps, updates=len(self._df))
