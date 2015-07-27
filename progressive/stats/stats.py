from progressive.core.common import ProgressiveError, typed_dataframe
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Stats(DataFrameModule):
    def __init__(self, column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Stats, self).__init__(dataframe_slot='stats', **kwds)
        self._column = column
        self.default_step_size = 10000

        columns = ['count', 'min', 'max'] + [self.UPDATE_COLUMN]
        dtypes = [np.dtype(float)] * len(columns)
        values = [0] + [np.nan] * (len(columns)-1)

        self._df = typed_dataframe(columns, dtypes, values)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Stats, self).is_ready()

    def run_step(self, step_size, howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(self._start_time, input_df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            raise ProgressiveError('%s module does not manage updates or deletes', self.__class__.name)
        dfslot.buffer_created()

        indices = dfslot.next_buffered(step_size)
        steps = len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=steps)
        x = input_df.loc[indices, self._column]
        df = self._df
        df.loc[0, 'count'] += x.count()
        df.loc[0, 'min']   = np.nanmin([df.loc[0, 'min'], x.min()])
        df.loc[0, 'max']   = np.nanmax([df.loc[0, 'max'], x.max()])
        df.loc[0, self.UPDATE_COLUMN] = np.nan  # to update time stamps
        return self._return_run_step(self.state_ready, steps_run=steps, reads=steps, updates=len(self._df))
