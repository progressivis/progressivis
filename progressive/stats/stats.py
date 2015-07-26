from progressive.core.common import ProgressiveError
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Stats(DataFrameModule):
    def __init__(self, column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Stats, self).__init__(dataframe_slot='in', **kwds)
        self._column = column
        self.default_step_size = 10000

        index = ['count', 'sum', 'mean', 'min', 'max']
        self._df = pd.DataFrame({'description': [np.nan]
                                 self.UPDATE_COLUMN: [self.EMPTY_TIMESTAMP]},
                                 index=index)

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
        desc = self._df['description']
        desc['count'] += x.count()
        desc['sum']   += x.sum()
        desc['mean']  = desc['sum'] / desc['count'] #TODO improve
        desc['min']   = np.nanmin([desc['min'], x.min()])
        desc['max']   = np.nanmax([desc['max'], x.max()])
        self._df[self.UPDATE_COLUMN] = np.nan  # to update time stamps
        return self._return_run_step(self.state_ready, steps_run=steps, reads=steps, updates=len(self._df))
